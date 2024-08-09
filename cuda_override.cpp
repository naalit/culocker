#include "cuda_runtime_api.h"
#include "cuda.h"
#include <iostream>
#include <dlfcn.h>
#include <sys/file.h>
#include <cstring>
#include <unordered_map>
#include <nvtx3/nvToolsExt.h>
#include <semaphore.h>
#include <assert.h>
#include <cstdlib>

// compile: g++ -fPIC -DPIC -c cudart_override.cpp -I/opt/cuda/include -fno-use-cxa-atexit -std=c++20
// link:    ld -shared -o cudart_override.so cudart_override.o -ldl
// then just add to LD_PRELOAD and it should work!

// NOTES:
// DtH cudaMemcpyAsync seems to not actually be async (otherwise cudaMemcpyAsync is properly async though)
// -> so actually it's only sync when pageable memory is used, it's async when pinned memory is used; and there's no way to detect pinned memory with no false negativesf that I've found.
//   so we have to pessimistically assume it's async all the time 🫤
//
// It seems like with green contexts if we wanted to do SM partitioning per process we could make a green context with the specified SMs,
//   then hook into cuCtxSetCurrent or whatever and replace the context with cuCtxFromGreenCtx on our green context
// I'm curious to see if using green contexts like this actually allows two processes to execute not timesliced? pry not?
// Could probably get gpu-microbench to test that, I'll try after lunch I think.
// Damn nope even with the green ctx they seem to run timesliced. Not that I'm actually sure the green ctx stuff is like working.
//
// In other news, Linux POSIX semaphores seem to be FIFO, based on looking at the implementation
// - it uses a spinlock for atomic access to a waitlist which is FIFO
//   and when we unlock and the waitlist is nonempty, we *don't* increase the count (so nobody new can access) and we set an `up` flag on the list entry that we're waking up,
//   so there's no way for things to end up out-of-order.
//
// oookay so the POSIX semaphores are implemented in glibc on top of futexes that aren't FIFO
// BUT the sysv semaphores go directly to the the kernel and actually are FIFO
// it's weird, the kernel semaphores in semaphore.c are apparently not available from userspace at all?
// like idk maybe they're used to implement something else in the kernel
// but sysv semaphores have their own implementation (still FIFO) and posix semaphores are in glibc on futexes. idk

const bool LOG_KERNELS = false;
const bool LOCK_KERNELS = true;
const bool LOG_LOCKS = false;
const bool LOG_MEMORY = false;
const bool LOG_PROC_ADDR = false;
const uint32_t KERNEL_N_SYNC = 10; // can be overriden by CUDA_OVERRIDE_KERNEL_N_SYNC environment variable
uint32_t _kernel_launches = 0;
uint32_t _sync_calls = 0;

// This needs to be after those definitions since it uses the logging flags
#include "mem_functions.hpp"

// Profiling with nvtx
void rangePush(const char* const name) {
	nvtxEventAttributes_t eventAttrib = {0};
	eventAttrib.version = NVTX_VERSION;
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
	nvtxRangePushEx(&eventAttrib);
}
void rangePop() {
	nvtxRangePop();
}

// -- SysV semaphores -- //
// much worse to use, but they have FIFO guarantees 🤷
// (well, the Linux implementation does; it doesn't seem like the SysV standard actually requires that)
#include <sys/ipc.h>
#include <sys/sem.h>

#define SEM_KEY 57992

union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

int create_semaphore() {
    int semid = semget(SEM_KEY, 1, IPC_CREAT | 0666);
    if (semid == -1) {
        perror("semget");
        exit(1);
    }
    
    union semun arg;
    arg.val = 1;  // Initialize semaphore value to 1
    if (semctl(semid, 0, SETVAL, arg) == -1) {
        perror("semctl");
        exit(1);
    }
    
    return semid;
}

void sem_waitv(int semid) {
    struct sembuf sb = {0, -1, 0};
    if (semop(semid, &sb, 1) == -1) {
        perror("semop");
        exit(1);
    }
}

void sem_postv(int semid) {
    struct sembuf sb = {0, 1, 0};
    if (semop(semid, &sb, 1) == -1) {
        perror("semop");
        exit(1);
    }
}

int sem_trywaitv(int semid) {
    struct sembuf sb = {0, -1, IPC_NOWAIT};
    if (semop(semid, &sb, 1) == -1) {
        if (errno == EAGAIN) {
            return 0;  // Semaphore is locked
        } else {
            perror("semop");
            exit(1);
        }
    }
    return 1;  // Successfully acquired the semaphore
}
// -- end SysV semaphores

// Semaphores!
bool holds_lock = false;
//sem_t *semaphore; // 1 if unlocked, 0 if locked
int semaphore;
void lock() {
    assert(!holds_lock);
    if (!semaphore) {
        semaphore = create_semaphore();//sem_open("/cuda-override-sem", O_CREAT, S_IRWXU, 1);
    }
    if (!sem_trywaitv(semaphore)) {
        //std::cout << "waiting" << std::endl;
        rangePush("wait");
        sem_waitv(semaphore);
        rangePop();
    }
    if constexpr (LOG_LOCKS)
        std::cout << "locked" << std::endl;
    rangePush("lock");
    holds_lock = true;
}
void unlock() {
    assert(holds_lock);
    if constexpr (LOG_LOCKS)
        std::cout << "unlocked" << std::endl;
    rangePop();
    sem_postv(semaphore);
    holds_lock = false;
}

// -- timing stuff --
#include <stddef.h>
#include <sys/resource.h>
#include <sys/time.h>
static double millisecond(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
// -- --

// Original function pointers
static void* (*original_dlsym)(void *handle, const char *symbol) = NULL;
uint32_t kernel_n_sync;
uint32_t sync_n_lock;
double max_sync_ms;
double next_sync_ms = 0;

// Constructor function to initialize original function pointers
__attribute__((constructor)) void init() {
    original_dlsym = (void* (*)(void*, const char*)) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");

    kernel_n_sync = KERNEL_N_SYNC;
    max_sync_ms = 0;
    sync_n_lock = 0;
    const char *s = getenv("CUDA_OVERRIDE_KERNEL_N_SYNC");
    if (s != NULL && (strcmp(s, "0") == 0 || atoi(s) != 0)) // allow a zero value for KERNEL_N_SYNC (syncing every kernel launch)
        kernel_n_sync = atoi(s);
    s = getenv("CUDA_OVERRIDE_SYNC_LOCK_SKIPS");
    if (s != NULL && atoi(s) != 0)
        sync_n_lock = atoi(s);
    s = getenv("CUDA_OVERRIDE_MAX_SYNC_MS");
    if (s != NULL && strtod(s, nullptr) != 0)
        max_sync_ms = strtod(s, nullptr);
}


CUresult CUDAAPI (*real_cuGetProcAddress)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
CUresult CUDAAPI (*real_cuLaunchKernel)(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra);
CUresult CUDAAPI (*real_cuLaunchKernelEx)(const CUlaunchConfig *config,
                                  CUfunction f,
                                  void **kernelParams,
                                  void **extra);
CUresult CUDAAPI (*real_cuLaunchCooperativeKernel)(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams);
CUresult CUDAAPI (*real_cuGraphLaunch)(CUgraphExec hGraphExec, CUstream hStream);
CUresult CUDAAPI (*real_cuStreamSynchronize) (CUstream hStream);
CUresult CUDAAPI (*real_cuStreamSynchronize_v2) (CUstream hStream);

static void _doSync() {
    if (sync_n_lock > 0) {
        if (_sync_calls++ < sync_n_lock) {
            return;
        } else {
            _sync_calls = 0;
        }
    }
    if (LOCK_KERNELS && holds_lock) {
        unlock();
    }
    _kernel_launches = 0;
}
static void _auxSync(CUstream hStream) {
    if (max_sync_ms > 0 ? millisecond() > next_sync_ms : ++_kernel_launches > kernel_n_sync) {
        if (max_sync_ms > 0)
            next_sync_ms = millisecond() + max_sync_ms;
        rangePush("auxSync");
        // This shows up as part of the kernel launch in Nsight, pry because we're still in the cudart kernel launch function
        // (hence the nvtx range to identify it)
        real_cuStreamSynchronize(hStream);
        rangePop();
        _doSync();
        if (!holds_lock)
            lock();
    }
}

static void _doKLaunch(CUstream hStream) {
    if (LOCK_KERNELS && !holds_lock) {
        lock();
    }
    _auxSync(hStream);
}

// Called by memory ops AFTER the memory operation
static void _doMemOpPost(bool is_async) {
    if (!is_async) {
        // this counts as a cuStreamSynchronize, so we can release locks - the GPU is empty
        _doSync();
    }
}
// Called by memory ops BEFORE the memory operation
static void _doMemOpPre(bool is_async, CUstream hStream) {
    if (LOCK_KERNELS && !holds_lock) {
        lock();
    }
    // for now, treat kernels and async memory operations the same way
    if (is_async) {
        _auxSync(hStream);
    }
}

CUresult CUDAAPI fake_cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    if constexpr (LOG_KERNELS)
        std::cout << "DRIVER-INJECT: KERNEL LAUNCH" << std::endl;
    _doKLaunch(hStream);
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
CUresult CUDAAPI fake_cuLaunchKernelEx(const CUlaunchConfig *config,
                                  CUfunction f,
                                  void **kernelParams,
                                  void **extra) {
    if constexpr (LOG_KERNELS)
        std::cout << "DRIVER-INJECT: KERNEL LAUNCH EX" << std::endl;
    _doKLaunch(config->hStream);
    return real_cuLaunchKernelEx(config, f, kernelParams, extra);
}
CUresult CUDAAPI fake_cuLaunchCooperativeKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams) {
    if constexpr (LOG_KERNELS)
        std::cout << "DRIVER-INJECT: KERNEL LAUNCH COOP" << std::endl;
    _doKLaunch(hStream);
    return real_cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}
CUresult CUDAAPI fake_cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
    if constexpr (LOG_KERNELS)
        std::cout << "DRIVER-INJECT: KERNEL LAUNCH GRAPH" << std::endl;
    _doKLaunch(hStream);
    return real_cuGraphLaunch(hGraphExec, hStream);
}
CUresult CUDAAPI fake_cuStreamSynchronize (CUstream hStream) {
    auto r = real_cuStreamSynchronize(hStream);
    _doSync();
    return r;
}
CUresult CUDAAPI fake_cuStreamSynchronize_v2 (CUstream hStream) {
    auto r = real_cuStreamSynchronize_v2(hStream);
    _doSync();
    return r;
}


std::unordered_map<std::string, std::pair<void*, void**>> MAP = {
    MEM_OPS_MAP
    {"cuLaunchKernel", {(void*) fake_cuLaunchKernel, (void**)&real_cuLaunchKernel}},
    {"cuLaunchKernelEx", {(void*) fake_cuLaunchKernelEx, (void**)&real_cuLaunchKernelEx}},
    {"cuLaunchCooperativeKernel", {(void*) fake_cuLaunchCooperativeKernel, (void**)&real_cuLaunchCooperativeKernel}},
    {"cuGraphLaunch", {(void*) fake_cuGraphLaunch, (void**)&real_cuGraphLaunch}},
    {"cuStreamSynchronize", {(void*) fake_cuStreamSynchronize, (void**)&real_cuStreamSynchronize}}
};

CUresult CUDAAPI fake_cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    if constexpr (LOG_PROC_ADDR)
        std::cout << "HOOK: cuGetProcAddress: '" << symbol << "' v" << cudaVersion << std::endl;
    if (strcmp(symbol, "cuGetProcAddress") == 0) {
        *pfn = (void*) fake_cuGetProcAddress;
        return CUDA_SUCCESS;
    }
    // As far as I can tell this is the only function where the version matters? But it does matter here, pytorch crashes without this
    if (strcmp(symbol, "cuStreamSynchronize") == 0 && cudaVersion == 2000) {
        *pfn = (void*) fake_cuStreamSynchronize_v2;
        return real_cuGetProcAddress(symbol, (void**)&real_cuStreamSynchronize_v2, cudaVersion, flags, symbolStatus);
    }
    if (auto search = MAP.find(symbol); search != MAP.end()) {
        *pfn = search->second.first;
        return real_cuGetProcAddress(symbol, search->second.second, cudaVersion, flags, symbolStatus);
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

// Hooked dlsym
void* dlsym(void *handle, const char *symbol) {
    if (!original_dlsym) {
        // I believe this is only needed when running in Nsight, bc otherwise the constructor actually runs
        init();
        // original_dlsym = (void* (*)(void*, const char*)) dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!original_dlsym) {
            fprintf(stderr, "Error in `dlsym` hook: original function not found.\n");
            exit(1);
        }
    }
    if constexpr (LOG_PROC_ADDR)
        printf("dlsym called with: %s\n", symbol);
    // Why is it cuGetProcAddress_v2 for dlsym and cuGetProcAddress for cuGetProcAddress? unclear. why call cuGetProcAddress to get cuGetProcAddress at all when you clearly already have it? again, who can tell.
    // ig maybe the versions are different? in which case this stuff is all ignoring that and unifying the versions. but like, we only have one version of cuda installed; i'm sure it's fine...
    if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        real_cuGetProcAddress = (CUresult CUDAAPI (*)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus)) original_dlsym(handle, symbol);
        return (void*) fake_cuGetProcAddress;
    }
    if (auto search = MAP.find(symbol); search != MAP.end()) {
        // ?? rely on cuGetProcAddress being called first ??
        // it did work in the pytorch setting ...
        return (void*) search->second.first;
    }
    return original_dlsym(handle, symbol);
}

