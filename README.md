# CUDA wrapper library for locking accesses to a shared GPU

A `libcuda` wrapper that uses locking to manage access to a GPU from multiple parallel processes.

## Building

Build with `make build`. The path to the CUDA installation is by default assumed to be `/usr/local/cuda`, but it can be overriden using the `CUDA_PATH` environment variable.

## Usage

The wrapper library is intended to be added to `LD_PRELOAD` so that it will load before `libcuda`.
It uses the C++ standard library, so if the program being run doesn't already link to the C++ standard library it might be necessary to add `/usr/lib/libstdc++.so` or wherever it is located to the `LD_PRELOAD` as well.

The wrapper library can be configured by a set of environment variables:
- `CUDA_OVERRIDE_KERNEL_N_SYNC` (default 10): the number of kernel launches or async memory ops before we force a `cuStreamSynchronize` and unlock the GPU lock
- `CUDA_OVERRIDE_MAX_SYNC_MS` (default 0): if non-zero, instead of sync/lock intervals being based on kernel launches, we sync and unlock when we've held the lock for more than this many milliseconds
- `CUDA_OVERRIDE_ALWAYS_LOCK` (default 0): can be set to 1 to lock before every kernel launch or async memory op and then sync and unlock after, replacing the previous two modes
- `CUDA_OVERRIDE_SYNC_LOCK_SKIPS` (default 0): by default we unlock every time a `cuStreamSynchronize` is executed, whether by the program or by the wrapper library. This option allows keeping the lock for the first N sync calls. It's not generally recommended, but is used by the experiment to simulate a higher-priority task that never yields to the lower-priority task

## Running the experiment

The experiment can be run with `make run` (which just runs `nu run.nu` after building everything). You'll need a few dependencies:
- [Nushell](https://www.nushell.sh/) to run the script (this can be easily installed by `cargo` if you already have a Rust toolchain installed: `cargo install nu`)
- [Pueue](https://github.com/Nukesor/pueue) to manage parallel tasks (also can be installed by `cargo install pueue`)
- The script assumes you have a Python virtual environment in `./venv` which contains recent versions of the `transformers`, `torch`, `torchvision`, `timm`, and `pillow` packages
  - The script just runs `./venv/bin/python`, if you have these packages installed system-wide feel free to `ln -s / venv` or `mkdir -p venv/bin && ln -s /bin/python venv/bin`
