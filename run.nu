use task.nu

print "running..."

$env.R_SUFFIX = '_T3_L5'

echo "frame,time,window_size\n" | save -f $'out/cu_log($env.R_SUFFIX).csv'
echo "time,window_size\n" | save -f $'out/cu_log_task($env.R_SUFFIX).csv'

def run-task [] {
    task spawn {
        let num = $env.CUDA_OVERRIDE_MAX_SYNC_MS
        $env.CUDA_OVERRIDE_MAX_SYNC_MS = 2
        $env.CUDA_OVERRIDE_SYNC_LOCK_SKIPS = 1 # skip unlocking on the streamsynchronize inside of the task; this simulates a high-priority task which will not be preempted
        cd cannyEdgeDetectorNPP
        # print $'here: ($env.PWD) with ($env.CUDA_OVERRIDE_KERNEL_N_SYNC)'
        let lib = if $num == 3 or $num == 2 { '' } else { '../cuda_override.so' }
        LD_PRELOAD=$lib ./cannyEdgeDetectorNPP
            | lines
            | filter { |x| ($x | str length) > 0 }
            | filter { |x| not ($x | str contains 'Ampere') }
            | each { append $',($num)' | str join }
            | save -a $'../out/cu_log_task($env.R_SUFFIX).csv' --raw
    }
}

for j in 1..15 {
    print $"-- running control task \(($j)/15\) --"

    # Uses 2 as the marker for "control" (not running concurrently with pytorch)
    $env.CUDA_OVERRIDE_MAX_SYNC_MS = 2
    let id = run-task
    sleep 15sec
    task kill $id
    task wait $id
    task remove $id

    # there's also another type of control which is where both are running but the locking code is not inserted
    # that can be 3
    print 'running control task 2'
    $env.CUDA_OVERRIDE_MAX_SYNC_MS = 3
    let id = run-task
    venv/bin/python cutest.py
            | from csv -n
            | rename frame time
            | filter { |x| $x.frame != 0 }
            | insert window_size 3
            | if $j mod 2 == 0 { reverse } else { echo $in }
            | to csv -n
            | save --append $'out/cu_log($env.R_SUFFIX).csv'
    task kill $id
    task wait $id
    task remove $id

    # i want to try with 11.5 and 12.5 (μs) or something - there's a jump in the kernel time CDF at 12 and another at 13.2-13.6 so check around there too. maybe just 11,12,14 μs?
    # realistically this pry won't be very interesting, like, we don't actually have that fine-grained control over critical section length
    # i also want to know what's going on between 0.05 and 0.2, there's a big jump there. ig not really that much, but enough to like care about
    for i in [0.001, 0.010, 0.0125, 0.014, 0.05, 0.08, 0.1, 0.2, 4] {
        print $'running window size ($i)'

        # $env.CUDA_OVERRIDE_KERNEL_N_SYNC = $i
        $env.CUDA_OVERRIDE_MAX_SYNC_MS = $i

        let id = run-task
        print $'id: ($id)'

        # 3,4,5,7 are actually KERNEL_N_SYNC 1,8,27,125
        # if $i > 2.0 {
        #     $env.CUDA_OVERRIDE_MAX_SYNC_MS = 0
        #     $env.CUDA_OVERRIDE_KERNEL_N_SYNC = ($i - 2) ** 3
        # }
        if $i == 4 {
        	$env.CUDA_OVERRIDE_MAX_SYNC_MS = 0
        	$env.CUDA_OVERRIDE_KERNEL_N_SYNC = 1
        }

        LD_PRELOAD="./cuda_override.so /usr/lib/libstdc++.so" venv/bin/python cutest.py
                | from csv -n
                | rename frame time
                | filter { |x| $x.frame != 0 }
                | insert window_size $i
                | if $j mod 2 == 0 { reverse } else { echo $in }
                | to csv -n
                | save --append $'out/cu_log($env.R_SUFFIX).csv'

        # print (task status)
        task kill $id
        task wait $id
        task remove $id
    }
}

venv/bin/python plot2.py
