use task.nu

print "running..."

$env.R_SUFFIX = '_T3_L2'

echo "frame,time,window_size\n" | save -f $'out/cu_log($env.R_SUFFIX).csv'
echo "time,window_size\n" | save -f $'out/cu_log_task($env.R_SUFFIX).csv'

def run-task [] {
    task spawn {
        let num = $env.CUDA_OVERRIDE_MAX_SYNC_MS
        $env.CUDA_OVERRIDE_MAX_SYNC_MS = 2
        $env.CUDA_OVERRIDE_SYNC_LOCK_SKIPS = 1 # skip unlocking on the streamsynchronize inside of the task; this simulates a high-priority task which will not be preempted
        cd cannyEdgeDetectorNPP
        # print $'here: ($env.PWD) with ($env.CUDA_OVERRIDE_KERNEL_N_SYNC)'
        LD_PRELOAD="../cuda_override.so" ./cannyEdgeDetectorNPP
            | lines
            | filter { |x| ($x | str length) > 0 }
            | filter { |x| not ($x | str contains 'Ampere') }
            | each { append $',($num)' | str join }
            | save -a $'../out/cu_log_task($env.R_SUFFIX).csv' --raw
    }
}

for j in 1..15 {
    print 'running control task'

    # Uses 2.5 as the marker for "control" (not running concurrently with pytorch)
    $env.CUDA_OVERRIDE_MAX_SYNC_MS = 2.5
    let id = run-task
    sleep 15sec
    task kill $id
    task wait $id
    task remove $id

    for i in [0.001, 0.01, 0.05, 0.2, 0.8, 1.0] {
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
