use task.nu

print "running..."

$env.R_SUFFIX = '_T2_L2'

echo "frame,time,window_size\n" | save -f $'out/cu_log($env.R_SUFFIX).csv'
echo "time,window_size\n" | save -f $'out/cu_log_task($env.R_SUFFIX).csv'

let start = date now

def run-task [] {
    task spawn {
        let num = $env.CUDA_OVERRIDE_MAX_SYNC_MS
        $env.CUDA_OVERRIDE_MAX_SYNC_MS = 2
        $env.CUDA_OVERRIDE_SYNC_LOCK_SKIPS = 1
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

    # Uses 3 as the marker for "control" (not running concurrently with pytorch)
    $env.CUDA_OVERRIDE_MAX_SYNC_MS = 3
    let id = run-task
    sleep 15sec
    task kill $id
    task wait $id
    task remove $id

    for i in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 2.0] { #[1,3,5,8,10,15,20,30,50,100,200,300,500,700,1000]
        print $'running window size ($i)'

        # $env.CUDA_OVERRIDE_KERNEL_N_SYNC = $i
        $env.CUDA_OVERRIDE_MAX_SYNC_MS = $i

        let id = run-task
        print $'id: ($id)'

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

# looking like 100 is about the place to be?
# but this data is super noisy
# could i do 10 runs and then get a median at each frame?
# would i do that here in the nu or in the python? pry python? idk how to do that ig but i can try
# wait box plots have median already... but might be nice to have per-frame median too idk?
