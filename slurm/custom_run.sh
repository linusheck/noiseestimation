#!/usr/bin/fish

for i in (seq 0 14)
    SLURM_ARRAY_TASK_ID=$i bash run_no_slurm.sh others/fitzhugh/noise_penalty.sh >> out.txt 2>&1
end
