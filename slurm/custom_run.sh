#!/usr/bin/fish

for i in (seq 1 6)
    SLURM_ARRAY_TASK_ID=$i bash run_no_slurm.sh others/noise_penalty_triple.sh >> out.txt 2>&1
end

# SLURM_ARRAY_TASK_ID=1 bash run_no_slurm.sh energybalance/noise_penalty_extreme_betas.sh >> out.txt 2>&1
