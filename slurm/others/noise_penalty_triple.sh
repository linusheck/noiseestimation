#!/bin/bash
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=triplewell --kl-anneal-iters=1000 --data-noise-level=2.5 --num-iters 10000 --beta 100"

variable_args=(
    "--noise-penalty=190"
    "--noise-penalty=200"
    "--noise-penalty=210"
    "--noise-penalty=220"
    "--noise-penalty=230"
    "--noise-penalty=240"
    "--noise-penalty=250"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
