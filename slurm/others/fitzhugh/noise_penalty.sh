#!/bin/bash
fixed_args="multistablesde/latent_sde.py --latent-size=2 --lr_gamma=0.9997 --model=fitzhughgamma --kl-anneal-iters=1000 --num-iters=10000 --beta=10"

variable_args=(
    "--noise-penalty=0"
    "--noise-penalty=50"
    "--noise-penalty=100"
    "--noise-penalty=150"
    "--noise-penalty=200"
    "--noise-penalty=250"
    "--noise-penalty=300"
    "--noise-penalty=350"
    "--noise-penalty=400"
    "--noise-penalty=450"
    "--noise-penalty=500"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
