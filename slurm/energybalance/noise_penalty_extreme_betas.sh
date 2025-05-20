#!/bin/bash
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energyconstant --kl-anneal-iters=1000 --data-noise-level=40 --num-iters 10000 --noise-penalty=0"

variable_args=(
    "--beta=0"
    "--beta=10000000000"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
