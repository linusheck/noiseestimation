#!/bin/bash
fixed_args="multistablesde/latent_sde.py --latent-size=1 --lr_gamma=0.9995 --model=energyconstant --kl-anneal-iters=1000 --data-noise-level=25 --num-iters 10000 --beta 10"

variable_args=(
    "--noise-penalty=0"
    "--noise-penalty=170"
    "--noise-penalty=175"
    "--noise-penalty=180"
    "--noise-penalty=150"
    "--noise-penalty=155"
    "--noise-penalty=145"
    "--noise-penalty=160"
    "--noise-penalty=165"
    "--noise-penalty=140"
)

srun python3 -m pipenv run python $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]} --train-dir $1
