# The multistablesde package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14534738.svg)](https://doi.org/10.5281/zenodo.14534738)

This package contains all code and experiments from the paper "Improving the Noise Estimation of Latent Neural Stochastic Differential Equations".

# Experiments in the paper

For all experiments, the scripts to execute them (see below how), as well as the results, including the training process, the full resulting models, and their analysis files, are available.

- Hyperparameter search on EBM (section 5):
    - Execute: `energybalance/betas_const.sh`
    - Archived Results: `results/betas_const`
- Varying constant noise level on EBM (section 6):
    - Execute: `energybalance/noise_level_const.sh`
    - Archived Results: `results/noise_level_const`
- Noise penalty on EBM (section 7):
    - Execute: `energybalance/noise_penalty_const.sh`
    - Archived Results: `results/noise_penalty_const`
- FitzHugh-Nagumo model (Appendix A):
    - Execute: `others/fitzhugh/noise_penalty.sh`
    - Archived Results: `results/fhn_noise`
- Hyperparameter search on OU process (Appendix B):
    - Execute: `ornstein/noise_penalty.sh`
    - Archived Results: `results/ornstein_noise_penalty`
- Hyperparameter search on EBM with linear diffusion (Appendix C):
    - Execute: `energybalance/noise_penalty_linear.sh`
    - Archived Results: `results/noise_penalty_linear`

# Running the experiments

Install `pipenv` and run `pipenv install` to install the dependencies and the package. If this doesn't work, try deleting `Pipfile.lock`, as your system might require different dependencies for `pytorch`.

## Using Slurm

The experiments are included as `slurm` array jobs. Each parametrization of a parameter sweep is a job inside of the array. You can run them as follows:

    cd slurm
    # Edit "account" and "output" folders and other slurm variables you need to tweak according to your cluster config
    vim run.sh
    sbatch --array=0-<number of jobs> run.sh <file>

Example:

    cd slurm
    sbatch --array=0-7 run.sh energybalance/betas_const.sh

## Without Slurm

You can run single experiments without using slurm, but you need to specify which experiment you want to run using an environment variable. For instance, looking into `energybalance/betas_const.sh`:

    variable_args=(
        "--beta=0.01"
        "--beta=0.1"
        "--beta=1"
        "--beta=10"
    ...

Say we want to run the experiment with $\beta=10$, which is the 3rd (zero-indexed) parametrization, run:

    SLURM_ARRAY_TASK_ID=3 ./run_no_slurm.sh energybalance/betas_const.sh

# Analyse Results

You can analyse the results by identifying the correct folder in `~/artifacts` and running `analyze.sh`:

## Using Slurm

    cd slurm
    # for matplotlib pdfs
    sbatch analyze.sh ~/artifacts/<output folder>/
    # for pgfs (latex)
    sbatch analyze_pgf.sh ~/artifacts/<output folder>/

## Without Slurm

    # for matplotlib pdfs
    python multistablesde/analyze.py -f $1
    # for pgfs (latex)
    python multistablesde/analyze.py -f $1 --pgf
