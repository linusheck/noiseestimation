srun() {
    exec "$@"
}
export -f srun
cd ..
./slurm/$1 artifacts/$(date +%s)
