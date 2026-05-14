#!/bin/sh
#SBATCH -J V0.8
#SBATCH -p normal
#SBATCH -N 10
#SBATCH -n 50
#SBATCH -t 48:00:00
#SBATCH -e job.err
#SBATCH -o job.out
#SBATCH --mail-type=all
#SBATCH --mail-user=PSHAR50@emory.edu
#SBATCH -A DMR21001
#SBATCH -V
cd $SLURM_SUBMIT_DIR

module load intel impi

rm job.*

export OMP_PROC_BIND=true
export OMP_PLACES=threads
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/00434/eijkhout/arpack/installation-3.7.0-intel/lib64
##export OMP_NUM_THREADS=28
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#export JULIA_NUM_THREADS=1

mpiexec -n 50 pair_correlation.jl input_$1.toml > out_L100_Vn0.8_input_$1.txt 2>&1
wait
