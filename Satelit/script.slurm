#!/bin/bash
#SBATCH --job-name="stochastic-satelit"
#SBATCH --output=schocastic-satelit.out
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --partition=compute-p2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=Education-EEMCS-Courses-CSEMinor

module load 2024r1
module load py-numpy
module load py-matplotlib

srun python cod.py