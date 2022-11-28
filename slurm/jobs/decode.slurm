#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=5:00
#SBATCH --output=/home1/sommerer/dev/applied_nlp/Sonnet-Gen/slurm/logs/R-%x-%j.out

module purge
module load conda
module load nvidia-hpc-sdk

conda activate sonnet

python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/decode.py