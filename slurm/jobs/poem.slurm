#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home1/sommerer/dev/applied_nlp/Sonnet-Gen/slurm/logs/R-%x-%j.out

conda activate sonnet_gpu
nvcc --version

TITLE="A Blooming Spring Field"

python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/generate_poem.py --title "${TITLE}"