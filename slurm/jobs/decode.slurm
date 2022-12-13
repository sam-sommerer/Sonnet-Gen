#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home1/sommerer/dev/applied_nlp/Sonnet-Gen/slurm/logs/R-%x-%j.out

#conda activate sonnet_gpu
source /home1/sommerer/dev/applied_nlp/Sonnet-Gen/sonnet_venv/bin/activate
nvcc --version
#module load cuda/11.7
#module purge
#module load conda
#module load nvidia-hpc-sdk
#module load cuda/10.2
#nvcc --version
python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/decode.py
