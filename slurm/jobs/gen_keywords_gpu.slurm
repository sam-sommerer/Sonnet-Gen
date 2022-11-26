#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB
#SBATCH --time=5:00
#SBATCH --output=/home1/sommerer/dev/applied_nlp/Sonnet-Gen/slurm/logs/R-%x-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

conda activate sonnet

python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/keyword/inference_bart_keywords_gen.py

