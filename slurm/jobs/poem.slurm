#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --time=1:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/home1/sommerer/dev/applied_nlp/Sonnet-Gen/slurm/logs/R-%x-%j.out

#conda activate sonnet_gpu
source /home1/sommerer/dev/applied_nlp/Sonnet-Gen/sonnet_venv/bin/activate
nvcc --version

#TITLE="A Blooming Spring Field"

#python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/generate_poem.py --title "${TITLE}"

declare -a titles=("Alone in the Subway"
                "Sweet Closure"
                "Watching the Skyline"
                "Running on Empty"
                "Evening in a Forest Glade"
                "Operating Systems"
                "A Pokemon Adventure"
                "The Journey to Mordor"
                )

for title in "${titles[@]}"
do
#   echo "$i"
   # or do whatever with individual element of the array
   python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/generate_poem.py --title "$title"
   python /home1/sommerer/dev/applied_nlp/Sonnet-Gen/generate_poem.py --title "$title" --polish
done


