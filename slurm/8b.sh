#!/bin/bash
#SBATCH --job-name=8b
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=8b.out

ml anaconda
conda activate llmr
python3 sample.py --config=configs/8b.yml
