#!/bin/bash
#SBATCH --job-name=1b_8b_aug=3
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=1b_8b_aug=3.out

ml anaconda
conda activate llmr
python3 router.py --config=configs/router/1b_8b_aug=3.yml