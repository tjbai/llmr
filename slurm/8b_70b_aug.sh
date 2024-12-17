#!/bin/bash
#SBATCH --job-name=8b_70b_aug
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=8b_70b_aug.out

ml anaconda
conda activate llmr
python3 router.py --config=configs/router_aug/8b_70b_aug.yml --train
