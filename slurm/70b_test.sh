#!/bin/bash
#SBATCH --job-name=70b_test
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=70b_test.out

ml anaconda
conda activate llmr
python3 sample.py --config=configs/70b_test.yml
