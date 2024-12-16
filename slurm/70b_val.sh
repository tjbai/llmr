#!/bin/bash
#SBATCH --job-name=70b_val
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=70b_val.out

ml anaconda
conda activate llmr
python3 sample.py --config=configs/sample/70b_val.yml
