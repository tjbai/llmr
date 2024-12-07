
#!/bin/bash

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <config_path> <model_file> [-r]"
    exit 1
fi

config_path="$1"
model_file="$2"
job_name=$(basename "$config_path" .yml)

cat << EOF > "slurm/${job_name}.sh"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time=8:00:0
#SBATCH --output=$job_name.out

ml anaconda
conda activate llmr
python3 $model_file --config=$config_path --device=cuda --wandb
EOF

echo "generated: ${job_name}.sh"

if [ "$3" = "-r" ]; then
    sbatch "slurm/${job_name}.sh"
fi
