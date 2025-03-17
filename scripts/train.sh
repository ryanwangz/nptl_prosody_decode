#!/bin/bash
#SBATCH --job-name=nn_training
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH -p owners,henderj
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# SBATCH --gres=gpu:1
# Load necessary modules
module load python/3.9.0
module load cuda/11.7.0
ml cudnn/8.6.0.163
ml gcc/10.1.0
ml ffmpeg
ml cmake

# Activate your virtual environment if you have one
source /home/groups/henderj/rzwang/vscode_env/bin/activate

# Create logs directory
mkdir -p logs

# Run the training script with parameters
python test_script_decoding.py \
    --window_size 5 \
    --stride 1 \
    --batch_size 16 \
    --n_epochs 50 \
    --learning_rate 0.001 \
    --train_split 0.8 \
    --output_dir results