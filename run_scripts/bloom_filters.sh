#!/bin/bash
#SBATCH --job-name=data_portraits
#SBATCH --output=data_portraits_%j.out
#SBATCH --error=data_portraits_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load any necessary modules (adjust as needed for your cluster)
module load python/3.9
module load cuda/11.7

# Create and activate a virtual environment (optional but recommended)
source .venv/bin/activate

# Set environment variables for huggingface
export HF_HOME=$SCRATCH/huggingface

# Run the script with setup first
python src/data_portraits.py --setup 

echo $HF_TOKEN | huggingface-cli login --token stdin

# Run the main analysis
python src/data_portraits.py --sample

# Deactivate the virtual environment
deactivate

echo "Job completed successfully!"