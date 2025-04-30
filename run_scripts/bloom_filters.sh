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
python -m venv data_portraits_env
source data_portraits_env/bin/activate

# Set environment variables for huggingface
export HF_HOME=$SCRATCH/huggingface

# Run the script with setup first
python data_portraits.py --setup

# Log in to Hugging Face (you may need to set up token authentication)
# Replace YOUR_HF_TOKEN with your actual token
export HF_TOKEN=YOUR_HF_TOKEN
echo $HF_TOKEN | huggingface-cli login --token stdin

# Run the main analysis
python data_portraits.py \
  --input "/path/to/prompts_morphine/*.txt" \
  --output "./results" \
  --exclude "ethyl"

# Deactivate the virtual environment
deactivate

echo "Job completed successfully!"