#!/bin/bash
#SBATCH --job-name=llm_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load any necessary modules
module load gcc python openmpi py-torch 

# Activate virtual environment if needed
source .venv/bin/activate 

# Run the script
python src/run_inference.py \
    --model_name "EleutherAI/pythia-2.8b" \
    --prompt "prompts.txt" \
    --max_length 200 \
    --output_file "output.txt"
