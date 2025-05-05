#!/bin/bash

# Variables you can easily change
MODEL_NAME="EleutherAI/pythia-6.9b-deduped"
PROMPT="Obdurodon dicksoni\n\nObdurodon dicksoni is an extinct species of ornithorhynchid monotreme discovered in Australia." 
MAX_LENGTH=400
N_GEN=5
TEMP=0.7
TOP_K=20
TOP_P=0.8
OUTPUT_DIR="outputs/perplexity/prompts_wiki"
BASE_NAME="pythia-6.9b-platypus-2"
PERPLEXITY_THRESHOLD=1.0

# Generate output filenames with parameters included
OUTPUT_BASENAME="${BASE_NAME}_temp${TEMP}_topk${TOP_K}_topp${TOP_P}"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_BASENAME}.txt"
JSON_OUTPUT="${OUTPUT_BASENAME}_results.json"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the command
sbatch run_scripts/run_perplexity.sh \
  --model_name "$MODEL_NAME" \
  --prompt "$PROMPT" \
  --max_length $MAX_LENGTH \
  --n_gen=$N_GEN \
  --temp $TEMP \
  --top_k $TOP_K \
  --top_p $TOP_P \
  --output_file "$OUTPUT_FILE" \
  --perplexity_threshold=$PERPLEXITY_THRESHOLD \
  --json_output="$JSON_OUTPUT" \

echo "Job submitted with output file: $OUTPUT_FILE"
echo "JSON results will be saved to: $JSON_OUTPUT"
