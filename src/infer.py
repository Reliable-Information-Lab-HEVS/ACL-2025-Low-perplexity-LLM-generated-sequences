import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
from utils import *
import time

PROMPT_ID_DIGITS = 4  # Number of digits to represent the prompt ID, e.g., 0001, 0002, etc.
GEN_ID_DIGITS = 2  # Number of digits to represent the generation ID, e.g., 01, 02, etc.
REGION_ID_DIGITS = 2  # Number of digits to represent the region ID, e.g., 01, 02, etc.
def main():
    parser = argparse.ArgumentParser(
        description="Generate text and compute perplexity per token"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Input prompt for generation"
    )
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument("--top_k", type=int, default=20, help="top_k for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="top_p for sampling")
    parser.add_argument("--penalty", type=float, default=1, help="Repetition penalty for sampling")
    parser.add_argument(
        "--n_gen", type=int, default=1, help="Number of generations to perform"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_perplexity.txt",
        help="Output file name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="perplexity_experiment",
        help="Name of the experiment for organizing outputs",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=6,
        help="Size of the window that contains the low perplexity tokens",
    )
    parser.add_argument(
        "--perplexity_threshold",
        type=float,
        default=1/0.9,
        help="Threshold for low perplexity in finding longest sequence",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window in low perplexity region extraction",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default="perplexity_results.json",
        help="JSON file to store perplexity results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging and detailed logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility, none for no seed"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose, args.experiment_name)
    logger.info("Starting perplexity analysis experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed for reproducibility if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set random seed to {args.seed}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
        device= model.device
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Handle prompt input
    if args.prompt.endswith('.json'):
        logger.info(f"Loading prompts from JSON file: {args.prompt}")
        try:
            with open(args.prompt, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
            prompts = prompt_data.get('prompts', [])
            if not prompts:
                prompts = prompt_data.get('variants', [])
            if not prompts:
                raise ValueError("No prompts found in the JSON file.")
            logger.info(f"Loaded {len(prompts)} prompts from file")
        except Exception as e:
            logger.error(f"Failed to load prompts from JSON: {e}")
            raise
    else:
        prompts = [args.prompt]
        logger.info("Using single prompt from command line")
    
    # Setup experiment directories
    experiment_dir = os.path.join(os.getcwd(), args.experiment_name)
    generation_dir = os.path.join(experiment_dir, 'inference_data/generations')
    perp_regions_dir = os.path.join(experiment_dir, 'perplexity_analysis/low_perp_regions')
    
    # Create directories
    os.makedirs(generation_dir, exist_ok=True)
    os.makedirs(perp_regions_dir, exist_ok=True)
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Determine starting prompt ID
    existing_files = [f for f in os.listdir(generation_dir) if f.startswith('prompt_') and f.endswith('.json')]
    prompt_base_id = len(existing_files)
    logger.info(f"Starting with prompt base ID: {prompt_base_id}")
    
    # Validate consistency
    existing_perp_files = [f for f in os.listdir(perp_regions_dir) if f.startswith('prompt_') and f.endswith('_low_perp.json')]
    if len(existing_perp_files) != prompt_base_id:
        logger.warning(f"Mismatch in existing files: {len(existing_files)} generation files vs {len(existing_perp_files)} perplexity files")
    
    # Process each prompt
    for i_prompt, prompt in enumerate(prompts):
        prompt_id = prompt_base_id + i_prompt
        prompt_id_str = f"P{prompt_id:0{PROMPT_ID_DIGITS}d}"
        
        logger.info(f"Processing prompt {prompt_id_str} ({i_prompt + 1}/{len(prompts)})")
        logger.debug(f"Prompt text: {prompt[:100]}...")
        
        # File paths
        generation_file = os.path.join(generation_dir, f"prompt_{prompt_id:03d}.json")
        low_perp_file = os.path.join(perp_regions_dir, f"prompt_{prompt_id:03d}_low_perp.json")
        
        # Initialize generation content
        generations_content = {
            'prompt_metadata': {
                'prompt_id': prompt_id_str,
                'creation_date': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                'prompt_text': prompt,
            },
            'model_info': {
                'model_name': args.model_name,
                'max_length': args.max_length,
                'temperature': args.temp,
                'top_k': args.top_k,
                'top_p': args.top_p,
                'repetition_penalty': args.penalty,
            },
            'tokenizer_info': {
                'tokenizer_name': tokenizer.name_or_path,
                'vocab_size': tokenizer.vocab_size,
                'special_tokens': {
                    'bos_token': tokenizer.bos_token,
                    'eos_token': tokenizer.eos_token,
                    'pad_token': tokenizer.pad_token,
                    'unk_token': tokenizer.unk_token,
                }
            },
            'generations': []
        }
        
        # Initialize low perplexity analysis
        low_perp_analysis = {
            "source_prompt_id": prompt_id_str,
            "analysis_metadata": {
                "analysis_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_file": os.path.relpath(generation_file, experiment_dir),
            },
            "analysis_parameters": {
                "window_size": args.window_size,
                "perplexity_threshold": args.perplexity_threshold,
                "stride": args.stride
            },
            'per_prompt_regions': []
        }
        
        # Generate multiple completions
        for i in range(args.n_gen):
            logger.debug(f"Generation {i+1}/{args.n_gen} for prompt {prompt_id_str}")
            
            try:
                # Generate text and compute perplexity
                token_ids, token_texts, token_perplexities = generate_and_compute_perplexity(
                    model, tokenizer, prompt, args.max_length, args.temp,
                    args.penalty, args.top_k, args.top_p, device, logger
                )
                
                generation_id = f"{prompt_id_str}_G{i:0{GEN_ID_DIGITS}d}"
                
                generation_content = {
                    'generation_id': generation_id,
                    'generated_text': "".join(token_texts),
                    'token_ids': token_ids,
                    'token_texts': token_texts,
                    'token_perplexities': token_perplexities,
                    'generation_params': {
                        'seed': args.seed,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                }
                
                generations_content['generations'].append(generation_content)
                
                # Extract low perplexity regions
                idx, contiguous, is_in_prompt, stats, tokens, token_ids_region = get_fixed_low_perplexity_windows(
                    prompt, token_perplexities, args.perplexity_threshold, 
                    token_texts, token_ids, args.window_size, args.stride, logger
                )
                
                regions = []
                for region_idx, (idx, contiguous, is_in_prompt, 
                                stats, tokens, token_ids_region) in enumerate(zip(idx, contiguous, is_in_prompt, stats, tokens, token_ids_region)):
                    start_idx, end_idx = idx
                    avg_perp, min_perp, max_perp = stats 
                    regions.append({
                        'region_id': f"{generation_id}_R{region_idx+1:0{REGION_ID_DIGITS}d}",
                        'start_index': start_idx,
                        'end_index': end_idx,
                        'tokens': tokens,
                        'token_ids': token_ids_region,
                        'is_contiguous': contiguous,
                        'is_in_prompt': is_in_prompt,
                        'avg_perplexity': avg_perp,
                        'min_perplexity': min_perp,
                        'max_perplexity': max_perp,
                    })
                
                low_perp_per_gen = {
                    "generation_id": generation_id,
                    "per_gen_regions": regions,
                }
                
                low_perp_analysis['per_prompt_regions'].append(low_perp_per_gen)
                logger.info(f"Completed generation {generation_id} with {len(regions)} low-perp regions")
                
            except Exception as e:
                logger.error(f"Failed generation {i+1} for prompt {prompt_id_str}: {e}")
                continue
        
        for generation_data in low_perp_analysis['per_prompt_regions']:
            regions = generation_data['per_gen_regions']
            enhanced_regions = add_standalone_perplexity_to_regions(
                regions, model, tokenizer, device, logger
            )
            generation_data['per_gen_regions'] = enhanced_regions
        
        # Write results to files
        try:
            safe_write_json(generations_content, generation_file, logger)
            safe_write_json(low_perp_analysis, low_perp_file, logger)
            logger.info(f"Successfully saved results for prompt {prompt_id_str}")
        except Exception as e:
            logger.error(f"Failed to save results for prompt {prompt_id_str}: {e}")
            continue
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
