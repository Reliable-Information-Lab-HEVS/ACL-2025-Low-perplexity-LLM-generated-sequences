import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
from utils import get_longest_low_perplexity

def generate_and_compute_perplexity(model, tokenizer, prompt, max_length, temperature, device):
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate response with output_scores=True
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Get generated tokens and scores
        generated_tokens = outputs.sequences[0, input_ids.shape[1]:]
        token_scores = outputs.scores
        
        # Decode generated response
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate perplexity for each token
        token_perplexities = []
        for i, (token, score_tensor) in enumerate(zip(generated_tokens, token_scores)):
            token_str = tokenizer.decode(token)
            
            # Convert logits to probabilities with softmax
            logits = score_tensor[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
            
            # Get probability of the chosen token
            token_prob = probs[token].item()
            
            # Calculate perplexity from probability
            token_perplexity = 1 / token_prob if token_prob > 0 else float('inf')
            token_perplexities.append((token_str, token_perplexity))
    
    return generated_text, token_perplexities, generated_tokens.tolist()

# This function is no longer needed as we're using get_longest_low_perplexity from utils

def save_to_json(json_file, prompt, generated_text, token_perplexities, longest_low_perp_indices, raw_tokens, filename):
    # Get the start and end indices of the longest low perplexity sequence
    start_idx, end_idx = longest_low_perp_indices
    
    # Extract the longest sequence based on indices
    longest_sequence = []
    if end_idx >= start_idx:  # Make sure the indices are valid
        longest_sequence = token_perplexities[start_idx:end_idx+1]
    
    # Create the longest sequence text as a single string
    longest_sequence_text = "".join([token for token, _ in longest_sequence])
    
    # Count the number of tokens in the longest sequence
    token_count = len(longest_sequence)
    
    # Create simplified dictionary with only the requested fields
    result = {
        # "prompt": prompt,
        # "generated_text": generated_text,
        "file": filename,
        "longest_low_perplexity_text": longest_sequence_text,
        "longest_low_perplexity_length": len(longest_sequence),
        "token_count": token_count,
        "avg_perplexity": np.mean([perp for _, perp in token_perplexities])
    }
    
    # Load existing results if file exists
    existing_results = []
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, start with empty list
            existing_results = []
    
    # Append new result
    if isinstance(existing_results, list):
        existing_results.append(result)
    else:
        existing_results = [result]
    
    # Write updated results to file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Generate text and compute perplexity per token')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--n_gen', type=int, default=1, help='Number of generations to perform')
    parser.add_argument('--output_file', type=str, default='output_perplexity.txt', help='Output file name')
    parser.add_argument('--candidates', type=str, default='candidates.json', help='Candidate json file name')
    parser.add_argument('--perplexity_threshold', type=float, default=1.0, 
                        help='Threshold for low perplexity in finding longest sequence')
    parser.add_argument('--json_output', type=str, default='perplexity_results.json', 
                        help='JSON file to store perplexity results')
    
    args = parser.parse_args()
    
    
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    # Check if the prompt argument is a .json file
    if args.prompt.endswith('.json'):
        with open(args.prompt, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
            prompts = [prompt_data['prompt']] + prompt_data.get('variants', [])
    else:
        prompts = [args.prompt]
        
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure the JSON output directory exists
    json_dir = os.path.dirname(args.json_output)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    for i_prompt, prompt in enumerate(prompts):
        # Process multiple generations
        for i in range(args.n_gen):
            # Get output file name with index
            base_name, ext = os.path.splitext(args.output_file)
            current_output_file = f"{base_name}_P{i_prompt}_{i}{ext}"
            
            # Generate text and compute perplexity
            generated_text, token_perplexities, raw_tokens = generate_and_compute_perplexity(
                model, tokenizer, prompt, args.max_length, args.temp, device
            )
            
            # Extract just the perplexity values for the get_longest_low_perplexity function
            perplexity_values = [perp for _, perp in token_perplexities]
            perplexity_values = np.log(perplexity_values)
            
            # Find longest sequence of low perplexity tokens
            longest_low_perp_indices = get_longest_low_perplexity(perplexity_values, args.perplexity_threshold)
            
            # Write results to file
            with open(current_output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write("Token perplexities:\n")
                for token, perplexity in token_perplexities:
                    f.write(f"{token}: {perplexity:.4f}\n")
                
                # Extract the longest sequence based on indices
                start_idx, end_idx = longest_low_perp_indices
                longest_sequence = []
                if end_idx >= start_idx:  # Make sure the indices are valid
                    longest_sequence = token_perplexities[start_idx:end_idx+1]
                
                f.write(f"\nLongest sequence of low perplexity tokens (threshold: {args.perplexity_threshold}):\n")
                f.write(f"Indices: [{start_idx}, {end_idx}]\n")
                for token, perplexity in longest_sequence:
                    f.write(f"{token}: {perplexity:.4f}\n")
                
                # Also print the combined text of the sequence
                longest_sequence_text = "".join([token for token, _ in longest_sequence])
                f.write(f"\nLongest low perplexity text: {longest_sequence_text}\n")
                
                f.write(f"Generated text:\n{generated_text}\n\n")
            
            # Save results to JSON
            save_to_json(
                args.json_output,
                prompt,
                generated_text,
                token_perplexities,
                longest_low_perp_indices,
                raw_tokens,
                current_output_file
            )
            
            print(f"Results written to {current_output_file}")
            print(f"JSON results appended to {args.json_output}")

if __name__ == "__main__":
    main()