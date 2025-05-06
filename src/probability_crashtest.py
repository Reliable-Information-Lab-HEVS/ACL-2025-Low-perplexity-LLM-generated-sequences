import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json
from utils import get_fixed_low_perplexity_windows

def calculate_expected_token_probs(
    model,
    tokenizer,
    expected_completion,
    device,
):
    """
    Calculate the probability of expected tokens given the prompt.
    Returns token-by-token probabilities of the expected completion.
    """
    if not expected_completion:
        return []
    
    # Tokenize the expected completion
    completion_ids = tokenizer.encode(
        expected_completion, add_special_tokens=False, return_tensors="pt"
    ).to(device)[0]
    
    # Store probabilities and tokens
    token_probs = []
    
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    prompt_ids = torch.tensor([[bos_token_id]], device=device) if bos_token_id is not None else torch.tensor([[]], device=device)

    # Get the decoded tokens for better output
    completion_tokens = []
    for token_id in completion_ids:
        token_str = tokenizer.decode(token_id)
        completion_tokens.append(token_str)
    
    current_context = prompt_ids
    
    # For each token in the expected completion
    for i, token_id in enumerate(completion_ids):
        # Forward pass with the current context
        with torch.no_grad():
            outputs = model(input_ids=current_context)
            logits = outputs.logits
            
            # Get probabilities for the next token
            next_token_logits = logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            
            # Get probability of the expected token
            token_prob = probs[token_id].item()
            token_perplexity = 1 / token_prob if token_prob > 0 else float("inf")
            
            # Save the token and its probability
            token_probs.append({
                "token": completion_tokens[i],
                "token_id": token_id.item(),
                "probability": token_prob,
                "perplexity": token_perplexity
            })
            
            # Add this token to the context for the next iteration
            current_context = torch.cat([current_context, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return token_probs


def generate_and_compute_perplexity(
    model,
    tokenizer,
    prompt,
    max_length,
    temperature,
    top_k,
    top_p,
    device,
):
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
            output_scores=True,
            top_k=top_k,
            top_p=top_p,
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

            # Calculate perplexity from probability
            token_prob = probs[token].item()
            token_perplexity = 1 / token_prob if token_prob > 0 else float("inf")
            token_perplexities.append((token_str, token_perplexity))

    return generated_text, token_perplexities, generated_tokens.tolist()


def generate_and_compute_probability(model, tokenizer, prompt, max_length, temperature, device):
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

    # Calculate probability for each token
    token_probabilities = []
    for i, (token, score_tensor) in enumerate(zip(generated_tokens, token_scores)):
        token_str = tokenizer.decode(token)
        
        # Convert logits to probabilities with softmax
        logits = score_tensor[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get probability of the chosen token
        token_prob = probs[token].item()
        
        # Save the token and its probability
        token_probabilities.append((token_str, token_prob))
        
    return generated_text, token_probabilities, generated_tokens.tolist()
# This function is no longer needed as we're using get_longest_low_perplexity from utils

def save_to_json(json_file, prompt, generated_text, token_perplexities, windows_texts, windows_probs, filename):
    
    dic = []
    for i, (window_text, window_probs) in enumerate(zip(windows_texts, windows_probs)):
        dic.append({
            "window_text": window_text,
            "window_avg_perplexity": window_probs
        })
        
    result = {
        # "prompt": prompt,
        # "generated_text": generated_text,
        "file": filename,
        "low_perplexity_windows": dic,
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
        "--temp", type=float, default=0.4, help="Temperature for sampling"
    )
    parser.add_argument("--top_k", type=int, default=20, help="top_k for sampling")
    parser.add_argument("--top_p", type=float, default=0.8, help="top_p for sampling")
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
        "--window-size",
        type=int,
        default=6,
        help="Size of the window that contains the low perplexity tokens",
    )
    parser.add_argument(
        "--perplexity_threshold",
        type=float,
        default=1.0,
        help="Threshold for low perplexity in finding longest sequence",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default="perplexity_results.json",
        help="JSON file to store perplexity results",
    )
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    #Llama2 tokenizer
    llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Check if the prompt argument is a .json file
    if args.prompt.endswith('.json'):
        with open(args.prompt, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        prompts = prompt_data.get('prompts', [])
        if not prompts:
            prompts = prompt_data.get('variants', [])
        if not prompts:
            raise ValueError("No prompts found in the JSON file.")
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
    
    # Perplexities version
    for i_prompt, prompt in enumerate(prompts):
        # Process multiple generations
        for i in range(args.n_gen):
            # Get output file name with index
            base_name, ext = os.path.splitext(args.output_file)
            current_output_file = f"{base_name}_P{i_prompt}_{i}{ext}"
            
            # Generate text and compute perplexity
            generated_text, token_perplexities, raw_tokens = generate_and_compute_perplexity(
                model, 
                tokenizer, 
                prompt, 
                args.max_length, 
                args.temp,
                args.top_k,
                args.top_p,
                device
            )
            
            # Extract just the perplexity values for the get_longest_low_perplexity function
            perplexity_values = [perp for _, perp in token_perplexities]
            perplexity_values = list(np.log(perplexity_values))
            
            # Find longest sequence of low perplexity tokens
            low_perp_windows = get_fixed_low_perplexity_windows(perplexity_values, args.perplexity_threshold, 6)
            
            perplexity_values = [(token, perp) for perp, (token, _) in zip(perplexity_values, token_perplexities)]
            
            # Write results to file
            with open(current_output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write("Token perplexities:\n")
                for token, perplexity in perplexity_values:
                    f.write(f"{token:<12}: {perplexity:<6.4f} (Probability: {np.exp(-perplexity):.4f})\n")


                windows_texts = []
                windows_perps = []

                for window in low_perp_windows:
                    start_idx, end_idx = window
                    longest_sequence = []
                    if end_idx >= start_idx:  # Make sure the indices are valid
                        longest_sequence = perplexity_values[start_idx:end_idx+1]
                
                    f.write(f"\nLongest sequence of low perplexity tokens (threshold: {args.perplexity_threshold}):\n")
                    f.write(f"Indices: [{start_idx}, {end_idx}]\n")
                    for token, perplexity in longest_sequence:
                        f.write(f"{token}: {perplexity:.4f}\n")
                    
                    # Also print the combined text of the sequence
                    longest_sequence_text = "".join([token for token, _ in longest_sequence])
                    f.write(f"\nLongest low perplexity text: {longest_sequence_text}\n")
                    
                    no_context_token_probs = calculate_expected_token_probs(
                        model,
                        tokenizer,
                        longest_sequence_text,
                        device,
                    )
                    
                    windows_texts.append(longest_sequence_text)
                    windows_perps.append(np.mean([np.log2(prob['perplexity']) for prob in no_context_token_probs]))
                                        
                    f.write(f"Token perplexities (no context): {[prob['perplexity'] for prob in no_context_token_probs]}\n")
                    f.write(f"Average perplexity (no context): {np.mean([prob['perplexity'] for prob in no_context_token_probs]):.4f}\n")
                    
                
                f.write(f"Generated text:\n{generated_text}\n\n")
            
            # Save results to JSON
            save_to_json(
                args.json_output,
                prompt,
                generated_text,
                perplexity_values,
                windows_texts,
                windows_perps,
                current_output_file
            )
            
            print(f"Results written to {current_output_file}")
            print(f"JSON results appended to {args.json_output}")

    
    #Probabilities version
    ''' for i_prompt, prompt in enumerate(prompts):
        # Process multiple generations
        for i in range(args.n_gen):
            # Get output file name with index
            base_name, ext = os.path.splitext(args.output_file)
            current_output_file = f"{base_name}_P{i_prompt}_{i}{ext}"
            
            # Generate text and compute perplexity
            generated_text, token_probabilities, raw_tokens = generate_and_compute_probability(
                model, tokenizer, prompt, args.max_length, args.temp, device
            )
            
            # Extract just the perplexity values for the get_longest_low_perplexity function
            probability_values = [prob for _, prob in token_probabilities]
            
            # Find longest sequence of high probability tokens
            longest_high_prob_indices = get_longest_high_probability(probability_values, args.probability_threshold)
            
            # Write results to file
            with open(current_output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write("Token probabilities:\n")
                for token, probability in token_probabilities:
                    f.write(f"{token}: {probability:.4f}\n")
                
                # Extract the longest sequence based on indices
                start_idx, end_idx = longest_high_prob_indices
                longest_sequence = []
                if end_idx >= start_idx:  # Make sure the indices are valid
                    longest_sequence = token_probabilities[start_idx:end_idx+1]
                
                f.write(f"\nLongest sequence of high probability tokens (threshold: {args.probability_threshold}):\n")
                f.write(f"Indices: [{start_idx}, {end_idx}]\n")
                for token, probability in longest_sequence:
                    f.write(f"{token}: {probability:.4f}\n")
                
                # Also print the combined text of the sequence
                longest_sequence_text = "".join([token for token, _ in longest_sequence])
                f.write(f"\nLongest high probability text: {longest_sequence_text}\n")
                
                f.write(f"Generated text:\n{generated_text}\n\n")
            
            # Save results to JSON
            save_to_json(
                args.json_output,
                prompt,
                generated_text,
                token_probabilities,
                longest_high_prob_indices,
                raw_tokens,
                current_output_file
            )
            
            print(f"Results written to {current_output_file}")
            print(f"JSON results appended to {args.json_output}")'''



if __name__ == "__main__":
    main()