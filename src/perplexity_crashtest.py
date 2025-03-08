import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import json

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
    
    return generated_text, token_perplexities

def main():
    parser = argparse.ArgumentParser(description='Generate text and compute perplexity per token')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--n_gen', type=int, default=1, help='Number of generations to perform')
    parser.add_argument('--output_file', type=str, default='output_perplexity.txt', help='Output file name')
    
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
    
    for prompt in prompts:
    
        # Process multiple generations
        for i in range(args.n_gen):
            # Get output file name with index
            base_name, ext = os.path.splitext(args.output_file)
            current_output_file = f"{base_name}_{i}{ext}"
            
            # Generate text and compute perplexity
            generated_text, token_perplexities = generate_and_compute_perplexity(
                model, tokenizer, prompt, args.max_length, args.temp, device
            )
            
            # Write results to file
            with open(current_output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write(f"Generated text:\n{generated_text}\n\n")
                f.write("Token perplexities:\n")
                for token, perplexity in token_perplexities:
                    f.write(f"{token}: {perplexity:.4f}\n")
            
            print(f"Results written to {current_output_file}")

if __name__ == "__main__":
    main()

