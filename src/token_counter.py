#!/usr/bin/env python3

import json
import sys
import argparse
from transformers import AutoTokenizer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate average token count for prompts using Pythia-6B tokenizer')
    parser.add_argument('json_file', help='Path to JSON file containing prompts')
    args = parser.parse_args()
    
    try:
        # Load JSON file
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if 'prompts' key exists
        if 'prompts' not in data:
            print(f"Error: 'prompts' key not found in {args.json_file}")
            sys.exit(1)
        
        prompts = data['prompts']
        
        # Ensure prompts is a list
        if not isinstance(prompts, list):
            print(f"Error: 'prompts' should be a list of strings")
            sys.exit(1)
        
        # Filter only string prompts
        string_prompts = [p for p in prompts if isinstance(p, str)]
        
        if not string_prompts:
            print("No string prompts found in the JSON file")
            sys.exit(1)
        
        print(f"Found {len(string_prompts)} string prompts")
        
        # Load Pythia-6B tokenizer
        print("Loading Pythia-6B tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            sys.exit(1)
        
        # Tokenize all prompts and count tokens
        token_counts = []
        
        for i, prompt in enumerate(string_prompts):
            try:
                tokens = tokenizer.encode(prompt)
                token_count = len(tokens)
                token_counts.append(token_count)
                print(f"Prompt {i+1}: {token_count} tokens")
            except Exception as e:
                print(f"Error tokenizing prompt {i+1}: {e}")
                continue
        
        # Calculate average
        if token_counts:
            average_tokens = sum(token_counts) / len(token_counts)
            print(f"\nAverage number of tokens: {average_tokens:.2f}")
            print(f"Min tokens: {min(token_counts)}")
            print(f"Max tokens: {max(token_counts)}")
        else:
            print("No prompts could be tokenized")
    
    except FileNotFoundError:
        print(f"Error: File {args.json_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.json_file}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()