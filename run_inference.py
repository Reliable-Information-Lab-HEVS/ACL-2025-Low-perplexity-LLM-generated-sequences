#!/usr/bin/env python3

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompt_file", type=str, required=True, help="File containing prompts")
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="output.txt")
    return parser.parse_args()

def read_prompts(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts

def generate_text(model, tokenizer, prompt, max_length):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    
    # Read prompts
    prompts = read_prompts(args.prompt_file)
    
    # Generate and save outputs
    with open(args.output_file, "w") as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, prompt in enumerate(prompts, 1):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Prompt {i} at {timestamp}:\n")
            f.write(f"{'='*50}\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            
            output = generate_text(model, tokenizer, prompt, args.max_length)
            
            f.write(f"RESPONSE:\n{output}\n")
            f.write(f"{'='*50}\n\n")

if __name__ == "__main__":
    main()
