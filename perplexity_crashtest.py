#!/usr/bin/env python3

import argparse
import os

import torch
from dotenv import load_dotenv
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--output_file", type=str, default="generation_perplexity.txt")
    return parser.parse_args()


def generate_and_analyze(model, tokenizer, prompt, max_length):
    # First generate the response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Get the full sequence including prompt and response
        generated_ids = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response_start = len(prompt)

        # Analyze perplexity on the full sequence
        model_outputs = model(generated_ids.unsqueeze(0))
        logits = model_outputs.logits[0]
        probs = softmax(logits, dim=-1)

        tokens = []
        perplexities = []

        for i in range(len(generated_ids) - 1):
            next_token_id = generated_ids[i + 1].item()
            next_token_prob = probs[i, next_token_id].item()
            perplexity = 1 / next_token_prob if next_token_prob > 0 else float("inf")

            tokens.append(tokenizer.decode(generated_ids[i]))
            perplexities.append(perplexity)

    return generated_text, response_start, tokens, perplexities


def main():
    args = parse_args()

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    model = GPTNeoXForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )

    # Generate and analyze
    full_text, response_start, tokens, perplexities = generate_and_analyze(
        model, tokenizer, args.prompt, args.max_length
    )

    # Write results
    with open(args.output_file, "w") as f:
        # Write complete input and output
        f.write("Input prompt:\n")
        f.write(f"{args.prompt}\n\n")
        f.write("Generated response:\n")
        f.write(f"{full_text[response_start:]}\n\n")

        # Write analysis
        f.write("\nToken-by-token analysis:\n")
        f.write("PROMPT TOKENS:\n")
        for i, (token, perp) in enumerate(zip(tokens, perplexities)):
            visible_token = token.replace("\n", "⏎")
            if i * len(token) < response_start:
                f.write(
                    f"Token {i:3d} | Perplexity: {perp:6.2f} | Token: '{visible_token}'\n"
                )
            elif i * len(token) == response_start:
                f.write("\nRESPONSE TOKENS:\n")
                f.write(
                    f"Token {i:3d} | Perplexity: {perp:6.2f} | Token: '{visible_token}'\n"
                )
            else:
                f.write(
                    f"Token {i:3d} | Perplexity: {perp:6.2f} | Token: '{visible_token}'\n"
                )

        # Add summary statistics
        prompt_perp = (
            sum(perplexities[:response_start]) / response_start
            if response_start > 0
            else 0
        )
        response_perp = sum(perplexities[response_start:]) / (
            len(perplexities) - response_start
        )

        f.write("\nSummary:\n")
        f.write(f"Average prompt perplexity: {prompt_perp:.2f}\n")
        f.write(f"Average response perplexity: {response_perp:.2f}\n")


if __name__ == "__main__":
    main()
