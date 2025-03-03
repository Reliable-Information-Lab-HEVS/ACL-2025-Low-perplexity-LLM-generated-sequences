import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def construct_few_shot_prompt(task_description, examples, query):
    """
    Construct a few-shot prompt with examples and a query.

    Args:
        task_description (str): Description of the task
        examples (list): List of dictionaries containing input/output pairs
        query (str): The query to be answered

    Returns:
        str: The constructed few-shot prompt
    """
    prompt = f"{task_description}\n\n"

    for i, example in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"

    prompt += f"Now, please answer the following:\n"
    prompt += f"Input: {query}\n"
    prompt += f"Output:"

    return prompt


def generate_response(
    prompt, model, tokenizer, max_new_tokens=100, temperature=0.7, top_p=0.9
):
    """
    Generate a response for the given prompt using the model.

    Args:
        prompt (str): The input prompt
        model: The transformer model
        tokenizer: The tokenizer
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness in generation
        top_p (float): Controls diversity via nucleus sampling

    Returns:
        str: The generated response
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode the output, skip the input tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )

    return response


def main():
    parser = argparse.ArgumentParser(description="Few-shot prompting with Transformers")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model identifier from Hugging Face",
    )
    parser.add_argument(
        "--task_description", type=str, required=True, help="Description of the task"
    )
    parser.add_argument(
        "--examples_file",
        type=str,
        required=True,
        help="JSON file containing few-shot examples",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Query to generate a response for"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Load examples from file
    import json

    with open(args.examples_file, "r") as f:
        examples = json.load(f)

    # Load model and tokenizer
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto",
    )

    # Construct the few-shot prompt
    prompt = construct_few_shot_prompt(args.task_description, examples, args.query)

    print("\nPrompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    # Generate the response
    response = generate_response(
        prompt,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\nGenerated Response:")
    print("-" * 50)
    print(response)
    print("-" * 50)


if __name__ == "__main__":
    main()
