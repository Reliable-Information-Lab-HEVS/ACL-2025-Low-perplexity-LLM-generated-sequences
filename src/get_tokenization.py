import argparse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Get the tokenization of a word")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model identifier from Hugging Face",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Query to generate a response for"
    )
    args = parser.parse_args()

    # Load examples from file
    import json

    with open(args.examples_file, "r") as f:
        examples = json.load(f)

    # Load model and tokenizer
    print(f"Loading tokenizer {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    tokens = tokenizer.tokenize(args.query)

    print(tokens)


if __name__ == "__main__":
    main()
