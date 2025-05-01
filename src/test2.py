import torch
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def generate_text_and_perplexity(prompts, model, tokenizer, device, output_file):

    with open(output_file, 'w') as file:
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                generated_tokens = outputs.sequences[0, input_ids.shape[1]:]
                token_scores = outputs.scores
                
                file.write(f"\nPrompt: {prompt}\n")
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                file.write(f"Generated text: {generated_text}\n")
                
                file.write("Token-by-token perplexity:\n")
                all_probs = []
                
                for i, (token, score_tensor) in enumerate(zip(generated_tokens, token_scores)):
                    token_str = tokenizer.decode(token)
                    logits = score_tensor[0]
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    token_prob = probs[token].item()
                    token_perplexity = 1 / token_prob if token_prob > 0 else float('inf')
                    all_probs.append(token_prob)
                    #print(f"Token {i+1}: '{token_str}' - Probability: {token_prob:.6f}, Perplexity: {token_perplexity:.4f}")
                    file.write(f"Token {i+1}: '{token_str}' - Probability: {token_prob:.6f}\n")
                
                
                avg_log_likelihood = sum(np.log(p) for p in all_probs) / len(all_probs)
                overall_perplexity = np.exp(-avg_log_likelihood)
                file.write(f"\nOverall perplexity: {overall_perplexity:.4f}\n")
                file.write("\n" + "="*50 + "\n")


#model_name = "allenai/paloma-1b-baseline-c4"  # A small Pythia model

model_name = "EleutherAI/pythia-12b"  # A small Pythia model
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# Example usage:
prompts = [
    "Law is a set of rules that are created and are enforceable by social or governmental institutions to regulate ",
    "A capital city, or just capital, is the municipality holding primary status in a country, state, province, department, or other subnational division, usually as its seat of the government. A capital is typically a", 
    "Paris is home to several United Nations organizations including UNESCO, as well as other international organizations such as the OECD, the OECD Development Centre, the International Bureau of Weights and Measures, the International Energy Agency, the International Federation for Human Rights, along with European bodies such as the European Space "]
output_file = "test_results/test3.txt"
generate_text_and_perplexity(prompts, model, tokenizer, device, output_file)