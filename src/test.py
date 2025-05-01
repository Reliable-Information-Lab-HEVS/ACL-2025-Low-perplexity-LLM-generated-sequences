import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load model and tokenizer
model_name = "EleutherAI/pythia-1b"  # A small Pythia model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Input prompt
prompt = "should have received a copy of the GNU General "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate response with output_scores=True
with torch.no_grad():
    # Generate text - this part is correct
    outputs = model.generate(
        input_ids, 
        max_new_tokens=20,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Get generated tokens
    generated_tokens = outputs.sequences[0, input_ids.shape[1]:]
    
    # The issue is here: scores need to be processed differently
    # scores from generate() are logits, not log probabilities
    token_scores = outputs.scores
    
    # Decode generated response
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated text: {generated_text}\n")
    
    # Calculate and print perplexity for each token - corrected version
    print("Token-by-token perplexity:")
    for i, (token, score_tensor) in enumerate(zip(generated_tokens, token_scores)):
        token_str = tokenizer.decode(token)
        
        # Convert logits to probabilities with softmax
        logits = score_tensor[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get probability of the chosen token
        token_prob = probs[token].item()
        
        # Calculate perplexity from probability
        token_perplexity = 1 / token_prob if token_prob > 0 else float('inf')
        
        print(f"Token {i+1}: '{token_str}' - Probability: {token_prob:.6f}, Perplexity: {token_perplexity:.4f}")
    
    # Calculate sequence perplexity
    all_probs = []
    for i, (token, score_tensor) in enumerate(zip(generated_tokens, token_scores)):
        logits = score_tensor[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        token_prob = probs[token].item()
        all_probs.append(token_prob)
    
    # Perplexity = exp(-average_log_likelihood)
    avg_log_likelihood = sum(np.log(p) for p in all_probs) / len(all_probs)
    overall_perplexity = np.exp(-avg_log_likelihood)
    print(f"\nOverall perplexity: {overall_perplexity:.4f}")
