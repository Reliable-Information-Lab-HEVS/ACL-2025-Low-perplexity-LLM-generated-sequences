import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class EmbeddingNoiseGenerator:
    def __init__(self, model_name):
        # Load the model and tokenizer from HuggingFace
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set the pad_token to eos_token if it's not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def get_embedding(self, sentence):
        """ Convert sentence to embedding. """
        # Tokenize and ensure padding is handled correctly with attention mask
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        
        # Manually ensure attention_mask is passed
        attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
        
        with torch.no_grad():
            outputs = self.model.base_model(
                input_ids=inputs['input_ids'],
                attention_mask=attention_mask
            )
        
        # Extracting the embedding from the last hidden state (usually [batch_size, seq_len, hidden_dim])
        embedding = outputs.last_hidden_state.mean(dim=1)  # Mean over token embeddings
        return embedding


    
    def add_gaussian_noise(self, embedding, noise_level):
        """ Add controlled Gaussian noise to the embedding. """
        noise = torch.normal(mean=0.0, std=noise_level, size=embedding.size())
        noisy_embedding = embedding + noise
        # Normalize embedding to maintain proper scaling
        noisy_embedding = torch.nn.functional.normalize(noisy_embedding, p=2, dim=-1)
        return noisy_embedding
    
    def generate_variations(self, sentence, noise_level, num_variations=10):
        """ Generate multiple variations by adding noise to the embedding. """
        original_embedding = self.get_embedding(sentence)
        variations = []
        
        for _ in range(num_variations):
            noisy_embedding = self.add_gaussian_noise(original_embedding, noise_level)
            # Decode from the modified embedding
            variation = self.generate_from_embedding(noisy_embedding)
            variations.append(variation)
        
        return variations
    
    def generate_from_embedding(self, noisy_embedding):
        """ Generate text based on the noisy embedding using the model. """
        # We will encode the noisy embedding as input to generate text (by modifying the last hidden state)
        # Generate input_ids from noisy embedding (simple greedy decoding strategy)
        
        # For simplicity, we generate text from the noisy embedding by performing a prompt-based generation
        input_ids = self.tokenizer.encode("<|endoftext|>", return_tensors="pt")  # Start with a dummy token
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text

# Example usage
if __name__ == "__main__":
    # Initialize generator with the model
    generator = EmbeddingNoiseGenerator("EleutherAI/pythia-70m")

    # Input sentence
    input_sentence = "Artificial intelligence continues to evolve rapidly."

    # Generate 10 variations with noise level 0.05
    variations = generator.generate_variations(
        input_sentence,
        noise_level=0.001,
        num_variations=10
    )

    # Print results
    for i, variation in enumerate(variations, 1):
        print(f"{i}. {variation}")
