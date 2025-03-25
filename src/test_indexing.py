import numpy as np
from sentence_transformers import SentenceTransformer
import heapq

def get_top_embedding_indices(text, model, top_k=10):
    """
    Get the indices of the top_k largest absolute values in the embedding.
    
    Args:
        text (str): Input text to embed
        model: SentenceTransformer model
        top_k (int): Number of top indices to return
    
    Returns:
        list: List of tuples (index, value) of top_k indices sorted by absolute value
    """
    # Get embedding for the text
    embedding = model.encode(text)
    
    # Get absolute values
    abs_embedding = np.abs(embedding)
    
    # Find indices of top_k largest absolute values
    top_indices = heapq.nlargest(top_k, enumerate(abs_embedding), key=lambda x: x[1])
    
    return top_indices

def main():
    # Load the model once (this can take some time)
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Using a smaller model for faster loading
    print("Model loaded!")
    
    print("\nEnter text to get the top embedding indices (or 'quit' to exit):")
    
    while True:
        user_input = input("\nText: ")
        
        if user_input.lower() == 'quit':
            break
            
        top_indices = get_top_embedding_indices(user_input, model)
        
        print(f"\nTop 10 indices with highest absolute values for: '{user_input}'")
        print("-" * 50)
        print("Rank | Index | Value")
        print("-" * 50)
        
        for rank, (index, value) in enumerate(top_indices, 1):
            print(f"{rank:4d} | {index:5d} | {value:.6f}")

if __name__ == "__main__":
    main()