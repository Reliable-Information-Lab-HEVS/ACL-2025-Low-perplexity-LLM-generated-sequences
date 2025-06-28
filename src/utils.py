import logging
import torch
import os
import json
import random
import numpy as np
from typing import List, Union, Tuple, Optional


def compute_standalone_perplexity(
    tokens: Union[List[str], str], 
    model, 
    tokenizer, 
    device: str = "cuda",
    logger: Optional[logging.Logger] = None
) -> Tuple[float, List[float], List[str], List[int]]:
    """
    Compute standalone perplexity of a text sequence without any context.
    
    This function tokenizes the input text and computes the perplexity of each token
    given only the preceding tokens in the sequence (no external context).
    
    Args:
        tokens: Either a string of text or a list of token strings
        model: HuggingFace language model (AutoModelForCausalLM)
        tokenizer: HuggingFace tokenizer (AutoTokenizer)
        device: Device to run computation on ("cuda" or "cpu")
        logger: Optional logger for debug information
        
    Returns:
        tuple: (
            average_perplexity: float,
            per_token_perplexities: List[float],
            token_strings: List[str],
            token_ids: List[int]
        )
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Handle input - convert tokens list to string if needed
    if isinstance(tokens, list):
        text = "".join(tokens)
    else:
        text = tokens
    
    if not text.strip():
        logger.warning("Empty text provided for standalone perplexity calculation")
        return float('inf'), [], [], []
    
    logger.debug(f"Computing standalone perplexity for text: {text[:50]}...")
    
    try:
        # Tokenize the text
        encoding = tokenizer.encode(text, return_tensors="pt")
        token_ids = encoding[0].tolist()
        
        if len(token_ids) == 0:
            logger.warning("Text tokenized to empty sequence")
            return float('inf'), [], [], []
        
        if len(token_ids) == 1:
            logger.warning("Text tokenized to single token - cannot compute perplexity")
            # For single token, we can't compute perplexity in the traditional sense
            # Return a high perplexity value
            token_text = tokenizer.decode([token_ids[0]])
            return float('inf'), [float('inf')], [token_text], token_ids
        
        # Convert token IDs back to strings
        token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]
        
        # Move to device
        input_ids = encoding.to(device)
        
        # Compute standalone perplexity
        model.eval()
        per_token_perplexities = []
        
        with torch.no_grad():
            # For each position, compute the probability of the token given the prefix
            for i in range(1, len(token_ids)):  # Start from 1 since we need at least one token of context
                # Create input with tokens up to position i-1
                context_ids = input_ids[:, :i]
                target_token_id = token_ids[i]
                
                # Get model outputs
                outputs = model(context_ids)
                logits = outputs.logits
                
                # Get logits for the last position (where we predict the next token)
                last_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Convert to probabilities
                probs = torch.softmax(last_token_logits, dim=-1)
                
                # Get probability of the actual target token
                target_prob = probs[target_token_id].item()
                
                # Compute perplexity (inverse of probability)
                if target_prob > 0:
                    token_perplexity = 1.0 / target_prob
                else:
                    token_perplexity = float('inf')
                
                per_token_perplexities.append(token_perplexity)
                
                logger.debug(f"Token {i}: '{token_strings[i]}' -> prob: {target_prob:.6f}, perplexity: {token_perplexity:.3f}")
        
        # For the first token, we can't compute perplexity in the same way
        # We'll use the unconditional probability or set it to a default value
        if len(per_token_perplexities) > 0:
            # Option 1: Use the average of other tokens
            first_token_perplexity = np.mean(per_token_perplexities)
            # Option 2: Compute using just the first token as context (uncomment if preferred)
            # first_token_perplexity = compute_first_token_perplexity(token_ids[0], model, tokenizer, device)
        else:
            first_token_perplexity = float('inf')
        
        # Prepend first token perplexity
        per_token_perplexities = [first_token_perplexity] + per_token_perplexities
        
        # Compute average perplexity (excluding infinite values)
        finite_perplexities = [p for p in per_token_perplexities if np.isfinite(p)]
        if finite_perplexities:
            avg_perplexity = np.mean(finite_perplexities)
        else:
            avg_perplexity = float('inf')
        
        logger.debug(f"Standalone perplexity computed: avg={avg_perplexity:.3f}, tokens={len(token_strings)}")
        
        return avg_perplexity, per_token_perplexities, token_strings, token_ids
        
    except Exception as e:
        logger.error(f"Error computing standalone perplexity: {e}")
        return float('inf'), [], [], []

def compute_first_token_perplexity(
    token_id: int, 
    model, 
    tokenizer, 
    device: str = "cuda"
) -> float:
    """
    Compute perplexity of the first token using the model's unconditional distribution.
    
    Args:
        token_id: ID of the first token
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        device: Device to run computation on
        
    Returns:
        float: Perplexity of the first token
    """
    try:
        model.eval()
        with torch.no_grad():
            # Use BOS token or empty context
            if tokenizer.bos_token_id is not None:
                context = torch.tensor([[tokenizer.bos_token_id]], device=device)
            else:
                # For models without BOS token, use a minimal context
                # This is a fallback - results may not be as meaningful
                context = torch.tensor([[token_id]], device=device)
                return float('inf')  # Cannot compute meaningful first token perplexity
            
            outputs = model(context)
            logits = outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)
            
            target_prob = probs[token_id].item()
            
            if target_prob > 0:
                return 1.0 / target_prob
            else:
                return float('inf')
                
    except Exception as e:
        return float('inf')

def batch_compute_standalone_perplexity(
    text_list: List[Union[str, List[str]]], 
    model, 
    tokenizer, 
    device: str = "cuda",
    logger: Optional[logging.Logger] = None
) -> List[Tuple[float, List[float], List[str], List[int]]]:
    """
    Compute standalone perplexity for multiple text sequences.
    
    Args:
        text_list: List of texts (strings or token lists)
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        device: Device to run computation on
        logger: Optional logger
        
    Returns:
        List of tuples, each containing (avg_perplexity, per_token_perplexities, token_strings, token_ids)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = []
    
    for i, text in enumerate(text_list):
        logger.debug(f"Processing text {i+1}/{len(text_list)}")
        result = compute_standalone_perplexity(text, model, tokenizer, device, logger)
        results.append(result)
    
    return results

# Integration function for use in your main analysis script
def add_standalone_perplexity_to_regions(
    regions_data: List[dict], 
    model, 
    tokenizer, 
    device: str = "cuda",
    logger: Optional[logging.Logger] = None
) -> List[dict]:
    """
    Add standalone perplexity calculations to existing region data.
    
    Args:
        regions_data: List of region dictionaries with 'tokens' field
        model: HuggingFace language model
        tokenizer: HuggingFace tokenizer
        device: Device to run computation on
        logger: Optional logger
        
    Returns:
        List of region dictionaries with added standalone perplexity fields
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Computing standalone perplexity for {len(regions_data)} regions")
    
    for i, region in enumerate(regions_data):
        if 'tokens' not in region:
            logger.warning(f"Region {i} missing 'tokens' field, skipping")
            region['standalone_avg_perplexity'] = float('inf')
            region['standalone_per_token_perplexities'] = []
            continue
        
        tokens = region['tokens']
        
        # Compute standalone perplexity
        avg_perp, per_token_perps, token_strings, token_ids = compute_standalone_perplexity(
            tokens, model, tokenizer, device, logger
        )
        
        # Add to region data
        region['standalone_avg_perplexity'] = avg_perp
        region['standalone_per_token_perplexities'] = per_token_perps
        region['standalone_token_strings'] = token_strings
        region['standalone_token_ids'] = token_ids
        
        # Compute ratio of contextual to standalone perplexity
        contextual_perp = region.get('avg_perplexity', float('inf'))
        if np.isfinite(avg_perp) and avg_perp > 0:
            region['perplexity_ratio'] = contextual_perp / avg_perp
        else:
            region['perplexity_ratio'] = float('inf')
        
        logger.debug(f"Region {i}: contextual={contextual_perp:.3f}, standalone={avg_perp:.3f}, ratio={region['perplexity_ratio']:.3f}")
    
    logger.info("Standalone perplexity computation completed")
    return regions_data

def calculate_expected_token_probs(
    model,
    tokenizer,
    expected_completion,
    device,
):
    """
    Calculate the probability of expected tokens given the prompt.
    Returns token-by-token probabilities of the expected completion.
    """
    if not expected_completion:
        return []
    
    # Tokenize the expected completion
    completion_ids = tokenizer.encode(
        expected_completion, add_special_tokens=False, return_tensors="pt"
    ).to(device)[0]
    
    # Store probabilities and tokens
    token_probs = []
    
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
    prompt_ids = torch.tensor([[bos_token_id]], device=device) if bos_token_id is not None else torch.tensor([[]], device=device)

    # Get the decoded tokens for better output
    completion_tokens = []
    for token_id in completion_ids:
        token_str = tokenizer.decode(token_id)
        completion_tokens.append(token_str)
    
    current_context = prompt_ids
    
    # For each token in the expected completion
    for i, token_id in enumerate(completion_ids):
        # Forward pass with the current context
        with torch.no_grad():
            outputs = model(input_ids=current_context)
            logits = outputs.logits
            
            # Get probabilities for the next token
            next_token_logits = logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=0)
            
            # Get probability of the expected token
            token_prob = probs[token_id].item()
            token_perplexity = 1 / token_prob if token_prob > 0 else float("inf")
            
            # Save the token and its probability
            token_probs.append({
                "token": completion_tokens[i],
                "token_id": token_id.item(),
                "probability": token_prob,
                "perplexity": token_perplexity
            })
            
            # Add this token to the context for the next iteration
            current_context = torch.cat([current_context, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return token_probs


def generate_and_compute_perplexity(
    model,
    tokenizer,
    prompt,
    max_length,
    temperature,
    penalty,
    top_k,
    top_p,
    device,
    logger
):
    """Generate text completion and compute perplexity per token.

    Args:
        model (_type_): model
        tokenizer (_type_): tokenizer
        prompt (_type_): prompt
        max_length (_type_): max_length of the generation
        temperature (_type_): temperature used for completion
        penalty (_type_): repetition penalty
        top_k (_type_): top_k for sampling
        top_p (_type_): top_p for sampling
        device (_type_): device to run the model on

    Returns:
        generated_text, perplexities, tokens: Generation text, token perplexities, and raw tokens.
    """
    
    logger.debug(f"Starting generation for prompt: {prompt[:50]}...")
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    logger.debug(f"Prompt tokenized to {input_ids.shape[1]} tokens")
    
    # Generate response with output_scores=True
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=penalty,
        )

        # Get generated tokens and scores
        generated_tokens = outputs.sequences[0, input_ids.shape[1]:]
        token_scores = outputs.scores

        # Initialize lists for token ids, texts, and perplexities
        token_ids = []
        token_texts = []
        token_perplexities = []

        # Calculate perplexity for each token
        for token, score_tensor in zip(generated_tokens, token_scores):
            token_ids.append(token.item())
            token_texts.append(tokenizer.decode(token))

            # Convert logits to probabilities with softmax
            logits = score_tensor[0]
            probs = torch.nn.functional.softmax(logits, dim=0)

            # Calculate perplexity from probability
            token_prob = probs[token].item()
            token_perplexity = 1 / token_prob if token_prob > 0 else float("inf")
            token_perplexities.append(token_perplexity)

        logger.debug(f"Computed perplexities for {len(token_perplexities)} tokens")
        
        return token_ids, token_texts, token_perplexities



def setup_logging(verbose: bool, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('perplexity_analyzer')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (in experiment directory)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name, exist_ok=True)
    
    file_handler = logging.FileHandler(
        os.path.join(experiment_name, 'experiment.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def get_fixed_low_perplexity_windows(prompt, perplexities, threshold, tokens, token_ids, window_size, stride, logger):
    """
    Find all fixed-length windows where every token has perplexity below the threshold.
    Additionally, return information about whether each window is contiguous with the previous one,
    the average, minimum, and maximum perplexities for each window, and the corresponding tokens/words.

    Parameters:
    -----------
    prompt: str
        The original prompt text. Used to check if the tokens in a window are part of the prompt.
    perplexities : list
        List of perplexity values. Each value corresponds to the perplexity of a token.
    threshold : float
        Threshold value for perplexity. Only windows where all perplexities are below this value are considered.
    tokens : list
        List of token words corresponding to the perplexities. Used to extract the tokens in each window.
    token_ids : list
        List of token IDs corresponding to the perplexities. Used to extract the token IDs in each window.
    window_size : int
        Fixed size of the sliding window. Determines the number of tokens in each window.
    stride : int, optional
        Step size for sliding the window (default is 1). Determines how far the window moves after each step.

    Returns:
    --------
    low_perp_windows : list of tuples
        Each tuple contains the start and end indices of a window where all perplexities are below the threshold.
    is_contiguous : list of bool
        Indicates whether each window is contiguous with the previous one, based on the stride.
    is_in_prompt : list of bool
        Indicates whether the tokens in each window are part of the original prompt.
    perplexity_stats : list of tuples
        Each tuple contains the average, minimum, and maximum perplexities for a window.
    window_tokens : list of lists
        Each inner list contains the tokens in a window.
    window_token_ids : list of lists
        Each inner list contains the token IDs in a window.
    """
    if len(perplexities) < window_size:
        return [], [], [], []
    
    logger.debug(f"Extracting low perplexity windows with threshold {threshold}")
    
    assert stride <= window_size, "Stride must be less than or equal to window size."

    low_perp_windows = []
    is_contiguous = []
    is_in_prompt = []
    perplexity_stats = []
    window_tokens = []
    window_token_ids = []
    

    i = 0
    while i <= len(perplexities) - window_size:
        # Check if all values in current window are below threshold
        window = perplexities[i:i + window_size]
        
        if all(perp <= threshold for perp in window):
            # Use standard indexing: start is inclusive, end is exclusive
            # So window containing indices i to i+window_size-1 is (i, i+window_size)
            low_perp_windows.append((i, i + window_size))
            
            contig = False
            # Check if this window is contiguous with the previous one
            if low_perp_windows and len(low_perp_windows) > 1 and stride < window_size:
                if stride < window_size:
                    contig = low_perp_windows[-1][0] < low_perp_windows[-2][1]
                else:
                    contig = low_perp_windows[-1][0] == low_perp_windows[-2][1]

            is_contiguous.append(contig)
            
            # Calculate perplexity statistics
            avg_perplexity = sum(window) / window_size
            min_perplexity = min(window)
            max_perplexity = max(window)
            perplexity_stats.append((avg_perplexity, min_perplexity, max_perplexity))
            in_prompt = False
            # Extract corresponding tokens/words if provided
            if tokens:
                window_tokens.append(tokens[i:i + window_size])
                in_prompt = ''.join(tokens[i:i + window_size]) in prompt
                window_token_ids.append(token_ids[i:i + window_size] if token_ids else [])
            else:
                window_tokens.append([])
                window_token_ids.append([])
            is_in_prompt.append(in_prompt)
            
            # Move the index by stride
            i += stride
        else:
            # Move the index by 1 until a low-perplexity region is found
            i += 1

    logger.debug(f"Found {len(low_perp_windows)} low perplexity regions")
    return low_perp_windows, is_contiguous, is_in_prompt, perplexity_stats, window_tokens, window_token_ids

def get_longest_high_probability(probabilities, threshold):
    """
    Find the longest subarray of perplexities where values are below the threshold,
    allowing up to a specified number of tolerance values above the threshold.

    Parameters:
    -----------
    perplexities : list
        List of perplexity values
    threshold : float
        Threshold value for perplexity

    Returns:
    --------
    start_index : int
        Starting index of the longest subarray
    end_index : int
        Ending index of the longest subarray
    """
    max_length = 0
    max_start = 0
    max_end = 0
    current_start = 0

    for i, prob in enumerate(probabilities):
        
        if prob < threshold:
            current_start = i
        else:
            if i - current_start > max_length:
                max_length = i - current_start
                max_start = current_start
                max_end = i

    return max_start, max_end

def safe_write_json(data, filepath, logger):
    """Safely write JSON data to file"""
    temp_filepath = filepath + ".tmp"
    try:
        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic move
        os.rename(temp_filepath, filepath)
        logger.debug(f"Successfully wrote {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to write {filepath}: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise