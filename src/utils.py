import logging
import torch
import os
import json
import random


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