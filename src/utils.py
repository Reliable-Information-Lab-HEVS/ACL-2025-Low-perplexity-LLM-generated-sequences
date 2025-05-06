def get_fixed_low_perplexity_windows(perplexities, threshold, window_size):
    """
    Find all fixed-length windows where every token has perplexity below the threshold.
    
    Parameters:
    -----------
    perplexities : list
        List of perplexity values
    threshold : float
        Threshold value for perplexity
    window_size : int
        Fixed size of the sliding window
    
    Returns:
    --------
    list of tuples
        List of (start_index, end_index) tuples where each window contains only
        tokens with perplexity below the threshold
    """
    if len(perplexities) < window_size:
        return []
    
    low_perp_windows = []
    
    # Slide the window across the perplexities
    for i in range(len(perplexities) - window_size + 1):
        # Check if all values in current window are below threshold
        window = perplexities[i:i + window_size]
        
        if all(perp <= threshold for perp in window):
            # Use standard indexing: start is inclusive, end is exclusive
            # So window containing indices i to i+window_size-1 is (i, i+window_size)
            low_perp_windows.append((i, i + window_size))
    
    return low_perp_windows

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