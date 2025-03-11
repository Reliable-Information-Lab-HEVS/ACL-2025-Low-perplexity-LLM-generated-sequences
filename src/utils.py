def get_longest_low_perplexity(perplexities, threshold):
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

    for i, perp in enumerate(perplexities):
        
        if perp > threshold:
            current_start = i
        else:
            if i - current_start > max_length:
                max_length = i - current_start
                max_start = current_start
                max_end = i

    return max_start, max_end