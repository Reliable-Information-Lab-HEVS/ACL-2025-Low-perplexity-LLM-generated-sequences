#!/usr/bin/env python3
import json
import requests
import argparse
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial
import random
import os

def query_infinigram(text, index="v4_piletrain_llama", max_retries=5, initial_backoff=1):
    """
    Query the infini-gram API to get count of a text phrase with retry logic
    
    Args:
        text (str): The text to search for
        index (str): The index to search in
        max_retries (int): Maximum number of retry attempts
        initial_backoff (float): Initial backoff time in seconds
        
    Returns:
        dict: API response with count and other info
    """
    payload = {
        'index': index,
        'query_type': 'count',
        'query': text.strip(),
    }
    
    retry_count = 0
    backoff = initial_backoff
    
    while retry_count <= max_retries:
        try:
            response = requests.post('https://api.infini-gram.io/', json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Check if there's an error in the response
            if 'error' in result:
                raise requests.exceptions.RequestException(f"API Error: {result['error']}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            
            if retry_count > max_retries:
                print(f"Failed after {max_retries} retries for '{text}': {str(e)}")
                return None
            
            # Add jitter to backoff to prevent thundering herd problem
            sleep_time = backoff + random.uniform(0, 0.5)
            print(f"Retry {retry_count}/{max_retries} for '{text[:30]}...' after {sleep_time:.2f}s: {str(e)}")
            time.sleep(sleep_time)
            
            # Exponential backoff with jitter
            backoff = min(backoff * 2, 30)  # Cap at 30 seconds

def process_window(window, index, max_retries=5, initial_backoff=1.0, file=""):
    """Process a single window entry with the API"""
    window_text = window.get('window_text', '')
    
    # Check if file exists
    if not os.path.exists(file):
        print(f"File {file} does not exist. Skipping window.")
        return window
    
    window['in_prompt'] = False

    # Open the file and check that the window text is after "Generated text:" in the file
    if file:
        # reads all content
        with open(file, 'r') as f:
            lines = f.readlines()
            text = "\n".join(lines)
            generation = text.split('Generated text:')
            if len(generation) <= 1 or window_text.replace("\n", "") not in generation[1].replace("\n", ""):
                print(f"Window text {window_text} not found. Skipping window.")
                return window
            generation = text.split('Token perplexities:')
            if len(generation) <= 1 or window_text.replace("\n", "") not in generation[0].replace("\n", ""):
                # print(f"Window text found in prompt..")
                window['in_prompt'] = True
            
            # for i, split in enumerate(text.split('Longest low perplexity text: ' + window_text)):
                
    else:
        print(f"File not found.")
        return window

    # In the file, check if the window 
    
    if not window_text:
        return window

    # Query the API
    result = query_infinigram(window_text, index, max_retries, initial_backoff)
    
    if result and 'error' not in result:
        # Add the count and approx info to the window
        window['infinigram_count'] = result.get('count', 0)
        window['infinigram_approx'] = result.get('approx', False)
        window['infinigram_tokens'] = result.get('tokens', [])
    else:
        # If there was an error, set count to 0
        window['infinigram_count'] = 0
        window['infinigram_approx'] = False
        window['infinigram_error'] = True if result and 'error' in result else "API request failed"
    
    return window

def process_entry(entry, index, max_retries=5, initial_backoff=1.0, remove_first_word=False, max_workers=5):
    """Process an entire entry with the API, handling multiple windows"""
    
    if 'low_perplexity_windows' in entry and entry['low_perplexity_windows']:
        # Get all windows from the entry
        windows = entry['low_perplexity_windows']
        
        # Process each window with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function for processing windows
            process_func = partial(
                process_window, 
                index=index, 
                max_retries=max_retries, 
                initial_backoff=initial_backoff,
                file=entry.get('file', '')  # Pass the file if it exists
            )
            
            # Submit all windows for processing
            futures = {executor.submit(process_func, window): i for i, window in enumerate(windows)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    windows[idx] = future.result()
                except Exception as e:
                    print(f"Error processing window {idx}: {str(e)}")
        
        # Update entry with processed windows
        entry['low_perplexity_windows'] = windows
    
    # Also check for the old style entry format with 'longest_low_perplexity_text'
    elif 'longest_low_perplexity_text' in entry and entry['longest_low_perplexity_text']:
        text = entry['longest_low_perplexity_text']
        
        result = query_infinigram(text, index, max_retries, initial_backoff)
        
        if result and 'error' not in result:
            # Add the count and approx info to the entry
            entry['infinigram_count'] = result.get('count', 0)
            entry['infinigram_approx'] = result.get('approx', False)
            entry['infinigram_tokens'] = result.get('tokens', [])
        else:
            # If there was an error, set count to 0
            entry['infinigram_count'] = 0
            entry['infinigram_approx'] = False
            entry['infinigram_error'] = True if result and 'error' in result else "API request failed"
            
    
    
    # Also check for the old style with window_text directly in the entry
    elif 'window_text' in entry and entry['window_text']:
        text = entry['window_text']
        
        if remove_first_word and len(text.split()) > 1:
            text = ' '.join(text.split()[1:])
        
        result = query_infinigram(text, index, max_retries, initial_backoff)
        
        if result and 'error' not in result:
            # Add the count and approx info to the entry
            entry['infinigram_count'] = result.get('count', 0)
            entry['infinigram_approx'] = result.get('approx', False)
            entry['infinigram_tokens'] = result.get('tokens', [])
        else:
            # If there was an error, set count to 0
            entry['infinigram_count'] = 0
            entry['infinigram_approx'] = False
            entry['infinigram_error'] = True if result and 'error' in result else "API request failed"
    
    return entry

def count_total_windows(data):
    """Count the total number of windows/texts that will be processed"""
    total_windows = 0
    
    for entry in data:
        if 'low_perplexity_windows' in entry and entry['low_perplexity_windows']:
            total_windows += len(entry['low_perplexity_windows'])
        elif 'longest_low_perplexity_text' in entry and entry['longest_low_perplexity_text']:
            total_windows += 1
        elif 'window_text' in entry and entry['window_text']:
            total_windows += 1
    
    return total_windows

def process_json_file(input_file, output_file=None, index="v4_piletrain_llama", 
                     max_workers=10, max_retries=5, initial_backoff=1.0, remove_first_word=False,
                     max_windows_per_batch=5, confirm=False):
    """
    Process a JSON file with perplexity data and add infinigram counts
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional)
        index (str): The infinigram index to use
        max_workers (int): Maximum number of concurrent requests for entries
        max_retries (int): Maximum retry attempts per request
        initial_backoff (float): Initial backoff time in seconds
        remove_first_word (bool): Whether to remove the first token, such that infinigram can match successfully
        max_windows_per_batch (int): Maximum number of windows to process concurrently within an entry
        confirm (bool): Whether to ask for confirmation before proceeding
    """
    if output_file is None:
        # Create output filename based on input if not provided
        output_file = input_file.rsplit('.', 1)[0] + '_with_counts.json'
    
    try:
        # Load the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Count total requests and ask for confirmation if requested
        if confirm:
            total_entries = len(data)
            total_windows = count_total_windows(data)
            
            print(f"\nTotal entries: {total_entries}")
            print(f"Total windows/texts to be processed: {total_windows}")
            print(f"API requests required: approximately {total_windows} (not counting retries)")
            print(f"Processing with up to {max_workers} concurrent workers")
            
            confirmation = input("\nProceed with processing? (y/N): ")
            if confirmation.lower() not in ['y', 'yes']:
                print("Operation cancelled by user.")
                return
        
        # Process entries in parallel
        print(f"Processing {len(data)} entries with up to {max_workers} concurrent requests...")
        
        # Create a partial function with the fixed parameters
        process_func = partial(
            process_entry, 
            index=index, 
            max_retries=max_retries, 
            initial_backoff=initial_backoff, 
            remove_first_word=remove_first_word,
            max_workers=max_windows_per_batch
        )
        
        # Use ThreadPoolExecutor for I/O-bound tasks like API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and get a map of futures to entries
            futures = {executor.submit(process_func, entry): i for i, entry in enumerate(data)}
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                idx = futures[future]
                try:
                    data[idx] = future.result()
                except Exception as e:
                    print(f"Error processing entry {idx}: {str(e)}")
        
        # Save the updated data
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Updated data saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Check infini-gram API for low perplexity text matches')
    parser.add_argument('input_file', help='Input JSON file with perplexity data')
    parser.add_argument('--output', '-o', help='Output JSON file (default: input_file_with_counts.json)')
    parser.add_argument('--index', '-i', default='v4_piletrain_llama', 
                        help='Infinigram index to use (default: v4_piletrain_llama)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Maximum number of concurrent requests (default: 10)')
    parser.add_argument('--retries', '-r', type=int, default=5,
                        help='Maximum number of retry attempts per request (default: 5)')
    parser.add_argument('--backoff', '-b', type=float, default=1.0,
                        help='Initial backoff time in seconds for retries (default: 1.0)')
    parser.add_argument('--remove-first-word', action='store_true', default=True, 
                        help='Remove the first word from the text to match infinigram (default: True)')
    parser.add_argument('--max-windows-per-batch', type=int, default=5,
                        help='Maximum number of windows to process concurrently within an entry (default: 5)')
    parser.add_argument('--confirm', '-c', action='store_true',
                        help='Show number of requests and ask for confirmation before proceeding')
    
    args = parser.parse_args()
    
    process_json_file(
        args.input_file, 
        args.output, 
        args.index, 
        args.workers, 
        args.retries, 
        args.backoff, 
        args.remove_first_word,
        args.max_windows_per_batch,
        args.confirm
    )

if __name__ == "__main__":
    main()