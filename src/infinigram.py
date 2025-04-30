#!/usr/bin/env python3
import json
import requests
import argparse
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial
import random

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

def process_entry(entry, index, max_retries=5, initial_backoff=1.0, remove_first_word=False):
    """Process a single entry with the API    
    """
    if 'longest_low_perplexity_text' in entry and entry['longest_low_perplexity_text']:
        # Query the API
        
        if remove_first_word and len(entry['longest_low_perplexity_text'].split()) > 1:
            entry['longest_low_perplexity_text'] = ' '.join(entry['longest_low_perplexity_text'].split()[1:])
        
        result = query_infinigram(entry['longest_low_perplexity_text'], index, max_retries, initial_backoff)
        
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

def process_json_file(input_file, output_file=None, index="v4_piletrain_llama", 
                     max_workers=10, max_retries=5, initial_backoff=1.0, remove_first_word=False):
    """
    Process a JSON file with perplexity data and add infinigram counts
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional)
        index (str): The infinigram index to use
        max_workers (int): Maximum number of concurrent requests
        max_retries (int): Maximum retry attempts per request
        initial_backoff (float): Initial backoff time in seconds
        remove_first_word (bool): Whether to remove the first token, such that infinigram can match successfully (recommended)
    """
    if output_file is None:
        # Create output filename based on input if not provided
        output_file = input_file.rsplit('.', 1)[0] + '_with_counts.json'
    
    try:
        # Load the JSON data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Process entries in parallel
        print(f"Processing {len(data)} entries with up to {max_workers} concurrent requests...")
        
        # Create a partial function with the index parameter
        process_func = partial(process_entry, index=index, max_retries=max_retries, initial_backoff=initial_backoff, remove_first_word=remove_first_word)
        
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
    
    parser.add_argument('--remove-first-word', type=bool, default=True, help='If True, remove the first word from the text to match infinigram (default: True)') 
    # This is because sometimes infinigram will give a different answer for " A" and "A", so removing the first token will counter this effect !
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output, args.index, args.workers, args.retries, args.backoff, args.remove_first_word)

if __name__ == "__main__":
    main()