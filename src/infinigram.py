#!/usr/bin/env python3
import json
import requests
import argparse
from tqdm import tqdm
import time
import concurrent.futures
from functools import partial

def query_infinigram(text, index="v4_piletrain_llama"):
    """
    Query the infini-gram API to get count of a text phrase
    
    Args:
        text (str): The text to search for
        index (str): The index to search in
        
    Returns:
        dict: API response with count and other info
    """
    payload = { # See the documentation at https://infini-gram.readthedocs.io/en/latest/api.html
        'index': index,
        'query_type': 'count',
        'query': text.strip(),
    }
    
    try:
        response = requests.post('https://api.infini-gram.io/', json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying API for '{text}': {str(e)}")
        return None

def process_entry(entry, index):
    """Process a single entry with the API"""
    if 'longest_low_perplexity_text' in entry and entry['longest_low_perplexity_text']:
        result = query_infinigram(entry['longest_low_perplexity_text'], index)
        
        if result and 'error' not in result:
            entry['infinigram_count'] = result.get('count', 0)
            entry['infinigram_approx'] = result.get('approx', False)
            entry['infinigram_tokens'] = result.get('tokens', [])
        else:
            # If there was an error, set count to 0
            entry['infinigram_count'] = 0
            entry['infinigram_approx'] = False
            entry['infinigram_error'] = True if result and 'error' in result else "API request failed"
    
    return entry

def process_json_file(input_file, output_file=None, index="v4_piletrain_llama", max_workers=10):
    """
    Process a JSON file with perplexity data and add infinigram counts
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional)
        index (str): The infinigram index to use
        max_workers (int): Maximum number of concurrent requests
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
        process_func = partial(process_entry, index=index)
        
        # Use ProcessPoolExecutor for CPU-bound tasks, ThreadPoolExecutor for I/O-bound tasks like API calls
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
    
    args = parser.parse_args()
    
    process_json_file(args.input_file, args.output, args.index, args.workers)

if __name__ == "__main__":
    main()