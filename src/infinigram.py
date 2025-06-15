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
import logging
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfinigramProcessor:
    """Modular class for processing files with Infinigram API"""
    
    def __init__(self, index="v4_piletrain_llama", max_retries=5, initial_backoff=1.0, 
                 max_workers=10, max_windows_per_batch=5):
        self.index = index
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_workers = max_workers
        self.max_windows_per_batch = max_windows_per_batch
        
    def query_infinigram(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Query the infini-gram API to get count of a text phrase with retry logic
        
        Args:
            text (str): The text to search for
            
        Returns:
            dict: API response with count and other info, or None if failed
        """
        payload = {
            'index': self.index,
            'query_type': 'count',
            'query': text.strip(),
        }
        
        retry_count = 0
        backoff = self.initial_backoff
        
        while retry_count <= self.max_retries:
            try:
                response = requests.post('https://api.infini-gram.io/', json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Check if there's an error in the response
                if 'error' in result:
                    raise requests.exceptions.RequestException(f"API Error: {result['error']}")
                    
                return result
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                
                if retry_count > self.max_retries:
                    logger.warning(f"Failed after {self.max_retries} retries for '{text[:30]}...': {str(e)}")
                    return None
                
                # Add jitter to backoff to prevent thundering herd problem
                sleep_time = backoff + random.uniform(0, 0.5)
                logger.debug(f"Retry {retry_count}/{self.max_retries} for '{text[:30]}...' after {sleep_time:.2f}s: {str(e)}")
                time.sleep(sleep_time)
                
                # Exponential backoff with jitter
                backoff = min(backoff * 2, 30)  # Cap at 30 seconds
        
        return None

    def extract_text_from_tokens(self, tokens: List[str]) -> str:
        """Convert token list to text string"""
        return "".join(tokens)

    def process_region_v2(self, region: Dict[str, Any], experiment_dir: str = None) -> Dict[str, Any]:
        """
        Process a single region from Version 2 structure
        
        Args:
            region: Region dictionary with tokens, token_ids, etc.
            experiment_dir: Base experiment directory for additional context
        """
        if 'tokens' not in region:
            logger.warning(f"Region {region.get('region_id', 'unknown')} missing tokens")
            return region
            
        # Extract text from tokens
        window_text = self.extract_text_from_tokens(region['tokens'])
        
        if not window_text.strip():
            logger.debug(f"Empty text for region {region.get('region_id', 'unknown')}")
            return region

        # Query the API
        result = self.query_infinigram(window_text)
        
        if result and 'error' not in result:
            # Add the count and approx info to the region
            region['infinigram_count'] = result.get('count', 0)
            region['infinigram_approx'] = result.get('approx', False)
            region['infinigram_tokens'] = result.get('tokens', [])
            region['infinigram_query_text'] = window_text
        else:
            # If there was an error, set count to 0
            region['infinigram_count'] = 0
            region['infinigram_approx'] = False
            region['infinigram_error'] = True if result and 'error' in result else "API request failed"
            region['infinigram_query_text'] = window_text
        
        return region

    def process_generation_v2(self, generation_data: Dict[str, Any], experiment_dir: str = None) -> Dict[str, Any]:
        """
        Process all regions in a generation from Version 2 structure
        
        Args:
            generation_data: Dictionary containing per_gen_regions
            experiment_dir: Base experiment directory
        """
        if 'per_gen_regions' not in generation_data:
            return generation_data
            
        regions = generation_data['per_gen_regions']
        
        # Process regions with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_windows_per_batch) as executor:
            # Create a partial function for processing regions
            process_func = partial(self.process_region_v2, experiment_dir=experiment_dir)
            
            # Submit all regions for processing
            futures = {executor.submit(process_func, region): i for i, region in enumerate(regions)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    regions[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing region {idx}: {str(e)}")
        
        # Update generation data with processed regions
        generation_data['per_gen_regions'] = regions
        
        return generation_data

    def process_prompt_v2(self, prompt_data: Dict[str, Any], experiment_dir: str = None) -> Dict[str, Any]:
        """
        Process all generations for a prompt from Version 2 structure
        
        Args:
            prompt_data: Dictionary containing per_prompt_regions
            experiment_dir: Base experiment directory
        """
        if 'per_prompt_regions' not in prompt_data:
            return prompt_data
            
        generations = prompt_data['per_prompt_regions']
        
        # Process generations sequentially to avoid overwhelming the API
        for i, generation_data in enumerate(generations):
            try:
                generations[i] = self.process_generation_v2(generation_data, experiment_dir)
            except Exception as e:
                logger.error(f"Error processing generation {i}: {str(e)}")
        
        # Update prompt data with processed generations
        prompt_data['per_prompt_regions'] = generations
        
        return prompt_data

    def process_experiment_v2(self, experiment_dir: str, update_in_place: bool = True) -> bool:
        """
        Process an entire Version 2 experiment directory
        
        Args:
            experiment_dir: Path to experiment directory
            update_in_place: Whether to modify files in place or create new ones
            
        Returns:
            bool: Success status
        """
        logger.info(f"Processing Version 2 experiment: {experiment_dir}")
        
        # Find all low perplexity region files
        perp_regions_dir = os.path.join(experiment_dir, 'perplexity_analysis', 'low_perp_regions')
        
        if not os.path.exists(perp_regions_dir):
            logger.error(f"Perplexity regions directory not found: {perp_regions_dir}")
            return False
        
        region_files = [f for f in os.listdir(perp_regions_dir) 
                       if f.startswith('prompt_') and f.endswith('.json')]
        
        if not region_files:
            logger.warning(f"No region files found in {perp_regions_dir}")
            return False
        
        logger.info(f"Found {len(region_files)} region files to process")
        
        # Process each file
        for region_file in tqdm(region_files, desc="Processing region files"):
            region_path = os.path.join(perp_regions_dir, region_file)
            
            try:
                # Load the region data
                with open(region_path, 'r', encoding='utf-8') as f:
                    region_data = json.load(f)
                
                # Process the prompt data
                processed_data = self.process_prompt_v2(region_data, experiment_dir)
                
                # Save the updated data
                output_path = region_path if update_in_place else region_path.replace('.json', '_with_counts.json')
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Successfully processed and saved {region_file}")
                
            except Exception as e:
                logger.error(f"Error processing {region_file}: {str(e)}")
                continue
        
        logger.info(f"Completed processing experiment: {experiment_dir}")
        return True

    def process_single_file_v2(self, file_path: str, update_in_place: bool = True) -> bool:
        """
        Process a single Version 2 low perplexity regions file
        
        Args:
            file_path: Path to the JSON file
            update_in_place: Whether to modify file in place
            
        Returns:
            bool: Success status
        """
        logger.info(f"Processing single file: {file_path}")
        
        try:
            # Load the data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process the data
            processed_data = self.process_prompt_v2(data)
            
            # Save the updated data
            output_path = file_path if update_in_place else file_path.replace('.json', '_with_counts.json')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed and saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def count_total_regions_v2(self, data: Dict[str, Any]) -> int:
        """Count total regions in Version 2 format"""
        total_regions = 0
        
        if 'per_prompt_regions' in data:
            for generation_data in data['per_prompt_regions']:
                if 'per_gen_regions' in generation_data:
                    total_regions += len(generation_data['per_gen_regions'])
        
        return total_regions

    def count_experiment_regions_v2(self, experiment_dir: str) -> int:
        """Count total regions in a Version 2 experiment"""
        perp_regions_dir = os.path.join(experiment_dir, 'perplexity_analysis', 'low_perp_regions')
        
        if not os.path.exists(perp_regions_dir):
            return 0
        
        total_regions = 0
        region_files = [f for f in os.listdir(perp_regions_dir) 
                       if f.startswith('prompt_') and f.endswith('.json')]
        
        for region_file in region_files:
            region_path = os.path.join(perp_regions_dir, region_file)
            try:
                with open(region_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                total_regions += self.count_total_regions_v2(data)
            except Exception as e:
                logger.warning(f"Could not count regions in {region_file}: {e}")
        
        return total_regions

    # Legacy methods for backward compatibility
    def process_window_legacy(self, window, experiment_dir=""):
        """Process a single window entry with the API (legacy format)"""
        window_text = window.get('window_text', '')
        window['in_prompt'] = False

        if not window_text:
            return window

        # Query the API
        result = self.query_infinigram(window_text)
        
        if result and 'error' not in result:
            window['infinigram_count'] = result.get('count', 0)
            window['infinigram_approx'] = result.get('approx', False)
            window['infinigram_tokens'] = result.get('tokens', [])
        else:
            window['infinigram_count'] = 0
            window['infinigram_approx'] = False
            window['infinigram_error'] = True if result and 'error' in result else "API request failed"
        
        return window

def process_json_file(input_file, output_file=None, index="v4_piletrain_llama", 
                     max_workers=10, max_retries=5, initial_backoff=1.0,
                     max_windows_per_batch=5, confirm=False, update_in_place=True):
    """
    Process a JSON file with perplexity data and add infinigram counts
    Supports both legacy format and Version 2 format
    """
    if not update_in_place and output_file is None:
        # Create output filename based on input if not provided
        output_file = input_file.rsplit('.', 1)[0] + '_with_counts.json'
    elif update_in_place:
        output_file = input_file
    
    # Initialize processor
    processor = InfinigramProcessor(
        index=index,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        max_workers=max_workers,
        max_windows_per_batch=max_windows_per_batch
    )
    
    try:
        # Load and inspect the JSON data to determine format
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine if this is Version 2 format
        is_v2_format = ('per_prompt_regions' in data or 
                       (isinstance(data, list) and len(data) > 0 and 'per_prompt_regions' in data[0]))
        
        if is_v2_format:
            logger.info("Detected Version 2 format")
            
            if isinstance(data, list):
                # Multiple prompt files loaded as list
                logger.error("Multiple prompt processing not supported in single file mode. Use experiment directory mode.")
                return False
            else:
                # Single prompt file
                total_regions = processor.count_total_regions_v2(data)
                
                if confirm:
                    print(f"\nVersion 2 format detected")
                    print(f"Total regions to process: {total_regions}")
                    print(f"Processing with up to {max_workers} concurrent workers")
                    
                    confirmation = input("\nProceed with processing? (y/N): ")
                    if confirmation.lower() not in ['y', 'yes']:
                        print("Operation cancelled by user.")
                        return False
                
                success = processor.process_single_file_v2(input_file, update_in_place)
                if success:
                    print(f"Updated data saved to {output_file}")
                return success
        else:
            logger.info("Detected legacy format")
            # Fall back to legacy processing
            return process_legacy_format(data, input_file, output_file, processor, confirm)
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False

def process_legacy_format(data, input_file, output_file, processor, confirm):
    """Process legacy format files"""
    # Count total requests and ask for confirmation if requested
    if confirm:
        total_entries = len(data) if isinstance(data, list) else 1
        
        print(f"\nLegacy format detected")
        print(f"Total entries: {total_entries}")
        print(f"Processing with legacy method")
        
        confirmation = input("\nProceed with processing? (y/N): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("Operation cancelled by user.")
            return False
    
    # Process with legacy method (simplified)
    if isinstance(data, list):
        for i, entry in enumerate(tqdm(data, desc="Processing entries")):
            try:
                if 'window_text' in entry:
                    data[i] = processor.process_window_legacy(entry)
            except Exception as e:
                logger.error(f"Error processing entry {i}: {str(e)}")
    
    # Save the updated data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return True

def process_experiment_directory(experiment_dir, index="v4_piletrain_llama", 
                               max_workers=10, max_retries=5, initial_backoff=1.0,
                               max_windows_per_batch=5, confirm=False, update_in_place=True):
    """
    Process an entire Version 2 experiment directory
    """
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return False
    
    # Initialize processor
    processor = InfinigramProcessor(
        index=index,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        max_workers=max_workers,
        max_windows_per_batch=max_windows_per_batch
    )
    
    if confirm:
        total_regions = processor.count_experiment_regions_v2(experiment_dir)
        print(f"\nExperiment directory: {experiment_dir}")
        print(f"Total regions to process: {total_regions}")
        print(f"Processing with up to {max_workers} concurrent workers")
        print(f"Update in place: {update_in_place}")
        
        confirmation = input("\nProceed with processing? (y/N): ")
        if confirmation.lower() not in ['y', 'yes']:
            print("Operation cancelled by user.")
            return False
    
    return processor.process_experiment_v2(experiment_dir, update_in_place)

def main():
    parser = argparse.ArgumentParser(description='Check infini-gram API for low perplexity text matches')
    parser.add_argument('input_path', help='Input JSON file or experiment directory')
    parser.add_argument('--output', '-o', help='Output JSON file (only for single file mode)')
    parser.add_argument('--index', '-i', default='v4_piletrain_llama', 
                        help='Infinigram index to use (default: v4_piletrain_llama)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='Maximum number of concurrent requests (default: 10)')
    parser.add_argument('--retries', '-r', type=int, default=5,
                        help='Maximum number of retry attempts per request (default: 5)')
    parser.add_argument('--backoff', '-b', type=float, default=1.0,
                        help='Initial backoff time in seconds for retries (default: 1.0)')
    parser.add_argument('--max-windows-per-batch', type=int, default=5,
                        help='Maximum number of windows to process concurrently within an entry (default: 5)')
    parser.add_argument('--confirm', '-c', action='store_true',
                        help='Show number of requests and ask for confirmation before proceeding')
    parser.add_argument('--create-copy', action='store_true',
                        help='Create copies with _with_counts suffix instead of updating in place')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    update_in_place = not args.create_copy
    
    # Determine if input is a file or directory
    if os.path.isfile(args.input_path):
        # Single file mode
        success = process_json_file(
            args.input_path, 
            args.output, 
            args.index, 
            args.workers, 
            args.retries, 
            args.backoff, 
            args.max_windows_per_batch,
            args.confirm,
            update_in_place
        )
    elif os.path.isdir(args.input_path):
        # Experiment directory mode
        if args.output:
            logger.warning("Output argument ignored in directory mode")
        
        success = process_experiment_directory(
            args.input_path,
            args.index,
            args.workers,
            args.retries,
            args.backoff,
            args.max_windows_per_batch,
            args.confirm,
            update_in_place
        )
    else:
        logger.error(f"Input path not found: {args.input_path}")
        success = False
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed!")
        exit(1)

if __name__ == "__main__":
    main()