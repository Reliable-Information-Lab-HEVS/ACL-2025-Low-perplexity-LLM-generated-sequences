#!/usr/bin/env python3
"""
Data Portraits analysis script for Slurm cluster
"""

import os
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_environment():
    """Set up the environment by installing dependencies"""
    os.system("git clone https://github.com/ruyimarone/data-portraits.git")
    os.chdir("data-portraits")
    os.system("./install_redis.sh")
    os.system("pip install -r requirements.txt")
    os.system("pip install -e .")
    os.system("python easy_redis.py --just-start")


def analyze_text_sample():
    """Run a sample analysis to verify setup"""
    import dataportraits
    # Download the model (about 26GB)
    portrait = dataportraits.from_hub("mmarone/dataportraits-pile", verbose=True)
    
    sample_text = """
    Johns Hopkins University is divided into nine schools, five of which serve undergraduates. 
    The Homewood Campus, one of the university's four campuses in and around Baltimore, 
    is the primary campus for undergraduates. Freshmen and sophomores are required to live on campus. 
    More than 1,300 students participate in the Greek community. Hopkins also has additional campuses 
    for its School of Advanced International Studies in Washington, D.C.; Bologna, Italy; and Nanjing, China. 
    Hopkins' graduate programs include the top-ranked Bloomberg School of Public Health and the highly ranked 
    School of Education, Whiting School of Engineering, School of Medicine and the well-regarded Peabody 
    Institute for music and dance. Johns Hopkins Hospital is a top-ranked hospital with highly ranked specialties.
    """
    
    report = portrait.contains_from_text([sample_text], sort_chains_by_length=True)
    print("Sample analysis result:")
    print(report[0]['chains'][0])
    return portrait


def analyze_generations(input_path, output_dir, exclude_pattern=None):
    """Analyze text generations from files matching the pattern"""
    import dataportraits
    
    # Load the model
    portrait = dataportraits.from_hub("mmarone/dataportraits-pile", verbose=True)
    
    # Collect files and generations
    generations = []
    files = []
    
    for filename in glob.glob(input_path):
        # Skip files matching the exclude pattern
        if exclude_pattern and exclude_pattern in filename:
            continue
            
        files.append(filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
            found_generated_text = False
            gen = ""
            for line in lines:
                if found_generated_text:
                    gen += line.strip()
                if line.startswith("Generated text:"):
                    found_generated_text = True
            generations.append(gen)
    
    print(f"Analyzing {len(generations)} text generations...")
    
    # Analyze with data portraits
    report = portrait.contains_from_text(generations, sort_chains_by_length=True)
    
    # Extract metrics
    n_matches = []
    length_original = []
    longest_match = []
    tot_length_matches = []
    badnesses = []
    
    for rep in report:
        badnesses.append(rep['badness'])
        length_original.append(len(rep['doc']))
        longest_match.append(rep['longest_chain'])
        n_matches.append(sum(rep['is_member']))
    
    # Create DataFrame
    data = {
        'original': files,
        'length_original': length_original,
        'longest_match': longest_match,
        'n_matches': n_matches,
        'badness': badnesses
    }
    
    df = pd.DataFrame(data)
    
    # Extract P values from filenames
    df['P_value'] = df['original'].apply(
        lambda x: re.search(r'-(\w+)_P', x).group(1) if re.search(r'-(\w+)_P', x) else None
    )
    
    # Remove rows with missing P values
    df_filtered = df.dropna(subset=['P_value'])
    
    # Save the DataFrame
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'analysis_results.csv'), index=False)
    
    # Create visualizations
    create_visualizations(df_filtered, output_dir)
    
    return df_filtered


def create_visualizations(df, output_dir):
    """Create and save visualizations of the analysis results"""
    # Create boxplot for badness scores
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='P_value', y='badness', data=df)
    plt.xlabel('P Value')
    plt.ylabel('Badness Score')
    plt.title('Distribution of Badness Scores by Prefix')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'badness_by_prefix.png'), dpi=300)
    
    # Create boxplot for longest match
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='P_value', y='longest_match', data=df)
    plt.xlabel('P Value')
    plt.ylabel('Longest Match')
    plt.title('Distribution of Longest Match by Prefix')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'longest_match_by_prefix.png'), dpi=300)
    
    # Create boxplot for number of matches
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='P_value', y='n_matches', data=df)
    plt.xlabel('P Value')
    plt.ylabel('Number of Matches')
    plt.title('Distribution of Number of Matches by Prefix')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'num_matches_by_prefix.png'), dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Data Portraits Analysis for Slurm')
    parser.add_argument('--input', type=str, required=True,
                        help='Input glob pattern for text files to analyze (e.g., "/path/to/prompts_morphine/*.txt")')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for results and visualizations')
    parser.add_argument('--exclude', type=str, default='ethyl',
                        help='Pattern to exclude from analysis (default: "ethyl")')
    parser.add_argument('--setup', action='store_true',
                        help='Run the initial setup (clone repo, install dependencies)')
    parser.add_argument('--sample', action='store_true',
                        help='Run a sample analysis to test the setup')
    
    args = parser.parse_args()
    
    if args.setup:
        print("Setting up the environment...")
        setup_environment()
    
    if args.sample:
        print("Running sample analysis...")
        analyze_text_sample()
    
    print(f"Analyzing files matching pattern: {args.input}")
    print(f"Excluding files containing: {args.exclude}")
    
    df = analyze_generations(args.input, args.output, args.exclude)
    
    # Print summary statistics
    print("\nSummary statistics by P value:")
    print(df.groupby('P_value')['longest_match'].describe())
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()