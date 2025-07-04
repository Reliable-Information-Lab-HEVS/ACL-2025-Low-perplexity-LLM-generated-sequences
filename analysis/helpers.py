import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def load_experiments(experiments_base_path, paths_dict):
    
    all_data = []
    
    for experiment_name in paths_dict.keys():
        experiment_dir = os.path.join(experiments_base_path, experiment_name)
        
        if not os.path.exists(experiment_dir):
            print(f"Warning: Experiment directory not found: {experiment_dir}")
            continue
        
        # Look for low perplexity region files
        regions_dir = os.path.join(experiment_dir, 'perplexity_analysis', 'low_perp_regions')
        
        if not os.path.exists(regions_dir):
            print(f"Warning: Regions directory not found: {regions_dir}")
            continue
        
        # Find all region files in this experiment
        region_files = glob.glob(os.path.join(regions_dir, 'prompt_*.json'))
        
        if not region_files:
            print(f"Warning: No region files found in: {regions_dir}")
            continue
        
        print(f"Processing experiment: {experiment_name} ({len(region_files)} files)")
        
        for region_file in region_files:
            try:
                # Load the region file
                with open(region_file, 'r', encoding='utf-8') as f:
                    region_data = json.load(f)
                
                # Extract experiment metadata
                source_prompt_id = region_data.get('source_prompt_id', 'unknown')
                analysis_params = region_data.get('analysis_parameters', {})
                
                # Load corresponding generation file to get model info and prompt
                prompt_number = region_file.split('prompt_')[-1].split('_')[0].split('.')[0]
                generation_file = os.path.join(experiment_dir, 'inference_data', 'generations', f'prompt_{prompt_number}.json')
                
                generation_data = {}
                if os.path.exists(generation_file):
                    with open(generation_file, 'r', encoding='utf-8') as f:
                        generation_data = json.load(f)
                
                # Extract model info and temperature
                model_info = generation_data.get('model_info', {})
                temperature = model_info.get('temperature', 0.7)
                model_name = model_info.get('model_name', 'unknown')
                
                # Check if this is a deduped experiment (based on experiment name or model info)
                deduped = 'deduped' in experiment_name or 'deduped' in model_name.lower()
                
                # Process each generation's regions
                for generation_regions in region_data.get('per_prompt_regions', []):
                    generation_id = generation_regions.get('generation_id', 'unknown')
                    
                    for region in generation_regions.get('per_gen_regions', []):
                        # Extract text from tokens
                        region_text = ''.join(region.get('tokens', []))
                        
                        # Create a dictionary for each data point
                        data_point = {
                            'file': region_file,
                            'experiment_name': experiment_name,
                            'dataset': experiment_name,  # Will be mapped later
                            'prompt_id': source_prompt_id,
                            'generation_id': generation_id,
                            'region_id': region.get('region_id', 'unknown'),
                            'deduped': deduped,
                            'infinigram_count': region.get('infinigram_count', 0),
                            'standalone_perplexity': region['standalone_avg_perplexity'],
                            'perplexity': region.get('avg_perplexity', 0),
                            'min_perplexity': region.get('min_perplexity', 0),
                            'max_perplexity': region.get('max_perplexity', 0),
                            'text': region_text,
                            'temperature': temperature,
                            'model_name': model_name,
                            'in_prompt': region.get('is_in_prompt', False),
                            'is_contiguous': region.get('is_contiguous', True),
                            'start_index': region.get('start_index', 0),
                            'end_index': region.get('end_index', 0),
                            'infinigram_approx': region.get('infinigram_approx', False)
                        }
                        all_data.append(data_point)
                
            except Exception as e:
                print(f"Error processing {region_file}: {e}")
                continue
        
        print(f"Processed {len([item for item in all_data if item['experiment_name'] == experiment_name])} regions from {experiment_name}")

    # Create a pandas DataFrame from the collected data
    df =  pd.DataFrame(all_data)

    if len(df) == 0:
        print("No data found! Check your experiment directories and file structure.")
    else:
        # Map experiment names to display names
        df['dataset'] = df['dataset'].map(paths_dict)
        
        # Create adjusted infinigram count (for log plotting)
        df['infinigram_count_adj'] = df['infinigram_count'].copy()
        df.loc[df['infinigram_count_adj'] == 0, 'infinigram_count_adj'] = 0.3
        
        # Define regions based on the conditions
        def categorize(row):
            if row['infinigram_count'] > 50:
                return "Frequently encountered text"
            elif 5 < row['infinigram_count'] <= 50:
                return "Segmental replication"
            elif 0.55 <= row['infinigram_count'] <= 5:
                return "Memorization"
            elif row['infinigram_count'] < 1:
                return "Synthetic coherence"
            else:
                return "Other"
        
        # Apply the function to categorize each row
        df['category'] = df.apply(categorize, axis=1)
    
    return df

def plot_scatter_with_color_gradients(df, figname):
    
    # I was struggling with the axis scale, so I ended up making the plot in linear scale but changing the x-value and the xticks, so that it appears log scaled.
    
    # Original boundaries (log scale)
    boundaries_log = np.array([0.1, 0.55, 5.5, 50, 1e6]) # 0.1 - 0.55 = STH, 0.55 - 5.5 = MEM, 5.5 - 50 = SEG, 50+ = FET
    # Map boundaries to linear axis by taking log10
    boundaries_linear = np.log10(boundaries_log)

    # Colors and names for regions
    region_colors = ["gray", "green", 'orange', 'red']
    region_names = ['STH', 'MEM', 'SEG', 'FET']

    # Centers of each region (in linear log space)
    centers_linear = [(boundaries_linear[i] + boundaries_linear[i+1]) / 2 for i in range(len(boundaries_linear) - 1)]

    # Normalize centers between 0 and 1 for colormap
    positions = [(c - boundaries_linear[0]) / (boundaries_linear[-1] - boundaries_linear[0]) for c in centers_linear]

    # Add 0 and 1 to positions and duplicate first/last colors to match colormap requirements
    positions = [0.0] + positions + [1.0]
    colors = [region_colors[0]] + region_colors + [region_colors[-1]]

    # Create colormap with colors centered on regions
    cmap = LinearSegmentedColormap.from_list("centered_gradient", list(zip(positions, colors)))

    # Create gradient data (1 x N array)
    n_points = 1000
    x_linear = np.linspace(boundaries_linear[0], boundaries_linear[-1], n_points).reshape(1, -1)

    x_counts = df['infinigram_count_adj'].values
    x_scatter = np.log10(x_counts)  # log10 transformed for linear axis

    y_data = df['standalone_perplexity'].apply(np.log2).values  # log2 transformed for y-axis

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show gradient background, extent correctly ordered [xmin, xmax, ymin, ymax]
    ax.imshow(x_linear, aspect='auto', cmap=cmap, alpha=0.3,
            extent=[boundaries_linear[0], boundaries_linear[-1], 0, 20], origin='lower')

    # Scatter points on the same x-axis (log10 counts)
    ax.scatter(x_scatter, y_data, s=6, alpha=0.7)

    # Vertical line at log10(0.55)
    ax.axvline(np.log10(0.55), color='r', linestyle='--')

    # Axis limits (linear)
    ax.set_xlim(-0.8, boundaries_linear[-1])
    ax.set_ylim(0, 20)

    ax.set_xlabel('Infinigram count $c$', fontsize=25)
    ax.set_ylabel(r'$\log_2($Standalone Perplexity $\hat{P})$', fontsize=25)

    # Custom ticks - original counts spaced by log10, but axis is linear
    tick_vals = [0.3, 1, 10, 100, 1000, 10000, 100000, 1e6]
    tick_pos = np.log10(tick_vals)
    tick_labels = ['$0$' if val < 1 else f'${val:g}$' if val < 1000 else f'$10^{{{int(np.log10(val))}}}$' for val in tick_vals]

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    ax.tick_params(axis='both', which='major', labelsize=25)

    # Legend patches with light transparency
    legend_patches = [Patch(color=c, alpha=0.15, label=l) for c, l in zip(region_colors, region_names)]
    ax.legend(handles=legend_patches)

    plt.tight_layout()

    if '/' in figname and not os.path.exists('/'.join(figname.split('/')[:-1])):
        os.makedirs('/'.join(figname.split('/')[:-1]))
    plt.savefig(figname)
    plt.show()