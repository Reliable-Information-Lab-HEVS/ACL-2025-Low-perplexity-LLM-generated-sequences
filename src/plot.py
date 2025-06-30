import argparse
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm
import html

class TokenPerplexityVisualizer:
    """Enhanced visualization class for token perplexities with support for Version 2 data structure"""
    
    def __init__(self, cmap_name="jet", font_size_px=16, line_height_factor=2.0):
        self.cmap_name = cmap_name
        self.font_size_px = font_size_px
        self.line_height_factor = line_height_factor
        
    def get_colormap(self, cmap_name: str = None):
        """Get matplotlib colormap with fallback"""
        if cmap_name is None:
            cmap_name = self.cmap_name
            
        try:
            return matplotlib.cm.get_cmap(cmap_name)
        except ValueError:
            print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis_r'.")
            return matplotlib.cm.get_cmap("viridis_r")
    
    def get_style_block(self) -> str:
        """Generate CSS style block for HTML visualization"""
        return f"""
<style>
    .token-perplexity-visualization-block {{
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #fdfdfd;
        margin-bottom: 20px;
        font-family: Arial, sans-serif;
    }}
    .token-perplexity-visualization-block h2 {{
        color: #333;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.4em;
    }}
    .token-perplexity-visualization-block h3 {{
        color: #555;
        margin: 15px 0 8px 0;
        font-size: 1.1em;
        font-weight: normal;
    }}
    .prompt-section {{
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #4a90e2;
        margin-bottom: 15px;
    }}
    .generation-section {{
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin-bottom: 15px;
    }}
    .tokens-container {{
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: {self.font_size_px}px;
        line-height: {self.font_size_px * self.line_height_factor}px;
        text-align: left;
        word-wrap: break-word;
        margin: 10px 0;
    }}
    .low-perp-region {{
        border: 2px solid #ff6b6b !important;
        box-shadow: 0 0 5px rgba(255, 107, 107, 0.3);
    }}
    .prompt-token {{
        opacity: 0.8;
        border-bottom: 2px dotted #666;
    }}
    .color-bar-area {{
        margin-top: 15px;
        font-size: 0.8em;
        color: #444;
    }}
    .color-bar-gradient {{
        display: flex;
        height: 20px;
        border: 1px solid #ccc;
        border-radius: 3px;
        margin: 5px 0;
    }}
    .color-bar-label {{
        display: flex;
        justify-content: space-between;
    }}
    .stats-section {{
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 0.9em;
    }}
    .generation-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    .region-info {{
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 3px;
        padding: 5px 8px;
        margin: 5px 0;
        font-size: 0.85em;
    }}
</style>
"""
    
    def normalize_perplexities(self, perplexities: List[float]) -> Tuple[np.ndarray, float, float]:
        """Normalize perplexities to 0-1 range"""
        p_array = np.array(perplexities)
        min_p = np.min(p_array)
        max_p = np.max(p_array)
        
        if min_p == max_p:
            norm_perplexities = np.full_like(p_array, 0.5, dtype=float)
        else:
            norm_perplexities = (p_array - min_p) / (max_p - min_p)
            
        return norm_perplexities, min_p, max_p
    
    def create_token_span(self, token: str, perplexity: float, norm_perplexity: float, 
                         cmap, is_low_perp: bool = False, is_prompt: bool = False) -> str:
        """Create HTML span for a single token"""
        # Get color from colormap
        rgba_color = cmap(norm_perplexity)
        rgb_tuple = tuple(int(x * 255) for x in rgba_color[:3])
        css_bg_color = f"rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})"
        
        # Determine text color for contrast
        luminance = 0.2126 * rgba_color[0] + 0.7152 * rgba_color[1] + 0.0722 * rgba_color[2]
        css_text_color = "black" if luminance > 0.5 else "white"
        
        escaped_token = html.escape(token)
        
        # Base styles
        span_style = (
            f"background-color: {css_bg_color}; "
            f"color: {css_text_color}; "
            "padding: 0.15em 0.4em; "
            "margin: 0.1em; "
            "border-radius: 4px; "
            "display: inline-block; "
            "white-space: pre-wrap; "
        )
        
        # Add special styling for low perplexity regions
        css_classes = []
        if is_low_perp:
            css_classes.append("low-perp-region")
        if is_prompt:
            css_classes.append("prompt-token")
            
        class_attr = f' class="{" ".join(css_classes)}"' if css_classes else ""
        
        span_title = f"Perplexity: {perplexity:.3f}"
        if is_prompt:
            span_title += " (Prompt)"
        if is_low_perp:
            span_title += " (Low Perplexity Region)"
            
        return f'<span style="{span_style}"{class_attr} title="{span_title}">{escaped_token}</span>'
    
    def create_color_bar(self, min_p: float, max_p: float, cmap) -> str:
        """Create HTML color bar"""
        gradient_segments_html = []
        num_segments = 20
        
        for i in range(num_segments):
            norm_val = i / (num_segments - 1)
            rgba_color_cb = cmap(norm_val)
            css_bg_color_cb = f"rgb({int(rgba_color_cb[0]*255)}, {int(rgba_color_cb[1]*255)}, {int(rgba_color_cb[2]*255)})"
            gradient_segments_html.append(f'<div style="flex-grow: 1; background-color: {css_bg_color_cb};"></div>')
        
        return f"""
<div class='color-bar-area'>
    <div>Perplexity Range</div>
    <div class='color-bar-gradient'>
        {''.join(gradient_segments_html)}
    </div>
    <div class='color-bar-label'>
        <span>{min_p:.2f}</span>
        <span>{max_p:.2f}</span>
    </div>
</div>
"""
    
    def get_low_perp_indices(self, regions: List[Dict[str, Any]]) -> set:
        """Get set of token indices that are in low perplexity regions"""
        low_perp_indices = set()
        for region in regions:
            start_idx = region.get('start_index', region.get('start_token_idx', 0))
            end_idx = region.get('end_index', region.get('end_token_idx', 0))
            low_perp_indices.update(range(start_idx, end_idx))
        return low_perp_indices
    
    def create_region_info(self, regions: List[Dict[str, Any]], prompt_text: str = "") -> str:
        """Create HTML info about low perplexity regions"""
        if not regions:
            return ""
            
        region_info_html = []
        for i, region in enumerate(regions):
            region_id = region.get('region_id', f'Region {i+1}')
            avg_perp = region.get('avg_perplexity', region.get('average_perplexity', 0))
            min_perp = region.get('min_perplexity', 0)
            max_perp = region.get('max_perplexity', 0)
            start_idx = region.get('start_index', region.get('start_token_idx', 0))
            end_idx = region.get('end_index', region.get('end_token_idx', 0))
            
            # Extract region text
            region_text = ""
            if 'tokens' in region:
                region_text = ''.join(region['tokens'])
            elif 'text' in region:
                region_text = region['text']
            elif 'window_text' in region:
                region_text = region['window_text']
            
            # Check if region text is contained in prompt
            is_in_prompt = False
            if prompt_text and region_text:
                # Clean up text for comparison (remove extra whitespace)
                clean_prompt = ' '.join(prompt_text.split())
                clean_region = ' '.join(region_text.split())
                is_in_prompt = clean_region.lower() in clean_prompt.lower()
            
            # Determine background color based on whether text is in prompt
            bg_color = "#ffebee" if is_in_prompt else "#fff3cd"  # Light red if in prompt, yellow otherwise
            border_color = "#f8bbd9" if is_in_prompt else "#ffeaa7"  # Darker red/yellow borders
            
            info_text = f"{region_id}: Indices {start_idx}-{end_idx}, Avg: {avg_perp:.3f}, Range: {min_perp:.3f}-{max_perp:.3f}"
            
            # Add infinigram info if available
            if 'infinigram_count' in region:
                count = region['infinigram_count']
                approx = " (approx)" if region.get('infinigram_approx', False) else ""
                info_text += f", Infinigram: {count}{approx}"
            
            # Add standalone perplexity info if available
            if 'standalone_avg_perplexity' in region:
                standalone_perp = region['standalone_avg_perplexity']
                info_text += f", Standalone: {standalone_perp:.3f}"
                if 'perplexity_ratio' in region:
                    ratio = region['perplexity_ratio']
                    info_text += f", Ratio: {ratio:.3f}"
            
            # Create region info div with text display
            region_html = f"""
<div class="region-info" style="background-color: {bg_color}; border: 1px solid {border_color};">
    <div style="font-weight: bold; margin-bottom: 3px;">{info_text}</div>
    <div style="font-family: monospace; font-size: 0.9em; padding: 3px 5px; background-color: rgba(0,0,0,0.05); border-radius: 2px; margin-top: 3px;">
        <strong>Text:</strong> {html.escape(region_text) if region_text else '(No text available)'}
    </div>
    {f'<div style="font-size: 0.8em; color: #d32f2f; font-style: italic; margin-top: 2px;">⚠️ Found in prompt</div>' if is_in_prompt else ''}
</div>
"""
            region_info_html.append(region_html)
        
        return f"""
<div>
    <strong>Low Perplexity Regions ({len(regions)} found):</strong>
    {''.join(region_info_html)}
</div>
"""
    
    def visualize_single_generation(self, tokens: List[str], perplexities: List[float], 
                                  regions: List[Dict[str, Any]] = None, 
                                  prompt_token_count: int = 0,
                                  generation_id: str = None,
                                  show_regions_info: bool = True,
                                  prompt_text: str = "") -> str:
        """Visualize a single generation with its tokens and perplexities"""
        if not tokens or not perplexities:
            return "<p>(No tokens to display)</p>"
            
        if len(tokens) != len(perplexities):
            raise ValueError("Number of tokens and perplexities must match.")
        
        # Get low perplexity regions
        low_perp_indices = set()
        if regions:
            low_perp_indices = self.get_low_perp_indices(regions)
        
        # Normalize perplexities
        norm_perplexities, min_p, max_p = self.normalize_perplexities(perplexities)
        cmap = self.get_colormap()
        
        # Create token spans
        token_spans_html = []
        for i, token in enumerate(tokens):
            is_low_perp = i in low_perp_indices
            is_prompt = i < prompt_token_count
            
            span = self.create_token_span(
                token, perplexities[i], norm_perplexities[i], 
                cmap, is_low_perp, is_prompt
            )
            token_spans_html.append(span)
        
        # Generation header
        header_html = ""
        if generation_id:
            stats = f"Tokens: {len(tokens)}, Avg Perplexity: {np.mean(perplexities):.3f}"
            header_html = f"""
<div class="generation-header">
    <h3>{generation_id}</h3>
    <span class="stats">{stats}</span>
</div>
"""
        
        # Tokens container
        tokens_html = f"""
<div class='tokens-container'>
    {''.join(token_spans_html)}
</div>
"""
        
        # Region info (now includes prompt_text for comparison)
        region_info_html = ""
        if show_regions_info and regions:
            region_info_html = self.create_region_info(regions, prompt_text)
        
        # Color bar
        color_bar_html = self.create_color_bar(min_p, max_p, cmap)
        
        return f"""
<div class="generation-section">
    {header_html}
    {tokens_html}
    {region_info_html}
    {color_bar_html}
</div>
"""
    
    def visualize_prompt_data_v2(self, prompt_data: Dict[str, Any], 
                               show_prompt: bool = True,
                               show_regions_info: bool = True,
                               max_generations: int = None) -> str:
        """Visualize prompt data in Version 2 format"""
        
        # Extract metadata
        prompt_metadata = prompt_data.get('prompt_metadata', {})
        prompt_id = prompt_metadata.get('prompt_id', 'Unknown')
        prompt_text = prompt_metadata.get('prompt_text', '')
        
        # Title
        title_html = f"<h2>Prompt Analysis: {prompt_id}</h2>"
        
        # Prompt section
        prompt_html = ""
        if show_prompt and prompt_text:
            prompt_html = f"""
<div class="prompt-section">
    <h3>Prompt</h3>
    <div style="font-family: monospace; padding: 5px;">{html.escape(prompt_text)}</div>
</div>
"""
        
        # Process generations
        generations_html = []
        per_prompt_regions = prompt_data.get('per_prompt_regions', [])
        
        generation_count = len(per_prompt_regions)
        if max_generations:
            generation_count = min(generation_count, max_generations)
        
        for i, generation_data in enumerate(per_prompt_regions[:generation_count]):
            generation_id = generation_data.get('generation_id', f'Generation {i+1}')
            regions = generation_data.get('per_gen_regions', [])
            
            # We need to get tokens and perplexities from the corresponding generation file
            # For now, we'll create a placeholder - this would need to be linked with generation data
            generation_html = f"""
<div class="generation-section">
    <h3>{generation_id}</h3>
    <p>Generation data needs to be loaded from: inference_data/generations/</p>
    {self.create_region_info(regions, '') if show_regions_info else ""}
</div>
"""
            generations_html.append(generation_html)
        
        # Stats
        total_regions = sum(len(gen.get('per_gen_regions', [])) for gen in per_prompt_regions)
        stats_html = f"""
<div class="stats-section">
    <strong>Summary:</strong> {len(per_prompt_regions)} generations, {total_regions} low perplexity regions total
</div>
"""
        
        return f"""
{self.get_style_block()}
<div class='token-perplexity-visualization-block'>
    {title_html}
    {prompt_html}
    {stats_html}
    {''.join(generations_html)}
</div>
"""
    
    def visualize_generation_with_data(self, generation_data: Dict[str, Any], 
                                     region_data: List[Dict[str, Any]] = None,
                                     prompt_text: str = "",
                                     show_prompt: bool = True,
                                     show_regions_info: bool = True) -> str:
        """Visualize generation with full token data"""
        
        generation_id = generation_data.get('generation_id', 'Unknown Generation')
        tokens = generation_data.get('token_texts', generation_data.get('tokens', []))
        perplexities = generation_data.get('token_perplexities', generation_data.get('perplexities', []))
        
        if not tokens or not perplexities:
            return f"<p>No token data available for {generation_id}</p>"
        
        # Calculate prompt token count
        prompt_token_count = 0
        if prompt_text:
            # Rough estimation - could be improved with actual tokenization
            prompt_token_count = len(prompt_text.split())
        
        # Title
        title_html = f"<h2>Generation Analysis: {generation_id}</h2>"
        
        # Prompt section
        prompt_html = ""
        if show_prompt and prompt_text:
            prompt_html = f"""
<div class="prompt-section">
    <h3>Prompt</h3>
    <div style="font-family: monospace; padding: 5px;">{html.escape(prompt_text)}</div>
</div>
"""
        
        # Generation visualization
        generation_html = self.visualize_single_generation(
            tokens, perplexities, region_data, 
            prompt_token_count, generation_id, show_regions_info, prompt_text
        )
        
        return f"""
{self.get_style_block()}
<div class='token-perplexity-visualization-block'>
    {title_html}
    {prompt_html}
    {generation_html}
</div>
"""
    
    def load_and_visualize_experiment(self, experiment_dir: str, prompt_id: str = None,
                                    generation_id: str = None, show_prompt: bool = True,
                                    show_regions_info: bool = True, max_generations: int = None) -> str:
        """Load and visualize data from Version 2 experiment directory"""
        
        if not os.path.exists(experiment_dir):
            return f"<p>Experiment directory not found: {experiment_dir}</p>"
        
        # Find files to process
        generations_dir = os.path.join(experiment_dir, 'inference_data', 'generations')
        regions_dir = os.path.join(experiment_dir, 'perplexity_analysis', 'low_perp_regions')
        
        if not os.path.exists(generations_dir) or not os.path.exists(regions_dir):
            return f"<p>Required directories not found in {experiment_dir}</p>"
        
        # Get list of prompt files
        generation_files = [f for f in os.listdir(generations_dir) if f.startswith('prompt_') and f.endswith('.json')]
        region_files = [f for f in os.listdir(regions_dir) if f.startswith('prompt_') and f.endswith('.json')]
        
        if not generation_files:
            return f"<p>No generation files found in {generations_dir}</p>"
        
        # If specific prompt_id requested, filter files
        if prompt_id:
            generation_files = [f for f in generation_files if prompt_id in f]
            region_files = [f for f in region_files if prompt_id in f]
        
        results_html = []
        
        for gen_file in sorted(generation_files)[:max_generations] if max_generations else sorted(generation_files):
            try:
                # Load generation data
                gen_path = os.path.join(generations_dir, gen_file)
                with open(gen_path, 'r', encoding='utf-8') as f:
                    gen_data = json.load(f)
                
                # Load corresponding region data
                region_file = gen_file.replace('prompt_', 'prompt_').replace('.json', '_regions.json')
                if region_file not in region_files:
                    # Try alternative naming
                    region_file = gen_file.replace('.json', '_low_perp.json')
                
                region_data = []
                if region_file in region_files:
                    region_path = os.path.join(regions_dir, region_file)
                    with open(region_path, 'r', encoding='utf-8') as f:
                        region_content = json.load(f)
                        
                    # Extract regions for all generations
                    if 'per_prompt_regions' in region_content:
                        for gen_regions in region_content['per_prompt_regions']:
                            region_data.extend(gen_regions.get('per_gen_regions', []))
                
                # Get prompt text
                prompt_text = gen_data.get('prompt_metadata', {}).get('prompt_text', '')
                
                # Visualize each generation
                for generation in gen_data.get('generations', []):
                    if generation_id and generation_id not in generation.get('generation_id', ''):
                        continue
                        
                    # Filter regions for this specific generation
                    gen_id = generation.get('generation_id', '')
                    gen_regions = [r for r in region_data if r.get('region_id', '').startswith(gen_id)]
                    
                    html_result = self.visualize_generation_with_data(
                        generation, gen_regions, prompt_text, show_prompt, show_regions_info
                    )
                    results_html.append(html_result)
                    
            except Exception as e:
                results_html.append(f"<p>Error processing {gen_file}: {str(e)}</p>")
        
        return '\n'.join(results_html)

# Legacy function for backward compatibility
def visualize_token_perplexity_html(
    tokens, perplexities, prompt="Token Perplexities:",
    cmap_name="jet", overall_title="Token Perplexity Visualization",
    show_color_bar=True, font_size_px=16, line_height_factor=2.0
):
    """Legacy function - use TokenPerplexityVisualizer class for new features"""
    visualizer = TokenPerplexityVisualizer(cmap_name, font_size_px, line_height_factor)
    
    if not tokens or not perplexities:
        return f"<div class='token-visualization'><h2>{html.escape(overall_title)}</h2><p>{html.escape(prompt)}</p><p>(No tokens to display)</p></div>"
    
    if len(tokens) != len(perplexities):
        raise ValueError("Number of tokens and perplexities must match.")
    
    norm_perplexities, min_p, max_p = visualizer.normalize_perplexities(perplexities)
    cmap = visualizer.get_colormap(cmap_name)
    
    token_spans_html = []
    for i, token in enumerate(tokens):
        span = visualizer.create_token_span(token, perplexities[i], norm_perplexities[i], cmap)
        token_spans_html.append(span)
    
    title_html = f"<h2>{html.escape(overall_title)}</h2>" if overall_title else ""
    prompt_html = f"<h3 class='prompt-title'>{html.escape(prompt)}</h3>"
    tokens_div_html = f"<div class='tokens-container'>{' '.join(token_spans_html)}</div>"
    color_bar_html = visualizer.create_color_bar(min_p, max_p, cmap) if show_color_bar else ""
    
    return f"""
{visualizer.get_style_block()}
<div class='token-perplexity-visualization-block'>
    {title_html}
    {prompt_html}
    {tokens_div_html}
    {color_bar_html}
</div>
"""

def main():
    """Command line interface for visualization"""
    parser = argparse.ArgumentParser(description='Visualize token perplexities from experiment data')
    parser.add_argument('input_path', help='Path to experiment directory or JSON file')
    parser.add_argument('--output', '-o', help='Output HTML file path')
    parser.add_argument('--prompt-id', help='Specific prompt ID to visualize')
    parser.add_argument('--generation-id', help='Specific generation ID to visualize')
    parser.add_argument('--no-prompt', action='store_true', help='Hide prompt text')
    parser.add_argument('--no-regions', action='store_true', help='Hide low perplexity region info')
    parser.add_argument('--max-generations', type=int, help='Maximum number of generations to show')
    parser.add_argument('--colormap', default='jet', help='Matplotlib colormap name')
    parser.add_argument('--font-size', type=int, default=16, help='Font size in pixels')
    
    args = parser.parse_args()
    
    visualizer = TokenPerplexityVisualizer(
        cmap_name=args.colormap,
        font_size_px=args.font_size
    )
    
    if os.path.isdir(args.input_path):
        # Experiment directory mode
        html_content = visualizer.load_and_visualize_experiment(
            args.input_path,
            prompt_id=args.prompt_id,
            generation_id=args.generation_id,
            show_prompt=not args.no_prompt,
            show_regions_info=not args.no_regions,
            max_generations=args.max_generations
        )
    elif os.path.isfile(args.input_path):
        # Single file mode
        try:
            with open(args.input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'per_prompt_regions' in data:
                # Region file
                html_content = visualizer.visualize_prompt_data_v2(
                    data,
                    show_prompt=not args.no_prompt,
                    show_regions_info=not args.no_regions,
                    max_generations=args.max_generations
                )
            elif 'generations' in data:
                # Generation file - would need to load corresponding region file
                html_content = "<p>Generation file visualization requires corresponding region file. Use experiment directory mode.</p>"
            else:
                html_content = "<p>Unrecognized file format</p>"
                
        except Exception as e:
            html_content = f"<p>Error loading file: {str(e)}</p>"
    else:
        html_content = f"<p>Input path not found: {args.input_path}</p>"
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Token Perplexity Visualization</title>
    <meta charset="utf-8">
</head>
<body>
{html_content}
</body>
</html>
""")
        print(f"Visualization saved to {args.output}")
    else:
        print(html_content)

if __name__ == "__main__":
    main()