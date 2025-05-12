import argparse

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np

import html
import numpy as np
import matplotlib.cm
import matplotlib.colors

def visualize_token_perplexity_html(
    tokens,
    perplexities,
    prompt="Token Perplexities:",
    cmap_name="jet", # Using a reversed viridis often makes higher perplexity "hotter"
    overall_title="Token Perplexity Visualization",
    show_color_bar=True,
    font_size_px=16,
    line_height_factor=2.0, # Multiplier of font_size_px for line height
):
    """
    Generates an HTML string to visualize token perplexities with color highlighting.

    Parameters:
    -----------
    tokens : list
        List of word tokens as strings.
    perplexities : list
        List of perplexity values corresponding to each token.
    prompt : str, default="Token Perplexities:"
        Text to display directly above the token visualization.
    cmap_name : str, default='viridis_r'
        Matplotlib colormap name to use for highlighting.
    overall_title : str, default="Token Perplexity Visualization"
        Title for the entire HTML block.
    show_color_bar : bool, default=True
        Whether to display a simple HTML-based color bar.
    font_size_px : int, default=16
        Font size in pixels for the tokens.
    line_height_factor : float, default=2.0
        Multiplier for font_size_px to determine line-height for the token block.

    Returns:
    --------
    str : HTML string for the visualization.
    """
    if not tokens or not perplexities:
        if overall_title:
            title_html = f"<h2 style='font-family: sans-serif; color: #333;'>{html.escape(overall_title)}</h2>"
        else:
            title_html = ""
        prompt_html_str = f"<p style='font-family: sans-serif; color: #555;'>{html.escape(prompt)}</p>"
        return f"<div class='token-visualization'>{title_html}{prompt_html_str}<p>(No tokens to display)</p></div>"

    if len(tokens) != len(perplexities):
        raise ValueError("Number of tokens and perplexities must match.")

    # 1. Normalize perplexities
    p_array = np.array(perplexities)
    min_p = np.min(p_array)
    max_p = np.max(p_array)

    if min_p == max_p: # Avoid division by zero if all perplexities are the same
        norm_perplexities = np.full_like(p_array, 0.5, dtype=float)
    else:
        norm_perplexities = (p_array - min_p) / (max_p - min_p)

    # 2. Get colormap
    try:
        cmap = matplotlib.cm.get_cmap(cmap_name)
    except ValueError:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'viridis_r'.")
        cmap = matplotlib.cm.get_cmap("viridis_r")


    # 3. Generate HTML for each token
    token_spans_html = []
    for i, token_str in enumerate(tokens):
        norm_p = norm_perplexities[i]
        raw_p = perplexities[i]

        # Get color from colormap
        rgba_color = cmap(norm_p)
        rgb_tuple = tuple(int(x * 255) for x in rgba_color[:3])
        css_bg_color = f"rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})"

        # Determine text color for contrast
        # Using luminance formula (Y = 0.2126R + 0.7152G + 0.0722B)
        # Values are 0-1 for R,G,B in formula
        luminance = 0.2126 * rgba_color[0] + 0.7152 * rgba_color[1] + 0.0722 * rgba_color[2]
        css_text_color = "black" if luminance > 0.5 else "white" # Threshold can be adjusted

        escaped_token = html.escape(token_str)
        span_style = (
            f"background-color: {css_bg_color}; "
            f"color: {css_text_color}; "
            "padding: 0.15em 0.4em; " # Vertical and horizontal padding
            "margin: 0.1em; "       # Margin around each token
            "border-radius: 4px; "
            "display: inline-block; " # Ensures padding and margin work well with text flow
            "white-space: pre-wrap;" # Preserve spaces within a token if any
        )
        span_title = f"Perplexity: {raw_p:.3f}"
        token_spans_html.append(f'<span style="{span_style}" title="{span_title}">{escaped_token}</span>')

    # 4. Assemble the full HTML
    # Basic CSS styles
    # Using a slightly more specific class name to avoid conflicts
    style_block = f"""
<style>
    .token-perplexity-visualization-block {{
        padding: 15px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #fdfdfd;
        margin-bottom: 20px;
    }}
    .token-perplexity-visualization-block h2 {{
        font-family: Arial, sans-serif;
        color: #333;
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 1.4em;
    }}
    .token-perplexity-visualization-block h3.prompt-title {{
        font-family: Arial, sans-serif;
        color: #555;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.1em;
        font-weight: normal;
    }}
    .tokens-container {{
        font-family: Menlo, Monaco, Consolas, "Courier New", monospace;
        font-size: {font_size_px}px;
        line-height: {font_size_px * line_height_factor}px;
        text-align: left; /* Or 'justify' if preferred */
        word-wrap: break-word; /* Ensure long tokens without spaces can wrap */
    }}
    .color-bar-area {{
        margin-top: 15px;
        font-family: Arial, sans-serif;
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
</style>
"""

    title_html_str = ""
    if overall_title:
        title_html_str = f"<h2>{html.escape(overall_title)}</h2>"

    prompt_html_str = f"<h3 class='prompt-title'>{html.escape(prompt)}</h3>"
    tokens_div_html = f"<div class='tokens-container'>{' '.join(token_spans_html)}</div>" # Join with space for natural flow

    # 5. Color Bar (Simple HTML Version)
    color_bar_html_str = ""
    if show_color_bar:
        gradient_segments_html = []
        num_segments = 20 # More segments for a smoother look
        for i in range(num_segments):
            norm_val = i / (num_segments -1) # Normalized value for this segment
            rgba_color_cb = cmap(norm_val)
            css_bg_color_cb = f"rgb({int(rgba_color_cb[0]*255)}, {int(rgba_color_cb[1]*255)}, {int(rgba_color_cb[2]*255)})"
            gradient_segments_html.append(f'<div style="flex-grow: 1; background-color: {css_bg_color_cb};"></div>')

        color_bar_html_str = f"""
<div class='color-bar-area'>
    <div>Perplexity</div>
    <div class='color-bar-gradient'>
        {''.join(gradient_segments_html)}
    </div>
    <div class='color-bar-label'>
        <span>{min_p:.2f}</span>
        <span>{max_p:.2f}</span>
    </div>
</div>
"""
    full_html = f"""
{style_block}
<div class='token-perplexity-visualization-block'>
    {title_html_str}
    {prompt_html_str}
    {tokens_div_html}
    {color_bar_html_str if show_color_bar else ""}
</div>
"""
    return full_html

if __name__ == '__main__':
    # Example Usage (requires IPython for HTML display in some environments)
    try:
        from IPython.display import HTML, display
        def display_html(html_str):
            display(HTML(html_str))
    except ImportError:
        def display_html(html_str):
            # Fallback for non-IPython environments: save to a file
            file_path = "token_perplexity_visualization.html"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_str)
            print(f"HTML saved to {file_path}")
            import webbrowser
            webbrowser.open(file_path)


    example_tokens = "This is an example sentence to demonstrate the HTML token perplexity visualization function, especially with longer text that requires wrapping across multiple lines for clarity. The quick brown fox jumps over the lazy dog.".split()
    example_prompt = "Sample text with perplexity scores:"
    # Generate some plausible perplexities
    np.random.seed(42)
    example_perplexities = [np.random.uniform(1, 20) + (len(w) * 0.3) for w in example_tokens]
    example_perplexities[5] = 35.0 # example
    example_perplexities[10] = 28.0 # visualization

    html_output = visualize_token_perplexity_html(
        tokens=example_tokens,
        perplexities=example_perplexities,
        prompt=example_prompt,
        overall_title="Demo of HTML Perplexity",
        cmap_name="coolwarm", # Try different colormaps: "viridis_r", "magma_r", "coolwarm", "RdYlBu_r"
        font_size_px=18,
        line_height_factor=2.2
    )
    display_html(html_output)

    # Test with fewer tokens
    short_tokens = "A short example.".split()
    short_perplexities = [2.1, 15.8, 5.5, 1.2]
    html_output_short = visualize_token_perplexity_html(
        tokens=short_tokens,
        perplexities=short_perplexities,
        prompt="Short text:",
        overall_title="Short Text Demo",
        cmap_name="plasma"
    )
    display_html(html_output_short)

    # Test with no tokens
    html_output_empty = visualize_token_perplexity_html(
        tokens=[],
        perplexities=[],
        prompt="This should show nothing.",
        overall_title="Empty Tokens Demo"
    )
    display_html(html_output_empty)
    
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a nice plot from a file that contains a list of tokens and perplexity"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path of the folder that contain token generation txts",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="path of the folder in which to put the figures",
    )
    parser.add_argument(
        "--show",
        type=bool,
        default=False,
        help="Whether to display the pictures or not",
    )
    args = parser.parse_args()
    # Sample data

    import glob

    print(args.input)

    prompt = ""

    lines = []

    for path in glob.glob(args.input):
        print(path)

        tokens = []
        perplexities = []
        with open(path, "r", encoding="utf-8") as file:
            b = False

            prompt = file.readline().split(": ")[1]

            for line in file:
                if "sequence of low perplexity" in line or "Generated" in line:
                    break
                if len(line) > 4:
                    lines.append(line)
                if b and ":" in line:
                    print(line)
                    token, perp_prob, _whatever = line.split(": ")
                    perp = perp_prob.split('(')[0]
                    tokens.append(token)
                    perplexities.append(float(perp))

                if "perplexities:" in line:
                    b = True

        # Create visualization
        fig, ax = visualize_token_perplexity(
            tokens,
            perplexities,
            highlight_sequence=None # get_longest_low_perplexity(perplexities, 1),
        )

        # Create visualization
        # fig, ax = visualize_token_perplexity(tokens, (perplexities > 0.5).astype(int))

        # Save figure (optional)
        plt.savefig(
            f"{args.output}/{path.split('/')[-1][:-4]}.png",
            dpi=100,
            bbox_inches="tight",
        )

        plt.show()
