import argparse

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from utils import get_longest_low_perplexity


def visualize_token_perplexity(
    tokens,
    prompt,
    perplexities,
    cmap="jet",
    figsize=(8, 8),
    fontsize=14,
    title="Token Perplexity Visualization",
    max_line_width=1500,
    line_spacing=2.2,
    highlight_sequence=None,  # if not None, displays some specific tokens higlighted, on the form (start_idx, end_idx)
):
    """
    Visualize word tokens with color highlighting based on their perplexity values.
    Automatically wraps text to multiple lines for better visibility.

    Parameters:
    -----------
    tokens : list
        List of word tokens as strings
    perplexities : list
        List of perplexity values corresponding to each token
    cmap : str or matplotlib colormap, default='viridis'
        Colormap to use for highlighting
    figsize : tuple, default=(12, 8)
        Figure size
    fontsize : int, default=14
        Font size for the tokens
    title : str, default="Token Perplexity Visualization"
        Title of the plot
    max_line_width : int, default=80
        Maximum number of characters per line before wrapping
    line_spacing : float, default=2.2
        Vertical spacing between lines as a multiple of token height

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Validate inputs
    if len(tokens) != len(perplexities):
        raise ValueError("Number of tokens and perplexities must match")

    # Set up colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    fig_width, fig_height = figsize

    # Normalize perplexity values to range [0, 1]
    min_perp = min(perplexities)
    max_perp = max(perplexities)
    norm_perplexities = [
        (p - min_perp) / (max_perp - min_perp) if max_perp > min_perp else 0.5
        for p in perplexities
    ]

    # Calculate token widths and determine line breaks
    token_widths = {token: max(50, len(token) * fig_width) for token in tokens}
    token_spacing = 10  # Spacing between two tokens
    print(token_widths, max_line_width)
    # Determine line breaks
    lines = []
    current_line = []
    current_line_width = 0
    current_line_indices = []

    for i, (token, width) in enumerate(zip(tokens, token_widths.values())):
        # If adding this token would exceed max line width, start a new line
        if current_line_width + width + token_spacing > width * 30:
            lines.append((current_line, current_line_indices))
            current_line = [token]
            current_line_indices = [i]
            current_line_width = width + token_spacing
        else:
            current_line.append(token)
            current_line_indices.append(i)
            current_line_width += width + token_spacing

    # Add the last line if it's not empty
    if current_line:
        lines.append((current_line, current_line_indices))

    # Calculate figure height based on number of lines
    token_height = fontsize * 2
    total_text_height = len(lines) * (token_height * line_spacing)

    # Create figure with proper spacing
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create a layout with space for colorbar at bottom
    gs = gridspec.GridSpec(2, 1, height_ratios=[max(len(lines) * 2, 3), 1])
    ax = plt.subplot(gs[0])
    cbar_ax = plt.subplot(gs[1])

    # Turn off axis
    ax.axis("off")

    # Find the index to higlight
    highlight_from = len(tokens)
    highlight_to = highlight_from + 1
    if highlight_sequence is not None:
        highlight_from, highlight_to = highlight_sequence

    # Set y-limits to accommodate all lines
    y_min = -fontsize
    y_max = total_text_height  # + fontsize * 2
    ax.set_ylim(y_min, y_max + fontsize * 4)

    # Draw tokens line by line
    for line_idx, (line_tokens, line_indices) in enumerate(lines):
        x_position = fontsize * 0.5
        y_position = y_max - (line_idx * (token_height * 1 * line_spacing))

        for token, i in zip(line_tokens, line_indices):
            norm_perp = norm_perplexities[i]
            perp = perplexities[i]

            # Get color from colormap
            color = cmap(norm_perp)

            # Create a box for the token
            token_width = token_widths[token]
            rect = patches.Rectangle(
                (x_position, y_position - fontsize),
                token_width,
                token_height * 1.8,
                facecolor=color,
                edgecolor="none",
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Determine text color based on background brightness
            rgb = np.array(color[:3])
            brightness = np.sqrt(
                0.299 * rgb[0] ** 2 + 0.587 * rgb[1] ** 2 + 0.114 * rgb[2] ** 2
            )
            text_color = "black" if brightness > 0.6 else "white"

            if highlight_from < i <= highlight_to:
                text_color = (0.6, 0.0, 0.0) if brightness > 0.6 else "pink"

            # Add the token text
            ax.text(
                x_position + token_width / 2,
                y_position + 1.5 * token_height / 2,
                token,
                ha="center",
                va="center",
                color=text_color,
                family="monospace",
                fontsize=fontsize,
                fontweight="bold",
            )

            # Add perplexity value below
            ax.text(
                x_position + token_width / 2,
                y_position - fontsize * 0.1,
                f"{perp:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize * 0.7,
                fontweight="bold",
            )

            # Update x position for next token
            x_position += token_width + token_spacing

        # Set x-limit for current line
        if line_idx == 0:
            ax.set_xlim(0, x_position + fontsize)

    ax.set_aspect(0.7)
    # Set title
    ax.set_title(prompt, fontsize=fontsize * 1.2, pad=20)

    # Add a horizontal colorbar at the bottom
    norm = plt.Normalize(min_perp, max_perp)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"log(Perplexity)", fontsize=fontsize)
    cbar_ax.tick_params(labelsize=fontsize * 0.8)
    plt.tight_layout()
    return fig, ax


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
                if "Longest sequence of low perplexity tokens" in line:
                    break
                if len(line) > 4:
                    lines.append(line)
                if b and ":" in line:
                    token, perp = line.split(": ")
                    perp = float(perp)
                    tokens.append(token)
                    perplexities.append(perp)

                if "perplexities:" in line:
                    b = True

        perplexities = np.log(perplexities)
        # Create visualization
        fig, ax = visualize_token_perplexity(
            tokens,
            prompt,
            perplexities,
            figsize=(16, 8),
            highlight_sequence=get_longest_low_perplexity(perplexities, 1),
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
