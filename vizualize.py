import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from matplotlib.collections import PatchCollection

# Set global plot style
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

def visualize_pidf(name):
    """
    Loads and plots the 'PIDF' bar plot for a given dataset (name).
    """
    # Load the base data
    with open(f'interpretability_{name}.pickle', 'rb') as h:
        data = pickle.load(h)
    with open(f'interpretability_std_{name}.pickle', 'rb') as h:
        std_data = pickle.load(h)
    with open(f'syns_and_reds_{name}.pickle', 'rb') as h:
        captions_data = pickle.load(h)
    
    # Convert all values to absolute for proper scaling
    for k in range(len(data)):
        for i in range(len(data[k])):
            if isinstance(data[k][i], list):
                data[k][i] = [abs(x) for x in data[k][i]]
            else:
                data[k][i] = abs(data[k][i])
    
    def generate_captions(caps):
        return [str(num) for num in caps]
    
    synergistic_captions = ['\n'.join(generate_captions(cd[0])) for cd in captions_data]
    redundant_captions = [generate_captions(cd[1]) for cd in captions_data]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    
    setup_plot(ax, data, std_data, synergistic_captions, redundant_captions, name)
    ax.set_title(f'PIDF for dataset: {name}')
    
    plt.tight_layout()
    plt.savefig(f'PIDF_{name}.svg', bbox_inches='tight')
    plt.show()

def setup_plot(ax, data, std_data, synergistic_captions, redundant_captions, name):
    """
    Creates the stacked bar chart for PIDF decomposition (MI, FWS, Redundancies, etc.)
    """
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, color='white', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ind = np.arange(len(data))
    width = 0.35
    
    ax.bar(
        ind - width / 2,
        data[:, 0],
        width,
        yerr=std_data[:, 0] / math.sqrt(5),
        label='MI',
        color='#ff4b4b',
        alpha=0.9,
        zorder=3,
        capsize=5
    )
    
    ax.bar(
        ind - width / 2,
        data[:, 1],
        width,
        yerr=std_data[:, 1] / math.sqrt(5),
        bottom=data[:, 0],
        label='FWS',
        color='#6fB46F',
        alpha=0.9,
        zorder=3,
        capsize=5
    )
    
    bottom_stack = np.zeros(len(data))
    colors = ['#ddb2f2', '#b95fe4', '#8d21c1']
    
    for i, red_values in enumerate(data[:, 2]):
        std_vals = std_data[:, 2][i]
        for j, val in enumerate(red_values):
            color = colors[j % len(colors)]
            bar = ax.bar(
                ind[i] + width / 2,
                val,
                width,
                yerr=std_vals[j] / math.sqrt(5),
                bottom=bottom_stack[i],
                color=color,
                alpha=0.9,
                zorder=3,
                capsize=3
            )
            bottom_stack[i] += val
            
            if val >= 0.1 and j < len(redundant_captions[i]):
                ax.text(
                    bar[0].get_x() + bar[0].get_width() / 2.,
                    bottom_stack[i] - val / 2,
                    redundant_captions[i][j],
                    ha='center',
                    va='center',
                    rotation=0,
                    zorder=4,
                    size=10
                )
    
    synergy_bars = ax.bar(
        ind - width / 2,
        data[:, 1],
        width,
        bottom=data[:, 0],
        alpha=0  # invisible, just for iteration
    )
    
    for idx, rect in enumerate(synergy_bars):
        height = rect.get_height()
        if height >= 0.1:
            ax.text(
                rect.get_x() + rect.get_width() / 2.,
                rect.get_y() + height / 2,
                synergistic_captions[idx],
                ha='center',
                va='center',
                zorder=4,
                size=10
            )
    
    ax.set_xticks(ind)
    ax.set_xticklabels([f'F$_{{{i+1}}}$' for i in range(len(data))], rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('PIDF')
    

