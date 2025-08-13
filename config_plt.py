import matplotlib.pyplot as plt
from cycler import cycler
import os.path

# Set a new default color palette
colors = ["#0072b2", "#e69f00", "#009e72", "#d55c00", "#f0e442", "#56b4e9", "#cc79a7"]
# Adapted from https://doi.org/10.1038/nmeth.1618

def config_plt():
    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=colors),
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "legend.title_fontsize": 13,
        "axes.edgecolor": "#222222",
    })
config_plt()

# Function for creating new axes
def get_fig_ax(figsize=(3.6, 2.5)):
    left_adjust = 0.17
    bottom_adjust = 0.22
    right_adjust = 0.96
    top_adjust = 0.97
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    plt.subplots_adjust(left_adjust, bottom_adjust, right_adjust, top_adjust)
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.1, zorder=0)
    # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.1, zorder=1)
    # ax.minorticks_on()
    return fig, ax

# Function for saving figures
def save_my_fig(fig, filename):
    joined_filename = os.path.join("./figures/", filename)
    print(joined_filename)
    fig.savefig(f"{joined_filename}.pdf")
    fig.savefig(f"{joined_filename}.png")
    fig.savefig(f"{joined_filename}.eps")