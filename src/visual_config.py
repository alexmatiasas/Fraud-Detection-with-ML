import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    # General plot settings
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams.update({
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),
    })