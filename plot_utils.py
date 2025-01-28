import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, ax=None, title="Custom Plot", xlabel="X-axis", ylabel="Y-axis", color="blue", linestyle="-", linewidth=2, grid=True):
    """
    Flexible plot function that returns figure and axes objects for further customization. Allows us to plot on an existing axes or create a new figure and axes.

    Parameters:
        x (array-like): Data for the x-axis.
        y (array-like): Data for the y-axis.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, a new figure and axes are created.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the line.
        linestyle (str): Style of the line (e.g., '-', '--', ':').
        linewidth (int): Width of the line.
        grid (bool): Whether to display a grid.

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object.
    """
    # Create a new figure and axes if none are provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Plot the data
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add grid if enabled
    if grid:
        ax.grid(True)
    
    # Return the figure and axes for further customization
    return fig, ax

def letter_annotation(ax, xoffset, yoffset, letter,size=12):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=size)

def panel(figsize=(8,6), nrowcols=2, nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0, wspace=0):
    """
    Creating a panel of subplots with equal aspect ratio and no space between subplots. Plotting images in a 2x2 panel.

    Parameters:
        figsize (tuple): Figure size.
        nrows (int): Number of rows in the panel.
        ncols (int): Number of columns in the panel.
        width_ratios (list): Ratios of the width of each column.
        height_ratios (list): Ratios of the height of each row.
        hspace (float): Space between rows.
        wspace (float): Space between columns.
    
    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        axes (np.ndarray): Array of axes objects.
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=hspace, wspace=wspace, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = gs.subplots()
    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_axis_off()
    return fig, axes