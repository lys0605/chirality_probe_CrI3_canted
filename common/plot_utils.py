import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, ax=None, linestyle="-", linewidth=2, **kwargs):
    """
    Flexible plot function that returns figure and axes objects for further customization. Allows us to plot on an existing axes or create a new figure and axes.

    Parameters:
        x (array-like): Data for the x-axis.
        y (array-like): Data for the y-axis.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If None, a new figure and axes are created.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        linestyle (str): Style of the line (e.g., '-', '--', ':').
        linewidth (int): Width of the line.
        grid (bool): Whether to display a grid.
        **kwargs: Additional keyword arguments passed to `ax.plot()`.

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
    ax.plot(x, y, linestyle=linestyle, linewidth=linewidth, **kwargs)
    
    # Add grid if enabled
    # if grid:
    #     ax.grid(True)
    
    # Return the figure and axes for further customization
    return fig, ax

def letter_annotation(ax, xoffset, yoffset, letter,size=12):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes, size=size)

def panel(figsize=(8,6), nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0, wspace=0):
    """
    Creating a panel of subplots with equal aspect ratio and no space between subplots.

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
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=hspace, wspace=wspace, width_ratios=width_ratios, height_ratios=height_ratios)
    axes = gs.subplots()
    # for ax in axes:
    #     ax.set_box_aspect(1)
    #     ax.set_axis_off()
    return fig, axes

def panel_unequal(figsize=(8,6), nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0, wspace=0):
    """
    Creating a panel of subplots with unequal aspect ratio and no space between subplots.

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
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, hspace=hspace, wspace=wspace, width_ratios=width_ratios, height_ratios=height_ratios)
    # for ax in axes:
    #     ax.set_box_aspect(1)
    #     ax.set_axis_off()
    return fig, gs

def plot_lines_with_colorbar(fig, ax, x, y, values_for_color, color_bar_title='', color_bar_label='' ,cmap='viridis', linestyle='-', linewidth=2, alpha=0.9):
    """
    Plot multiple lines on the same axes with colors determined by a colormap and add a colorbar.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        x (array-like): Data for the x-axis.
        y (2D array-like): Each row corresponds to a different line to plot.
        values_for_color (array-like): Values used to determine the color of each line.
        cmap (str): Colormap name.
        linestyle (str): Style of the line (e.g., '-', '--', ':').
        linewidth (int): Width of the line.

    Returns:
        sm (matplotlib.cm.ScalarMappable): The ScalarMappable for the colorbar.
    """
    import matplotlib as mpl

    # 1. Normalize the values for color mapping
    norm = mpl.colors.Normalize(vmin=np.min(values_for_color), vmax=np.max(values_for_color))
    cmap = plt.get_cmap(cmap)

    # 2. Create ScalarMappable before the loop so colors are available immediately
    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])

    # 3. Plot each line with the corresponding color
    for i, val in enumerate(values_for_color):
        ax.plot(x, y[i],
                color=cmap(norm(val)),
                alpha=alpha,
                linestyle=linestyle,
                linewidth=linewidth)
    cbar = fig.colorbar(scalar_map, ax=ax)
    cbar.set_label(color_bar_label)
    cbar.ax.set_title(color_bar_title)

    return fig, ax


# ---------------------------------------------------------------------------
# RCD / pump-probe spectral plotting helpers
# ---------------------------------------------------------------------------

def plot_frequency_resolved_RCD(ax, w, chi, s_values, ls='-', **kwarg):
    """
    Plot frequency-resolved RCD spectra for one or more magnetic-field values.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    w            : array_like  frequency axis (ℏω/J)
    chi          : array_like  RCD spectrum; shape (plot_length, len(w)) or (len(w),)
    s_values     : array_like  magnetic-field values B/Bs for legend labels
    ls           : str         line style (default '-')
    **kwarg      : must contain:
                     plot_length (int)    number of curves to draw
                     color (str or list)  colour(s) for the curves
                     label (str)          base label string
    """
    if kwarg['plot_length'] != 1:
        for j in range(kwarg['plot_length']):
            ax.plot(w, chi[j], ls=ls, color=kwarg['color'][j],
                    label=kwarg['label'] + fr' $B={s_values[j]}B_s$')
    else:
        ax.plot(w, chi, ls=ls, color=kwarg['color'], label=kwarg['label'])


def plot_frequency_temperature_resolved_RCD(ax, w, chi, temperatures, ls='-', **kwarg):
    """
    Plot frequency-resolved RCD spectra for one or more temperatures.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    w            : array_like  frequency axis (ℏω/J)
    chi          : array_like  RCD spectrum; shape (plot_length, len(w)) or (len(w),)
    temperatures : array_like  temperature values in K for legend labels
    ls           : str         line style (default '-')
    **kwarg      : must contain:
                     plot_length (int)    number of curves to draw
                     color (str or list)  colour(s) for the curves
                     label (str)          base label string
    """
    if kwarg['plot_length'] != 1:
        for j in range(kwarg['plot_length']):
            ax.plot(w, chi[j], ls=ls, color=kwarg['color'][j],
                    label=kwarg['label'] + fr' $T={temperatures[j]}K$')
    else:
        ax.plot(w, chi, ls=ls, color=kwarg['color'], label=kwarg['label'])