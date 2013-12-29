import matplotlib.pyplot as plt


def set_scientific(low, high, axis=None, ax=None):
    """Set the axes or axis specified by `axis` to use scientific notation for
    ticklabels, if the value is <10**low or >10**high.

    Parameters
    ----------
    low : int
        Lower exponent bound for non-scientific notation
    high : int
        Upper exponent bound for non-scientific notation
    axis : str (default=None)
        Which axis to format ('x', 'y', or None for both)
    ax : axis object (default=pyplot.gca())
        The matplotlib axis object to use

    """
    # get the axis
    if ax is None:
        ax = plt.gca()
    # create the tick label formatter
    fmt = plt.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((low, high))
    # format the axis/axes
    if axis is None or axis == 'x':
        ax.get_yaxis().set_major_formatter(fmt)
    if axis is None or axis == 'y':
        ax.get_yaxis().set_major_formatter(fmt)
