import matplotlib.pyplot as plt
import numpy as np

from util_c import slice_sample as _slice_sample


def set_scientific(ax, low, high, axis=None):
    """Set the axes or axis specified by `axis` to use scientific notation for
    ticklabels, if the value is <10**low or >10**high.

    Parameters
    ----------
    ax : axis object
        The matplotlib axis object to use
    low : int
        Lower exponent bound for non-scientific notation
    high : int
        Upper exponent bound for non-scientific notation
    axis : str (default=None)
        Which axis to format ('x', 'y', or None for both)

    """
    # create the tick label formatter
    fmt = plt.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((low, high))

    # format the x axis
    if axis is None or axis == 'x':
        ax.get_yaxis().set_major_formatter(fmt)

    # format the y axis
    if axis is None or axis == 'y':
        ax.get_yaxis().set_major_formatter(fmt)


def slice_sample(logpdf, niter, xval=None, nburn=0, freq=1, window=10.0):
    """Draws samples from 'logpdf', optionally starting from 'xval'.  The
    pdf should return log values.

    Parameters
    ----------
    logpdf : function
        Target distribution. logpdf(xval) should return ln(Pr(xval))
    niter : int
        Number of iterations to run
    xval : float
        The initial starting value.  If not given, rejection sampling
        is used within [-window, window] until a value with
        non-zero probability is found.
    nburn : int (default 0)
        Number of samples to skip at the beginning
    freq : int (default 1)
        How often to record samples
    window : float (default 10.0)
        The interval to sample from.  Samples are drawn initially from
        [xval-window, xval+window], and this range is shortened when
        samples have zero probability.  The window is reset when
        starting each sample.

    """

    samples = np.empty((niter, 1))
    if xval is None:
        raise NotImplementedError
    else:
        samples[0] = xval

    _slice_sample(samples, logpdf, xval, window)

    # don't return burnin samples or inbetween samples
    out = samples[nburn:][::freq]
    return out
