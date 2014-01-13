import matplotlib.pyplot as plt
import numpy as np
import logging
import scipy.optimize as optim

from . import bq_c
from .util_c import slice_sample as _slice_sample

logger = logging.getLogger("bayesian_quadrature.util")
DTYPE = np.dtype('float64')
PREC = np.finfo(DTYPE).precision
MIN = np.log(np.exp2(np.float64(np.finfo(np.float64).minexp + 4)))


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


def slice_sample(logpdf, niter, w, xval, nburn=1, freq=1):
    """Draws samples from 'logpdf', optionally starting from 'xval'.  The
    pdf should return log values.

    Parameters
    ----------
    logpdf : function
        Target distribution. logpdf(xval) should return ln(Pr(xval))
    niter : int
        Number of iterations to run
    w : np.ndarray
        The step by which to adjust the window size.
    xval : numpy.ndarray
        The initial starting value.
    nburn : int (default 1)
        Number of samples to skip at the beginning
    freq : int (default 1)
        How often to record samples

    """

    samples = np.empty((niter, xval.size))
    samples[0] = xval

    # zero means unset, so we don't want to log in that case
    verbose = (logger.level != 0) and (logger.level < 10)
    _slice_sample(samples, logpdf, xval, w, verbose)

    # don't return burnin samples or inbetween samples
    out = samples[nburn:][::freq]
    return out


def vlines(ax, x, **kwargs):
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ymin, ymax)


def hlines(ax, y, **kwargs):
    xmin, xmax = ax.get_xlim()
    ax.hlines(y, xmin, xmax, **kwargs)
    ax.set_xlim(xmin, xmax)


def improve_conditioning(gp):
    Kxx = gp.Kxx
    cond = np.linalg.cond(Kxx)
    logger.debug("Kxx conditioning number is %s", cond)

    if hasattr(gp, "jitter"):
        jitter = gp.jitter
    else:
        jitter = np.zeros(Kxx.shape[0], dtype=DTYPE)
        gp.jitter = jitter

    # the conditioning is really bad -- just increase the variance
    # a little for all the elements until it's less bad
    idx = np.arange(Kxx.shape[0])
    while np.log10(cond) > (PREC / 2.0):
        logger.debug("Adding jitter to all elements")
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)
        cond = np.linalg.cond(Kxx)
        logger.debug("Kxx conditioning number is now %s", cond)

    # now improve just for those elements which result in a
    # negative variance, until there are no more negative elements
    # in the diagonal
    gp._memoized = {'Kxx': Kxx}
    var = np.diag(gp.cov(gp._x))
    while (var < 0).any():
        idx = np.nonzero(var < 0)[0]

        logger.debug("Adding jitter to indices %s", idx)
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)

        Kxx = gp.Kxx
        gp._memoized = {'Kxx': Kxx}
        var = np.diag(gp.cov(gp._x))
        cond = np.linalg.cond(Kxx)
        logger.debug("Kxx conditioning number is now %s", cond)


def improve_tail_covariance(gp):
    Kxx = gp.Kxx
    gp._memoized = {'Kxx': Kxx}
    max_jitter = np.diag(Kxx).max() * 1e-2
    new_jitter = np.clip(-gp.x * 1e-4, 0, max_jitter)
    Kxx += np.eye(gp.x.size) * new_jitter
    gp.jitter += new_jitter


def _anneal(*args, **kwargs):
    """Hack, because sometimes scipy's anneal function throws a TypeError
    for no particular reason. So just try again until it works.

    """
    while True:
        try:
            res = optim.minimize(*args, **kwargs)
        except TypeError:
            pass
        else:
            break
    return res

def find_good_parameters(logpdf, x0, ntry=10):
    logger.debug("Trying to find good parameters...")

    for i in xrange(ntry):
        
        logger.debug("Attempt #%d with Powell", i+1)
        res = optim.minimize(
            fun=lambda x: -logpdf(x),
            x0=x0,
            method='Powell')
        
        logger.debug(res)
        p = logpdf(res['x'])
        if p > MIN:
            return res['x']
        if logpdf(x0) < p:
            x0 = res['x']

    return None
