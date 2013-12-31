import numpy as np
import scipy.stats
import pytest
import matplotlib.pyplot as plt

from .. import BQ
from .. import bq_c

DTYPE = np.dtype('float64')

options = {
    'ntry': 10,
    'n_candidate': 10,
    'x_mean': 0.0,
    'x_var': 10.0,
    'candidate_thresh': 0.5,
}


def npseed():
    np.random.seed(8728)


def make_x(n=9):
    x = np.linspace(-5, 5, n)
    return x


def f_x(x):
    y = scipy.stats.norm.pdf(x, 0, 1)
    return y


def make_xy(n=9):
    x = make_x(n=n)
    y = f_x(x)
    return x, y


def make_bq(n=9, x=None, nc=None):
    if x is None:
        x, y = make_xy(n=n)
    else:
        y = f_x(x)

    opt = options.copy()
    if nc is not None:
        opt['n_candidate'] = nc

    bq = BQ(x, y, **opt)
    bq._fit_log_l(params=(30, 5, 0))
    bq._fit_l(params=(y.max(), 1, 0))
    return bq


def make_xo():
    return np.linspace(-10, 10, 1000)
