import numpy as np
import scipy.stats
import pytest
import matplotlib.pyplot as plt
from gp import PeriodicKernel

from .. import BQ
from .. import bq_c

DTYPE = np.dtype('float64')

options = {
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
    bq.fit_log_l((30, 5, 0))
    bq.fit_l((y.max(), 1, 0))
    return bq


def make_xo():
    return np.linspace(-10, 10, 1000)


def vmpdf(x, mu, kappa):
    C = -np.log(2 * np.pi * scipy.special.iv(0, kappa))
    p = np.exp(C + (kappa * np.cos(x - mu)))
    return p


def f_xp(x):
    return vmpdf(x, mu=0.1, kappa=1.1)


def make_periodic_bq(x=None, nc=None):
    opt = options.copy()
    opt['kernel'] = PeriodicKernel
    if nc is not None:
        opt['n_candidate'] = nc

    if x is None:
        x = np.linspace(-np.pi, np.pi, 9)
    y = f_xp(x)
    
    bq = BQ(x, y, **opt)
    bq.fit_log_l((3.5, np.pi, 1, 0))
    bq.fit_l((0.4, np.pi / 2., 1, 0))

    return bq
