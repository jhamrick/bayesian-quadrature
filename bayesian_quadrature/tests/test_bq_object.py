import numpy as np
import pytest
import matplotlib.pyplot as plt

from .. import BQ
from . import util

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = util.DTYPE
options = util.options


def test_init():
    util.npseed()
    x, y = util.make_xy()
    bq = BQ(x, y, **options)
    assert (x == bq.x_s).all()
    assert (y == bq.l_s).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, **options)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], **options)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, **options)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], **options)


def test_choose_candidates():
    util.npseed()
    bq = util.make_bq()
    assert bq.x_c.ndim == 1
    assert bq.x_sc.size >= bq.x_s.size

    diff = np.abs(bq.x_sc[:, None] - bq.x_c[None])
    assert ((diff > 1e-1) | (diff == 0)).all()


@pytest.mark.xfail
def test_fit_l_same():
    params = None
    util.npseed()
    x, y = util.make_xy()
    bq = BQ(x, y, **options)

    for i in xrange(10):
        util.npseed()
        bq.fit()
        if params is None:
            params = bq.gp_l.params.copy()
        assert (params == bq.gp_l.params).all()


@pytest.mark.xfail
def test_fit_log_l_same():
    params = None
    util.npseed()
    x, y = util.make_xy()
    bq = BQ(x, y, **options)

    for i in xrange(10):
        util.npseed()
        bq.fit()
        if params is None:
            params = bq.gp_log_l.params.copy()
        assert (params == bq.gp_log_l.params).all()


def test_l_mean():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    yo = util.f_x(xo)
    l = bq.l_mean(xo)
    assert np.allclose(l, yo, atol=1e-3)


def test_Z_mean():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    approx_Z = bq.approx_Z_mean(xo)
    calc_Z = bq.Z_mean()

    assert np.allclose(approx_Z, calc_Z)


def test_Z_mean_same():
    util.npseed()
    bq = util.make_bq()

    means = np.empty(100)
    for i in xrange(100):
        means[i] = bq.Z_mean()
    assert (means[0] == means).all()


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_Z_var_same():
    util.npseed()
    bq = util.make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert (vars[0] == vars).all()


def test_Z_var_close():
    util.npseed()
    bq = util.make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert np.allclose(vars[0], vars)


def test_Z_var():
    # int int m_l(x) m_l(x') C_tl(x, x') dx dx'
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    approx_var = bq.approx_Z_var(xo)
    calc_var = bq.Z_var()
    assert np.allclose(approx_var, calc_var, atol=1e-4)


def test_expected_Z_var_close():
    util.npseed()
    bq = util.make_bq()
    Z_var = bq.Z_var()
    E_Z_var = bq.expected_Z_var(bq.x_s)
    assert np.allclose(E_Z_var, Z_var, atol=1e-4)


def test_expected_squared_mean():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 10)
    esm = bq.expected_squared_mean(x_a)
    assert (esm >= 0).all()


def test_plot_gp_log_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_log_l(ax)
    ax.cla()

    bq.plot_gp_log_l(ax, f_l=lambda x: np.log(util.f_x(x)))
    ax.cla()

    bq.plot_gp_log_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_gp_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_l(ax)
    ax.cla()

    bq.plot_gp_l(ax, f_l=util.f_x)
    ax.cla()

    bq.plot_gp_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_l(ax)
    ax.cla()

    bq.plot_l(ax, f_l=util.f_x)
    ax.cla()

    bq.plot_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot():
    util.npseed()
    bq = util.make_bq()

    bq.plot()
    plt.close('all')

    bq.plot(f_l=util.f_x)
    plt.close('all')

    bq.plot(xmin=-10, xmax=10)
    plt.close('all')


def test_plot_expected_variance():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_variance(ax)
    ax.cla()

    bq.plot_expected_variance(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_expected_squared_mean():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_squared_mean(ax)
    ax.cla()

    bq.plot_expected_squared_mean(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_l():
    util.npseed()
    bq = util.make_bq()
    assert (np.log(bq.l_s) == bq.tl_s).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()
    assert (bq.l_sc[bq.ns:] == np.exp(bq.gp_log_l.mean(bq.x_c))).all()


def test_expected_squared_mean_1():
    X = np.linspace(-5, 5, 20)[:, None]
    for x in X:
        bq = util.make_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2
        E_m2 = bq.expected_squared_mean(x)
        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2, atol=1e-4)
        assert np.allclose(m2, E_m2_close, atol=1e-4)
