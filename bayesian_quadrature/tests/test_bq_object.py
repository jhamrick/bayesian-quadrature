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
    with pytest.raises(ValueError):
        BQ(x, -y, **options)


def test_choose_candidates():
    util.npseed()
    bq = util.make_bq()
    assert bq.x_c.ndim == 1
    assert bq.x_sc.size >= bq.x_s.size

    diff = np.abs(bq.x_sc[:, None] - bq.x_c[None])
    assert ((diff > 1e-1) | (diff == 0)).all()


def test_l_mean():
    util.npseed()
    bq = util.make_bq()
    l = bq.l_mean(bq.x_s)
    assert np.allclose(l, bq.l_s, atol=1e-4)


def test_Z_mean():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    approx_Z = bq._approx_Z_mean(xo)
    calc_Z = bq._exact_Z_mean()

    assert np.allclose(approx_Z, calc_Z, atol=1e-5)


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
    approx_var = bq._approx_Z_var(xo)
    calc_var = bq._exact_Z_var()
    assert np.allclose(approx_var, calc_var, atol=1e-4)


def test_expected_Z_var_close():
    util.npseed()
    bq = util.make_bq()
    Z_var = bq.Z_var()
    E_Z_var = bq.expected_Z_var(bq.x_s)
    assert np.allclose(E_Z_var, Z_var, atol=1e-4)


def test_expected_squared_mean_valid():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 10)
    esm = bq.expected_squared_mean(x_a)
    assert (esm >= 0).all()


def test_expected_squared_mean():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 20)[:, None]
    x = bq._make_approx_x()
    for xa in x_a:
        esm = bq._exact_expected_squared_mean(xa)
        approx = bq._approx_expected_squared_mean(xa, x)
        assert np.allclose(esm, approx, atol=1e-4)


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
    util.npseed()
    X = np.linspace(-5, 5, 20)[:, None]
    for x in X:
        bq = util.make_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2

        E_m2 = bq.expected_squared_mean(x)
        assert np.allclose(m2, E_m2, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2_close, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-8)
        assert np.allclose(m2, E_m2_close, atol=1e-4)


def test_periodic():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    y = util.f_xp(x)
    assert np.allclose(bq.l_mean(x), y, atol=1e-3)


def test_periodic_z_mean():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    l = bq.l_mean(x)
    p_x = bq._make_approx_px(x)
    approx_z = np.trapz(l * p_x, x)
    assert np.allclose(bq.Z_mean(), approx_z)
    

def test_periodic_z_var():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    l = bq.l_mean(x)
    C = bq.gp_log_l.cov(x)
    p_x = bq._make_approx_px(x)
    approx_z = np.trapz(np.trapz(C * l * p_x, x) * l * p_x, x)
    assert np.allclose(bq.Z_var(), approx_z)


@pytest.mark.xfail(reason="poorly conditioned matrix")
def test_periodic_expected_squared_mean():
    util.npseed()
    bq = util.make_periodic_bq(nc=0)
    x_a = np.random.uniform(-np.pi, np.pi, 20)[:, None]
    x = np.linspace(-np.pi, np.pi, 1000)

    for xa in x_a:
        esm = bq.expected_squared_mean(xa)
        approx = bq._approx_expected_squared_mean(xa, x)
        assert np.allclose(esm, approx)


def test_periodic_expected_squared_mean_1():
    util.npseed()
    X = np.linspace(-np.pi, np.pi, 20)[:, None]
    for x in X:
        bq = util.make_periodic_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2
        E_m2 = bq.expected_squared_mean(x)
        assert np.allclose(m2, E_m2, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2_close, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-8)
        assert np.allclose(m2, E_m2_close, atol=1e-4)


def test_add_observation():
    util.npseed()
    bq = util.make_bq()
    x = bq.x_s.copy()
    l = bq.l_s.copy()
    tl = bq.tl_s.copy()

    x_a = np.random.uniform(-5, 5, 1)
    l_a = util.f_x(x_a)
    tl_a = np.log(l_a)

    bq.add_observation(x_a, l_a)
    assert (bq.x_s == np.append(x, x_a)).all()
    assert (bq.l_s == np.append(l, l_a)).all()
    assert (bq.tl_s == np.append(tl, tl_a)).all()

    assert (bq.x_s == bq.x_sc[:bq.ns]).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()


def test_approx_add_observation():
    util.npseed()
    bq = util.make_periodic_bq()
    x = bq.x_s.copy()
    l = bq.l_s.copy()
    tl = bq.tl_s.copy()

    x_a = np.random.randint(-np.pi, np.pi, 1)
    l_a = util.f_x(x_a)
    tl_a = np.log(l_a)

    bq.add_observation(x_a, l_a)
    assert (bq.x_s == np.append(x, x_a)).all()
    assert (bq.l_s == np.append(l, l_a)).all()
    assert (bq.tl_s == np.append(tl, tl_a)).all()

    assert (bq.x_s == bq.x_sc[:bq.ns]).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()


def test_choose_next():
    util.npseed()
    bq = util.make_bq()
    bq.choose_next(n=1)


def test_choose_next_with_cost():
    util.npseed()
    bq = util.make_bq()
    f = lambda x: x ** 2
    bq.choose_next(cost_fun=f, n=1)
