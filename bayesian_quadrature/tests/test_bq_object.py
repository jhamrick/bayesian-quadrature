import numpy as np
import scipy.stats
import pytest
import matplotlib.pyplot as plt

from .. import BQ
from .. import bq_c

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = np.dtype('float64')

ntry = 10
n_candidate = 10
x_mean = 0.0
x_var = 10.0


def npseed():
    np.random.seed(8728)


def make_x(n=10):
    x = np.linspace(-5, 5, n)
    return x


def f_x(x):
    y = scipy.stats.norm.pdf(x, 0, 1)
    return y


def make_xy(n=10):
    x = make_x(n=n)
    y = f_x(x)
    return x, y


def make_bq(n=10, x=None, nc=None):
    if x is None:
        x, y = make_xy(n=n)
    else:
        y = f_x(x)

    if nc is None:
        nc = n_candidate

    bq = BQ(x, y, ntry, nc, x_mean, x_var, s=0, h=30, w=1)
    bq._fit_log_l(params=(10, 5, 0))
    bq._fit_l(params=(y.max(), 1, 0))
    return bq


def make_xo():
    return np.linspace(-10, 10, 1000)


def test_improve_covariance_conditioning():
    npseed()
    bq = make_bq()

    K = bq.gp_l.Kxx
    K_old = K.copy()
    jitter = np.zeros(K.shape[0], dtype=DTYPE)
    bq_c.improve_covariance_conditioning(
        K, jitter, np.arange(K.shape[0]))
    assert (K == bq.gp_l.Kxx).all()
    assert K is bq.gp_l.Kxx
    assert (K_old == (K - (np.eye(K.shape[0]) * jitter))).all()

    K = bq.gp_log_l.Kxx
    K_old = K.copy()
    jitter = np.zeros(K.shape[0], dtype=DTYPE)
    bq_c.improve_covariance_conditioning(
        K, jitter, np.arange(K.shape[0]))
    assert (K == bq.gp_log_l.Kxx).all()
    assert K is bq.gp_log_l.Kxx
    assert (K_old == (K - (np.eye(K.shape[0]) * jitter))).all()


def test_init():
    npseed()
    x, y = make_xy()
    bq = BQ(x, y, ntry, n_candidate, x_mean, x_var, s=0)
    assert (x == bq.x_s).all()
    assert (y == bq.l_s).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], ntry, n_candidate, x_mean, x_var, s=0)


def test_choose_candidates():
    npseed()
    bq = make_bq()
    assert bq.x_c.ndim == 1
    assert bq.x_sc.size >= bq.x_s.size

    diff = np.abs(bq.x_sc[:, None] - bq.x_c[None])
    assert ((diff > 1e-1) | (diff == 0)).all()


@pytest.mark.xfail
def test_fit_l_same():
    params = None
    npseed()
    x, y = make_xy()
    bq = BQ(x, y, ntry, n_candidate, x_mean, x_var, s=0)

    for i in xrange(10):
        npseed()
        bq.fit()
        if params is None:
            params = bq.gp_l.params.copy()
        assert (params == bq.gp_l.params).all()


@pytest.mark.xfail
def test_fit_log_l_same():
    params = None
    npseed()
    x, y = make_xy()
    bq = BQ(x, y, ntry, n_candidate, x_mean, x_var, s=0)

    for i in xrange(10):
        npseed()
        bq.fit()
        if params is None:
            params = bq.gp_log_l.params.copy()
        assert (params == bq.gp_log_l.params).all()


def test_l_mean():
    npseed()
    bq = make_bq()
    xo = make_xo()
    yo = f_x(xo)
    l = bq.l_mean(xo)
    assert np.allclose(l, yo, atol=1e-3)


def test_mvn_logpdf():
    npseed()
    x = np.random.uniform(-10, 10, 20)
    y = scipy.stats.norm.pdf(x, x_mean, np.sqrt(x_var))
    pdf = np.empty_like(y)
    mu = np.array([x_mean])
    cov = np.array([[x_var]])
    bq_c.mvn_logpdf(pdf, x[:, None], mu, cov)
    assert np.allclose(np.log(y), pdf)


def test_mvn_logpdf_same():
    npseed()
    x = np.random.uniform(-10, 10, 20)
    mu = np.array([x_mean])
    cov = np.array([[x_var]])
    pdf = np.empty((100, x.size))
    for i in xrange(pdf.shape[0]):
        bq_c.mvn_logpdf(pdf[i], x[:, None], mu, cov)

    assert (pdf[0] == pdf).all()


def test_int_K():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_K(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty(bq.gp_l.x.shape[0])
    bq_c.int_K(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_K_same():
    npseed()
    bq = make_bq()

    vals = np.empty((100, bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K(
            vals[i], bq.gp_log_l.x[:, None],
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_K1_K2():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_K1_K2(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0]))
    bq_c.int_K1_K2(
        calc_int, bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_int_K1_K2_same():
    npseed()
    bq = make_bq()

    vals = np.empty((100, bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K1_K2(
            vals[i], bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2_K1():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_int_K1_K2_K1(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]))
    bq_c.int_int_K1_K2_K1(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_int_K1_K2_K1_same():
    npseed()
    bq = make_bq()

    vals = np.empty((100, bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2_K1(
            vals[i], bq.gp_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_int_K1_K2(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_cov)

    calc_int = np.empty(bq.gp_log_l.x.shape[0])
    bq_c.int_int_K1_K2(
        calc_int, bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_int_K1_K2_same():
    npseed()
    bq = make_bq()

    vals = np.empty((100, bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2(
            vals[i], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_int_K(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = bq_c.int_int_K(
        1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_int_K_same():
    npseed()
    bq = make_bq()

    vals = np.empty(100)
    for i in xrange(vals.shape[0]):
        vals[i] = bq_c.int_int_K(
            1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


@pytest.mark.xfail(reason="implementation has a bug, see visual tests")
def test_int_K1_dK2():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_K1_dK2(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0], 1))
    bq_c.int_K1_dK2(
        calc_int, bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int)


def test_int_dK():
    npseed()
    bq = make_bq()
    xo = make_xo()

    approx_int = bq_c.approx_int_dK(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty((bq.gp_l.x.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_Z_mean():
    npseed()
    bq = make_bq()
    xo = make_xo()
    approx_Z = bq.approx_Z_mean(xo)
    calc_Z = bq.Z_mean()

    assert np.allclose(approx_Z, calc_Z)


def test_Z_mean_same():
    npseed()
    bq = make_bq()

    means = np.empty(100)
    for i in xrange(100):
        means[i] = bq.Z_mean()
    assert (means[0] == means).all()


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_Z_var_same():
    npseed()
    bq = make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert (vars[0] == vars).all()


def test_Z_var_close():
    npseed()
    bq = make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert np.allclose(vars[0], vars)


def test_Z_var():
    # int int m_l(x) m_l(x') C_tl(x, x') dx dx'
    npseed()
    bq = make_bq()
    xo = make_xo()
    approx_var = bq.approx_Z_var(xo)
    calc_var = bq.Z_var()
    assert np.allclose(approx_var, calc_var, atol=1e-4)


@pytest.mark.xfail(reason="bug")
def test_expected_Z_var_close():
    npseed()
    bq = make_bq()
    Z_var = bq.Z_var()
    E_Z_var = bq.expected_Z_var(bq.x_s)
    assert np.allclose(E_Z_var, Z_var)


@pytest.mark.xfail(reason="bug")
def test_expected_squared_mean():
    npseed()
    bq = make_bq()
    x_a = np.random.uniform(-10, 10, 10)
    esm = bq.expected_squared_mean(x_a)
    assert (esm >= 0).all()


def test_plot_gp_log_l():
    npseed()
    bq = make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_log_l(ax)
    ax.cla()

    bq.plot_gp_log_l(ax, f_l=lambda x: np.log(f_x(x)))
    ax.cla()

    bq.plot_gp_log_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_gp_l():
    npseed()
    bq = make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_l(ax)
    ax.cla()

    bq.plot_gp_l(ax, f_l=f_x)
    ax.cla()

    bq.plot_gp_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_l():
    npseed()
    bq = make_bq()
    fig, ax = plt.subplots()

    bq.plot_l(ax)
    ax.cla()

    bq.plot_l(ax, f_l=f_x)
    ax.cla()

    bq.plot_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot():
    npseed()
    bq = make_bq()

    bq.plot()
    plt.close('all')

    bq.plot(f_l=f_x)
    plt.close('all')

    bq.plot(xmin=-10, xmax=10)
    plt.close('all')


def test_plot_expected_variance():
    npseed()
    bq = make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_variance(ax)
    ax.cla()

    bq.plot_expected_variance(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_expected_squared_mean():
    npseed()
    bq = make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_squared_mean(ax)
    ax.cla()

    bq.plot_expected_squared_mean(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_l():
    npseed()
    bq = make_bq()
    assert (np.log(bq.l_s) == bq.tl_s).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()
    assert (bq.l_sc[bq.ns:] == np.exp(bq.gp_log_l.mean(bq.x_c))).all()


def test_expected_squared_mean_1():
    X = np.linspace(-5, 5, 20)[:, None]
    for x in X:
        bq = make_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2
        E_m2 = bq.expected_squared_mean(x)
        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2, atol=1e-4)
        assert np.allclose(m2, E_m2_close, atol=1e-4)


def test_remove_jitter():
    n = 2

    arr = np.ones((n, n))
    jitter = np.zeros(n)
    idx = np.arange(n)
    bq_c.improve_covariance_conditioning(arr, jitter, idx)
    bq_c.remove_jitter(arr, jitter, idx)
    assert (arr == np.ones((n, n))).all()
    assert (jitter == np.zeros(n)).all()

    bq_c.improve_covariance_conditioning(arr, jitter, idx)
    j = jitter[-1]
    aj = arr[-1, -1]
    bq_c.remove_jitter(arr, jitter, idx[:-1])
    assert (arr[:-1, :-1] == np.ones((n - 1, n - 1))).all()
    assert (arr[:-1, -1] == np.ones(n - 1)).all()
    assert (arr[-1, :-1] == np.ones(n - 1)).all()
    assert arr[-1, -1] == aj
    assert (jitter[:-1] == np.zeros(n - 1)).all()
    assert jitter[-1] == j


def test_int_exp_norm():
    def approx_int_exp_norm(xo, c, m, S):
        e = np.exp(xo * c)
        p = scipy.stats.norm.pdf(xo, m, np.sqrt(S))
        return np.trapz(e * p, xo)

    xo = np.linspace(-20, 20, 1000)

    approx = approx_int_exp_norm(xo, 2, 0, 1)
    calc = bq_c.int_exp_norm(2, 0, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 1, 0, 1)
    calc = bq_c.int_exp_norm(1, 0, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 2, 1, 1)
    calc = bq_c.int_exp_norm(2, 1, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 2, 1, 2)
    calc = bq_c.int_exp_norm(2, 1, 2)
    assert np.allclose(approx, calc)
