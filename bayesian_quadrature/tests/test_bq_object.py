import numpy as np
import scipy.stats
import pytest

from .. import BQ
from .. import bq_c

import logging
logger = logging.getLogger("bayesian_quadrature.tests")
logger.setLevel("DEBUG")

DTYPE = np.dtype('float64')

ntry = 10
n_candidate = 3
x_mean = 3.141592653589793
x_var = 10.0


def npseed():
    np.random.seed(8728)


def make_x(n=3):
    x = np.random.uniform(-3, 3, n)
    return x


def f_x(x):
    y = scipy.stats.norm.pdf(x, 0, 1)
    return y


def make_xy(n=3):
    x = make_x(n=n)
    y = f_x(x)
    return x, y


def make_bq(n=3):
    x, y = make_xy(n=n)
    bq = BQ(x, y, ntry, n_candidate, x_mean, x_var, s=0, h=30, w=1)
    bq._fit_log_l()
    bq.gp_log_l.params = (50, 5, 0)
    bq._fit_l()
    bq.gp_l.params = (y.max(), 1, 0)
    return bq


def make_xo():
    return np.linspace(-10, 10, 1000)


def test_improve_covariance_conditioning():
    npseed()
    bq = make_bq()

    K_l = bq.gp_l.Kxx
    bq_c.improve_covariance_conditioning(
        K_l, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_l == bq.gp_l.Kxx).all()
    assert K_l is bq.gp_l.Kxx

    K_tl = bq.gp_log_l.Kxx
    bq_c.improve_covariance_conditioning(
        K_tl, np.arange(K_tl.shape[0], dtype=DTYPE))
    assert (K_tl == bq.gp_log_l.Kxx).all()
    assert K_tl is bq.gp_log_l.Kxx


def test_init():
    npseed()
    x, y = make_xy()
    bq = BQ(x, y, ntry, n_candidate, x_mean, x_var, s=0)
    assert (x == bq.x).all()
    assert (y == bq.l).all()

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
    assert bq.x_c.size >= bq.x.size

    diff = np.abs(bq.x_c[:, None] - bq.x[None])
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
    assert np.allclose(l, yo, atol=1e-4)


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
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    approx_int = bq_c.approx_int_K(xo, bq.gp_log_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty(bq.gp_log_l.x.shape[0])
    bq_c.int_K(
        calc_int, bq.gp_log_l.x[:, None],
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


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

    assert np.allclose(calc_int, approx_int, atol=1e-6)


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

    approx_int = bq_c(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]))
    bq_c.int_int_K1_K2_K1(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-7)


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

    approx_int = bq.approx_int_int_K1_K2(
        xo, bq.gp_l, bq.gp_log_l, bq.x_mean, bq.x_co)

    calc_int = np.empty(bq.gp_log_l.x.shape[0])
    bq_c.int_int_K1_K2(
        calc_int, bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-6)


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
    assert np.allclose(calc_int, approx_int, atol=1e-4)

    approx_int = bq_c.approx_int_int_K(xo, bq.gp_log_l, bq.x_mean, bq.x_cov)
    calc_int = bq_c.int_int_K(
        1, bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)


def test_int_int_K_same():
    npseed()
    bq = make_bq()

    vals = np.empty(100)
    for i in xrange(vals.shape[0]):
        vals[i] = bq_c.int_int_K(
            1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


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

    assert np.allclose(calc_int, approx_int, atol=1e-5)


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
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    approx_int = bq_c.approx_int_dK(xo, bq.gp_log_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty((bq.gp_log_l.x.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.gp_log_l.x[:, None],
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_Z_mean():
    npseed()
    bq = make_bq()
    xo = make_xo()

    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    l = bq.l_mean(xo)
    approx_Z = np.trapz(l * p_xo, xo)
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

    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    m_l = bq.gp_l.mean(xo)
    C_tl = bq.gp_log_l.cov(xo)
    approx_var = np.trapz(np.trapz(C_tl * m_l * p_xo, xo) * m_l * p_xo, xo)

    calc_var = bq.Z_var()

    assert np.allclose(approx_var, calc_var, atol=1e-4)


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_expected_Z_var_same():
    npseed()
    bq = make_bq()
    Z_var = bq.Z_var()
    for x in bq.x:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert E_Z_var == Z_var


def test_expected_Z_var_close():
    npseed()
    bq = make_bq()
    Z_var = bq.Z_var()
    for x in bq.x:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert np.allclose(E_Z_var, Z_var)


def test_expected_squared_mean():
    npseed()
    bq = make_bq()
    x_a = np.random.uniform(-10, 10)
    esm = bq.expected_squared_mean(np.array([x_a]))
    assert esm >= 0
