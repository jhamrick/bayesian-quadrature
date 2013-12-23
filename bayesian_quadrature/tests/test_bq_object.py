import numpy as np
import scipy.stats
import pytest

from .. import BQ
from .. import bq_c

import logging
logger = logging.getLogger("bayesian_quadrature.tests")
logger.setLevel("DEBUG")

DTYPE = np.dtype('float64')

gamma = 1
ntry = 10
n_candidate = 10
x_mean = 3.141592653589793
x_var = 10.0


def npseed():
    np.random.seed(87293)


def make_1d_gaussian(x=None, seed=True, n=30):
    if seed:
        npseed()
    if x is None:
        x = np.random.uniform(-8, 8, n)
    y = scipy.stats.norm.pdf(x, 0, 1)
    return x, y


def make_random_bq(seed=True, n=30):
    x, y = make_1d_gaussian(seed=seed, n=n)
    bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    return bq


def make_random_bq_and_fit(seed=True, n=30):
    bq = make_random_bq(seed=seed, n=n)
    try:
        bq.fit()
    except RuntimeError:
        bq.fit()
    return bq


def make_bq(n=30):
    x, y = make_1d_gaussian(np.linspace(-8, 8, n))
    bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    return bq


def make_bq_and_fit(n=30):
    bq = make_bq(n=n)
    bq._fit_l()
    bq.gp_l.params = (0.4, 1.1, 0)
    bq._fit_log_l()
    bq._fit_Dc()
    return bq


def make_xo():
    return np.linspace(-10, 10, 100)


def test_improve_covariance_conditioning():
    bq = make_bq_and_fit()

    K_l = bq.gp_l.Kxx
    bq_c.improve_covariance_conditioning(
        K_l, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_l == bq.gp_l.Kxx).all()
    assert K_l is bq.gp_l.Kxx

    K_tl = bq.gp_log_l.Kxx
    bq_c.improve_covariance_conditioning(
        K_tl, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_tl == bq.gp_log_l.Kxx).all()
    assert K_tl is bq.gp_log_l.Kxx

    K_del = bq.gp_Dc.Kxx
    bq_c.improve_covariance_conditioning(
        K_del, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_del == bq.gp_Dc.Kxx).all()
    assert K_del is bq.gp_Dc.Kxx


def test_init():
    x, y = make_1d_gaussian()
    bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    assert (x == bq.x).all()
    assert (y == bq.l).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], gamma, ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], gamma, ntry, n_candidate, x_mean, x_var, s=0)


def test_log_transform():
    x, y = make_1d_gaussian()
    log_y = np.log((y / float(gamma)) + 1)
    bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    assert np.allclose(bq.log_l, log_y)


def test_choose_candidates():
    bq = make_random_bq()
    bq._fit_l()
    xc = bq._choose_candidates()
    assert xc.ndim == 1
    assert xc.size >= bq.x.size

    diff = np.abs(xc[:, None] - bq.x[None])
    assert ((diff > 1e-4) | (diff == 0)).all()


@pytest.mark.xfail
def test_fit_l_same():
    params = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
        npseed()
        bq.fit()
        if params is None:
            params = bq.gp_l.params.copy()
        assert (params == bq.gp_l.params).all()


@pytest.mark.xfail
def test_fit_log_l_same():
    params = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
        npseed()
        bq.fit()
        if params is None:
            params = bq.gp_log_l.params.copy()
        assert (params == bq.gp_log_l.params).all()


@pytest.mark.xfail
def test_fit_Dc_same():
    params = None
    candidates = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
        npseed()
        bq.fit()
        if params is None:
            params = bq.gp_Dc.params.copy()
        if candidates is None:
            candidates = bq.xc.copy()
        assert (params == bq.gp_Dc.params).all()
        assert (candidates == bq.xc).all()


def test_l_mean():
    npseed()

    x, y = make_1d_gaussian(np.linspace(-8, 8, 30))
    xo, yo = make_1d_gaussian(make_xo())
    bq = BQ(x, y, gamma, ntry, n_candidate, x_mean, x_var, s=0)
    bq._fit_l()
    bq.gp_l.params = (0.4, 1.1, 0)
    bq._fit_log_l()
    bq._fit_Dc()
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
    bq = make_bq_and_fit()
    xo = make_xo()

    Kxxo = bq.gp_l.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    calc_int = np.empty(bq.x.shape[0])
    bq_c.int_K(
        calc_int, bq.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    Kxxo = bq.gp_log_l.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    bq_c.int_K(
        calc_int, bq.x[:, None],
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_int_K_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K(
            vals[i], bq.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_K1_K2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_l.Kxxo(xo)
    K2xxo = bq.gp_log_l.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(K1xxo[:, None] * K2xxo[None, :] * p_xo, xo)

    calc_int = np.empty((bq.x.shape[0], bq.x.shape[0]))
    bq_c.int_K1_K2(
        calc_int, bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_K1_K2_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.x.shape[0], bq.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K1_K2(
            vals[i], bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2_K1():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_l.Kxxo(xo)
    K2xoxo = bq.gp_log_l.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    int1 = np.trapz(K1xxo[:, None, :] * K2xoxo * p_xo, xo)
    approx_int = np.trapz(K1xxo[:, None] * int1[None] * p_xo, xo)

    calc_int = np.empty((bq.x.shape[0], bq.x.shape[0]))
    bq_c.int_int_K1_K2_K1(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-7)


def test_int_int_K1_K2_K1_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.x.shape[0], bq.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2_K1(
            vals[i], bq.gp_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xoxo = bq.gp_l.Kxoxo(xo)
    K2xxo = bq.gp_log_l.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    int1 = np.trapz(K1xoxo * K2xxo[:, :, None] * p_xo, xo)
    approx_int = np.trapz(int1 * p_xo, xo)

    calc_int = np.empty(bq.x.shape[0])
    bq_c.int_int_K1_K2(
        calc_int, bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_int_K1_K2_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2(
            vals[i], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K():
    bq = make_bq_and_fit()
    xo = make_xo()

    Kxoxo = bq.gp_l.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(np.trapz(Kxoxo * p_xo, xo) * p_xo, xo)
    calc_int = bq_c.int_int_K(
        1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)

    Kxoxo = bq.gp_log_l.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(np.trapz(Kxoxo * p_xo, xo) * p_xo, xo)
    calc_int = bq_c.int_int_K(
        1, bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)


def test_int_int_K_same():
    bq = make_bq_and_fit()

    vals = np.empty(100)
    for i in xrange(vals.shape[0]):
        vals[i] = bq_c.int_int_K(
            1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_K1_dK2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_l.Kxxo(xo)
    dK2xxo = bq.gp_log_l.K.dK_dw(bq.gp_log_l._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(
        K1xxo[None, :] * dK2xxo[:, None] * p_xo, xo)[..., None]

    calc_int = np.empty((bq.x.shape[0], bq.x.shape[0], 1))
    bq_c.int_K1_dK2(
        calc_int, bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_dK():
    bq = make_bq_and_fit()
    xo = make_xo()

    dKxxo = bq.gp_l.K.dK_dw(bq.gp_l._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(dKxxo * p_xo, xo)[..., None]
    calc_int = np.empty((bq.x.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    dKxxo = bq.gp_log_l.K.dK_dw(bq.gp_log_l._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    approx_int = np.trapz(dKxxo * p_xo, xo)[..., None]
    calc_int = np.empty((bq.x.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.x[:, None],
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_Z_mean():
    bq = make_bq_and_fit()
    xo = make_xo()

    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    l = bq.l_mean(xo)
    approx_Z = np.trapz(l * p_xo, xo)
    calc_Z = bq.Z_mean()

    assert np.allclose(approx_Z, calc_Z)


def test_Z_mean_same():
    bq = make_bq_and_fit(n=10)

    means = np.empty(100)
    for i in xrange(100):
        means[i] = bq.Z_mean()
    assert (means[0] == means).all()


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_Z_var_same():
    bq = make_bq_and_fit(n=10)

    vars = np.empty(100)
    eps = np.empty(100)
    for i in xrange(100):
        vars[i], eps[i] = bq._Z_var_and_eps()
    assert (vars[0] == vars).all()
    assert (eps[0] == eps).all()


def test_Z_var_close():
    bq = make_bq_and_fit(n=10)

    vars = np.empty(100)
    eps = np.empty(100)
    for i in xrange(100):
        vars[i], eps[i] = bq._Z_var_and_eps()
    assert np.allclose(vars[0], vars)
    assert np.allclose(eps[0], eps)


def test_Z_var():
    # int int m_l(x) m_l(x') (C_tl(x, x') + dm_dw*C_w*dm_dw) dx dx'
    bq = make_bq_and_fit()
    xo = make_xo()

    p_xo = scipy.stats.norm.pdf(xo, bq.x_mean[0], np.sqrt(bq.x_cov[0, 0]))
    m_l = bq.gp_l.mean(xo) + gamma
    C_tl = bq.gp_log_l.cov(xo)
    approx_var = np.trapz(np.trapz(C_tl * m_l * p_xo, xo) * m_l * p_xo, xo)

    dm_dw = bq.dm_dw(xo)
    nu = np.trapz(m_l * dm_dw * p_xo, xo)
    Cw = bq.Cw(bq.gp_log_l)
    approx_eps = nu ** 2 * Cw

    calc_var, calc_eps = bq._Z_var_and_eps()

    assert np.allclose(approx_var, calc_var, atol=1e-4)
    assert np.allclose(approx_eps, calc_eps)


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_expected_Z_var_same():
    bq = make_bq_and_fit()
    Z_var = bq.Z_var()
    for x in bq.x:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert E_Z_var == Z_var


def test_expected_Z_var_close():
    bq = make_bq_and_fit()
    Z_var = bq.Z_var()
    for x in bq.x:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert np.allclose(E_Z_var, Z_var)


def test_expected_squared_mean():
    bq = make_bq_and_fit()
    x_a = np.random.uniform(-10, 10)
    esm = bq.expected_squared_mean(np.array([x_a]))
    assert esm >= 0
