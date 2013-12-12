import numpy as np
import scipy.stats
import pytest

from mental_rotation import config, DTYPE
from mental_rotation.extra import BQ
import mental_rotation.extra.bq_c as bq_c
from . import util

import logging
logger = logging.getLogger("mental_rotation.extra.bq")
logger.setLevel("DEBUG")


gamma = config.getfloat("bq", "gamma")
ntry = config.getint("bq", "ntry")
n_candidate = config.getint("bq", "n_candidate")
R_mean = config.getfloat("model", "R_mu")
R_var = 1. / config.getfloat("model", "R_kappa")


def make_1d_gaussian(x=None, seed=True, n=30):
    if seed:
        util.seed()
    if x is None:
        x = np.random.uniform(-8, 8, n)
    y = scipy.stats.norm.pdf(x, 0, 1)
    return x, y


def make_random_bq(seed=True, n=30):
    x, y = make_1d_gaussian(seed=seed, n=n)
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
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
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    return bq


def make_bq_and_fit(n=30):
    bq = make_bq(n=n)
    bq._fit_S()
    bq.gp_S.params = (0.4, 1.1, 0)
    bq._fit_log_S()
    bq._fit_Dc()
    return bq


def make_xo():
    return np.linspace(-10, 10, 100)


def test_improve_covariance_conditioning():
    bq = make_bq_and_fit()

    K_l = bq.gp_S.Kxx
    bq_c.improve_covariance_conditioning(
        K_l, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_l == bq.gp_S.Kxx).all()
    assert K_l is bq.gp_S.Kxx

    K_tl = bq.gp_log_S.Kxx
    bq_c.improve_covariance_conditioning(
        K_tl, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_tl == bq.gp_log_S.Kxx).all()
    assert K_tl is bq.gp_log_S.Kxx

    K_del = bq.gp_Dc.Kxx
    bq_c.improve_covariance_conditioning(
        K_del, np.arange(K_l.shape[0], dtype=DTYPE))
    assert (K_del == bq.gp_Dc.Kxx).all()
    assert K_del is bq.gp_Dc.Kxx


def test_init():
    x, y = make_1d_gaussian()
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    assert (x == bq.R).all()
    assert (y == bq.S).all()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], gamma, ntry, n_candidate, R_mean, R_var, s=0)


def test_log_transform():
    x, y = make_1d_gaussian()
    log_y = np.log((y / float(gamma)) + 1)
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    assert np.allclose(bq.log_S, log_y)


def test_choose_candidates():
    bq = make_random_bq()
    bq._fit_S()
    Rc = bq._choose_candidates()
    assert Rc.ndim == 1
    assert Rc.size >= bq.R.size

    diff = np.abs(Rc[:, None] - bq.R[None])
    assert ((diff > 1e-4) | (diff == 0)).all()


@pytest.mark.xfail
def test_fit_S_same():
    params = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
        util.seed()
        bq.fit()
        if params is None:
            params = bq.gp_S.params.copy()
        assert (params == bq.gp_S.params).all()


@pytest.mark.xfail
def test_fit_log_S_same():
    params = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
        util.seed()
        bq.fit()
        if params is None:
            params = bq.gp_log_S.params.copy()
        assert (params == bq.gp_log_S.params).all()


@pytest.mark.xfail
def test_fit_Dc_same():
    params = None
    candidates = None
    for i in xrange(10):
        x, y = make_1d_gaussian(seed=True, n=10)
        bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
        util.seed()
        bq.fit()
        if params is None:
            params = bq.gp_Dc.params.copy()
        if candidates is None:
            candidates = bq.Rc.copy()
        assert (params == bq.gp_Dc.params).all()
        assert (candidates == bq.Rc).all()


def test_S_mean():
    util.seed()

    x, y = make_1d_gaussian(np.linspace(-8, 8, 30))
    xo, yo = make_1d_gaussian(make_xo())
    bq = BQ(x, y, gamma, ntry, n_candidate, R_mean, R_var, s=0)
    bq._fit_S()
    bq.gp_S.params = (0.4, 1.1, 0)
    bq._fit_log_S()
    bq._fit_Dc()
    S = bq.S_mean(xo)
    assert np.allclose(S, yo, atol=1e-4)


def test_mvn_logpdf():
    util.seed()
    x = np.random.uniform(-10, 10, 20)
    y = scipy.stats.norm.pdf(x, R_mean, np.sqrt(R_var))
    pdf = np.empty_like(y)
    mu = np.array([R_mean])
    cov = np.array([[R_var]])
    bq_c.mvn_logpdf(pdf, x[:, None], mu, cov)
    assert np.allclose(np.log(y), pdf)


def test_mvn_logpdf_same():
    util.seed()
    x = np.random.uniform(-10, 10, 20)
    mu = np.array([R_mean])
    cov = np.array([[R_var]])
    pdf = np.empty((100, x.size))
    for i in xrange(pdf.shape[0]):
        bq_c.mvn_logpdf(pdf[i], x[:, None], mu, cov)

    assert (pdf[0] == pdf).all()


def test_int_K():
    bq = make_bq_and_fit()
    xo = make_xo()

    Kxxo = bq.gp_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    calc_int = np.empty(bq.R.shape[0])
    bq_c.int_K(
        calc_int, bq.R[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    Kxxo = bq.gp_log_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    bq_c.int_K(
        calc_int, bq.R[:, None],
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_int_K_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.R.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K(
            vals[i], bq.R[:, None],
            bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
            bq.R_mean, bq.R_cov)

    assert (vals[0] == vals).all()


def test_int_K1_K2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_S.Kxxo(xo)
    K2xxo = bq.gp_log_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(K1xxo[:, None] * K2xxo[None, :] * p_xo, xo)

    calc_int = np.empty((bq.R.shape[0], bq.R.shape[0]))
    bq_c.int_K1_K2(
        calc_int, bq.gp_S.x[:, None], bq.gp_log_S.x[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_K1_K2_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.R.shape[0], bq.R.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K1_K2(
            vals[i], bq.gp_S.x[:, None], bq.gp_log_S.x[:, None],
            bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
            bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
            bq.R_mean, bq.R_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2_K1():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_S.Kxxo(xo)
    K2xoxo = bq.gp_log_S.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    int1 = np.trapz(K1xxo[:, None, :] * K2xoxo * p_xo, xo)
    approx_int = np.trapz(K1xxo[:, None] * int1[None] * p_xo, xo)

    calc_int = np.empty((bq.R.shape[0], bq.R.shape[0]))
    bq_c.int_int_K1_K2_K1(
        calc_int, bq.gp_S.x[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-7)


def test_int_int_K1_K2_K1_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.R.shape[0], bq.R.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2_K1(
            vals[i], bq.gp_S.x[:, None],
            bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
            bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
            bq.R_mean, bq.R_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xoxo = bq.gp_S.Kxoxo(xo)
    K2xxo = bq.gp_log_S.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    int1 = np.trapz(K1xoxo * K2xxo[:, :, None] * p_xo, xo)
    approx_int = np.trapz(int1 * p_xo, xo)

    calc_int = np.empty(bq.R.shape[0])
    bq_c.int_int_K1_K2(
        calc_int, bq.gp_log_S.x[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_int_K1_K2_same():
    bq = make_bq_and_fit()

    vals = np.empty((100, bq.R.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2(
            vals[i], bq.gp_log_S.x[:, None],
            bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
            bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
            bq.R_mean, bq.R_cov)

    assert (vals[0] == vals).all()


def test_int_int_K():
    bq = make_bq_and_fit()
    xo = make_xo()

    Kxoxo = bq.gp_S.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(np.trapz(Kxoxo * p_xo, xo) * p_xo, xo)
    calc_int = bq_c.int_int_K(
        1, bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)

    Kxoxo = bq.gp_log_S.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(np.trapz(Kxoxo * p_xo, xo) * p_xo, xo)
    calc_int = bq_c.int_int_K(
        1, bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)


def test_int_int_K_same():
    bq = make_bq_and_fit()

    vals = np.empty(100)
    for i in xrange(vals.shape[0]):
        vals[i] = bq_c.int_int_K(
            1, bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
            bq.R_mean, bq.R_cov)

    assert (vals[0] == vals).all()


def test_int_K1_dK2():
    bq = make_bq_and_fit()
    xo = make_xo()

    K1xxo = bq.gp_S.Kxxo(xo)
    dK2xxo = bq.gp_log_S.K.dK_dw(bq.gp_log_S._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(
        K1xxo[None, :] * dK2xxo[:, None] * p_xo, xo)[..., None]

    calc_int = np.empty((bq.R.shape[0], bq.R.shape[0], 1))
    bq_c.int_K1_dK2(
        calc_int, bq.gp_S.x[:, None], bq.gp_log_S.x[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_dK():
    bq = make_bq_and_fit()
    xo = make_xo()

    dKxxo = bq.gp_S.K.dK_dw(bq.gp_S._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(dKxxo * p_xo, xo)[..., None]
    calc_int = np.empty((bq.R.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.R[:, None],
        bq.gp_S.K.h, np.array([bq.gp_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)

    dKxxo = bq.gp_log_S.K.dK_dw(bq.gp_log_S._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    approx_int = np.trapz(dKxxo * p_xo, xo)[..., None]
    calc_int = np.empty((bq.R.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.R[:, None],
        bq.gp_log_S.K.h, np.array([bq.gp_log_S.K.w]),
        bq.R_mean, bq.R_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_Z_mean():
    bq = make_bq_and_fit()
    xo = make_xo()

    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    S = bq.S_mean(xo)
    approx_Z = np.trapz(S * p_xo, xo)
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

    p_xo = scipy.stats.norm.pdf(xo, bq.R_mean[0], np.sqrt(bq.R_cov[0, 0]))
    m_l = bq.gp_S.mean(xo) + gamma
    C_tl = bq.gp_log_S.cov(xo)
    approx_var = np.trapz(np.trapz(C_tl * m_l * p_xo, xo) * m_l * p_xo, xo)

    dm_dw = bq.dm_dw(xo)
    nu = np.trapz(m_l * dm_dw * p_xo, xo)
    Cw = bq.Cw(bq.gp_log_S)
    approx_eps = nu ** 2 * Cw

    calc_var, calc_eps = bq._Z_var_and_eps()

    assert np.allclose(approx_var, calc_var, atol=1e-4)
    assert np.allclose(approx_eps, calc_eps)


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_expected_Z_var_same():
    bq = make_bq_and_fit()
    Z_var = bq.Z_var()
    for x in bq.R:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert E_Z_var == Z_var


def test_expected_Z_var_close():
    bq = make_bq_and_fit()
    Z_var = bq.Z_var()
    for x in bq.R:
        E_Z_var = bq.expected_Z_var(np.array([x]))
        assert np.allclose(E_Z_var, Z_var)


def test_expected_squared_mean():
    bq = make_bq_and_fit()
    x_a = np.random.uniform(-10, 10)
    esm = bq.expected_squared_mean(np.array([x_a]))
    assert esm >= 0
