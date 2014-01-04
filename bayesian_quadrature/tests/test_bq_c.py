import numpy as np
import scipy.stats
import pytest

from .. import bq_c
from . import util

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = util.DTYPE
options = util.options


def test_improve_covariance_conditioning():
    util.npseed()
    bq = util.make_bq()

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


def test_mvn_logpdf():
    x_mean = options['x_mean']
    x_var = options['x_var']

    util.npseed()
    x = np.random.uniform(-10, 10, 20)
    y = scipy.stats.norm.pdf(x, x_mean, np.sqrt(x_var))
    pdf = np.empty_like(y)
    mu = np.array([x_mean])
    cov = np.array([[x_var]])
    bq_c.mvn_logpdf(pdf, x[:, None], mu, cov)
    assert np.allclose(np.log(y), pdf)


def test_mvn_logpdf_same():
    x_mean = options['x_mean']
    x_var = options['x_var']

    util.npseed()
    x = np.random.uniform(-10, 10, 20)
    mu = np.array([x_mean])
    cov = np.array([[x_var]])
    pdf = np.empty((100, x.size))
    for i in xrange(pdf.shape[0]):
        bq_c.mvn_logpdf(pdf[i], x[:, None], mu, cov)

    assert (pdf[0] == pdf).all()


def test_int_K():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    approx_int = bq_c.approx_int_K(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty(bq.gp_l.x.shape[0])
    bq_c.int_K(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_K_same():
    util.npseed()
    bq = util.make_bq()

    vals = np.empty((100, bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K(
            vals[i], bq.gp_log_l.x[:, None],
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_K1_K2():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

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
    util.npseed()
    bq = util.make_bq()

    vals = np.empty((100, bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_K1_K2(
            vals[i], bq.gp_l.x[:, None], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2_K1():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

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
    util.npseed()
    bq = util.make_bq()

    vals = np.empty((100, bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2_K1(
            vals[i], bq.gp_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K1_K2():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

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
    util.npseed()
    bq = util.make_bq()

    vals = np.empty((100, bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        bq_c.int_int_K1_K2(
            vals[i], bq.gp_log_l.x[:, None],
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    approx_int = bq_c.approx_int_int_K(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = bq_c.int_int_K(
        1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_int_K_same():
    util.npseed()
    bq = util.make_bq()

    vals = np.empty(100)
    for i in xrange(vals.shape[0]):
        vals[i] = bq_c.int_int_K(
            1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.x_mean, bq.x_cov)

    assert (vals[0] == vals).all()


@pytest.mark.xfail(reason="implementation has a bug, see visual tests")
def test_int_K1_dK2():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

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
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    approx_int = bq_c.approx_int_dK(xo, bq.gp_l, bq.x_mean, bq.x_cov)
    calc_int = np.empty((bq.gp_l.x.shape[0], 1))
    bq_c.int_dK(
        calc_int, bq.gp_l.x[:, None],
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.x_mean, bq.x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-4)


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
