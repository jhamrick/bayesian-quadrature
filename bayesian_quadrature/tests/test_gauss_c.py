import numpy as np
import scipy.stats
import pytest

from .. import gauss_c
from .. import linalg_c as la
from . import util

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = util.DTYPE
options = util.options


def test_mvn_logpdf():
    util.npseed()

    x_mean = options['x_mean']
    x_var = options['x_var']

    mu = np.array([x_mean], order='F')
    cov = np.array([[x_var]], order='F')
    la.cho_factor(cov, cov)
    logdet = la.logdet(cov)

    n = 20
    x = np.array(np.random.uniform(-10, 10, n)[None], order='F')
    y = np.log(np.array(
        scipy.stats.norm.pdf(x, x_mean, np.sqrt(x_var)), order='F'))

    pdf = np.empty(n, order='F')
    for i in xrange(n):
        pdf[i] = gauss_c.mvn_logpdf(x[:, i], mu, cov, logdet)

    assert np.allclose(y, pdf)


def test_mvn_logpdf_same():
    util.npseed()

    mu = np.array([options['x_mean']], order='F')
    cov = np.array([[options['x_var']]], order='F')
    la.cho_factor(cov, cov)
    logdet = la.logdet(cov)

    n = 20
    m = 20
    x = np.array(np.random.uniform(-10, 10, n)[None], order='F')
    pdf = np.empty((m, n), order='F')
    for i in xrange(m):
        for j in xrange(n):
            pdf[i, j] = gauss_c.mvn_logpdf(x[:, j], mu, cov, logdet)

    assert (pdf[0] == pdf).all()


def test_int_exp_norm():
    def approx_int_exp_norm(xo, c, m, S):
        e = np.exp(xo * c)
        p = scipy.stats.norm.pdf(xo, m, np.sqrt(S))
        return np.trapz(e * p, xo)

    xo = np.linspace(-20, 20, 1000)

    approx = approx_int_exp_norm(xo, 2, 0, 1)
    calc = gauss_c.int_exp_norm(2, 0, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 1, 0, 1)
    calc = gauss_c.int_exp_norm(1, 0, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 2, 1, 1)
    calc = gauss_c.int_exp_norm(2, 1, 1)
    assert np.allclose(approx, calc)

    approx = approx_int_exp_norm(xo, 2, 1, 2)
    calc = gauss_c.int_exp_norm(2, 1, 2)
    assert np.allclose(approx, calc)


def test_int_K():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    Kxxo = np.array(bq.gp_l.Kxxo(xo), order='F')
    approx_int = np.empty(bq.gp_l.x.shape[0], order='F')
    gauss_c.approx_int_K(
        approx_int, np.array(xo[None], order='F'), 
        Kxxo, x_mean, x_cov)

    calc_int = np.empty(bq.gp_l.x.shape[0], order='F')
    gauss_c.int_K(
        calc_int, np.array(bq.gp_l.x[None], order='F'),
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        x_mean, x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_K_same():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    vals = np.empty((bq.gp_l.x.shape[0], 20), order='F')
    for i in xrange(20):
        gauss_c.int_K(
            vals[:, i], np.array(bq.gp_l.x[None], order='F'),
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            x_mean, x_cov)

    assert (vals[:, [0]] == vals).all()


def test_approx_int_K_same():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    Kxxo = np.array(bq.gp_l.Kxxo(xo), order='F')

    vals = np.empty((bq.gp_l.x.shape[0], 20), order='F')
    xo = np.array(xo[None], order='F')
    for i in xrange(20):
        gauss_c.approx_int_K(
            vals[:, i], xo, 
            np.array(Kxxo, order='F'), 
            x_mean, x_cov)

    assert (vals[:, [0]] == vals).all()


def test_int_K1_K2():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    K1xxo = np.array(bq.gp_l.Kxxo(xo), order='F')
    K2xxo = np.array(bq.gp_log_l.Kxxo(xo), order='F')
    approx_int = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0]), order='F')
    gauss_c.approx_int_K1_K2(
        approx_int, np.array(xo[None], order='F'), 
        K1xxo, K2xxo, x_mean, x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0]), order='F')
    gauss_c.int_K1_K2(
        calc_int, 
        np.array(bq.gp_l.x[None], order='F'),
        np.array(bq.gp_log_l.x[None], order='F'),
        bq.gp_l.K.h, np.array([bq.gp_l.K.w], order='F'),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w], order='F'),
        x_mean, x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-3)


def test_int_K1_K2_same():
    util.npseed()
    bq = util.make_bq()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    vals = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0], 20), order='F')
    for i in xrange(vals.shape[-1]):
        gauss_c.int_K1_K2(
            vals[:, :, i], 
            np.array(bq.gp_l.x[None], order='F'),
            np.array(bq.gp_log_l.x[None], order='F'),
            bq.gp_l.K.h, np.array([bq.gp_l.K.w], order='F'),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w], order='F'),
            x_mean, x_cov)

    assert (vals[:, :, [0]] == vals).all()


def test_approx_int_K1_K2_same():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    K1xxo = np.array(bq.gp_l.Kxxo(xo), order='F')
    K2xxo = np.array(bq.gp_log_l.Kxxo(xo), order='F')
    vals = np.empty((bq.gp_l.x.shape[0], bq.gp_log_l.x.shape[0], 20), order='F')
    for i in xrange(vals.shape[-1]):
        gauss_c.approx_int_K1_K2(
            vals[:, :, i], np.array(xo[None], order='F'), 
            K1xxo, K2xxo, x_mean, x_cov)

    assert (vals[:, :, [0]] == vals).all()


def test_int_int_K1_K2_K1():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    K1xxo = np.array(bq.gp_l.Kxxo(xo), order='F')
    K2xoxo = np.array(bq.gp_log_l.Kxoxo(xo), order='F')
    approx_int = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]), order='F')
    gauss_c.approx_int_int_K1_K2_K1(
        approx_int, np.array(xo[None], order='F'), 
        K1xxo, K2xoxo, x_mean, x_cov)

    calc_int = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0]), order='F')
    gauss_c.int_int_K1_K2_K1(
        calc_int, np.array(bq.gp_l.x[None], order='F'),
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        x_mean, x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_int_K1_K2_K1_same():
    util.npseed()
    bq = util.make_bq()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    vals = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0], 20), order='F')
    for i in xrange(vals.shape[-1]):
        gauss_c.int_int_K1_K2_K1(
            vals[:, :, i], np.array(bq.gp_l.x[None], order='F'),
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            x_mean, x_cov)

    assert (vals[:, :, [0]] == vals).all()


def test_approx_int_int_K1_K2_K1_same():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    K1xxo = np.array(bq.gp_l.Kxxo(xo), order='F')
    K2xoxo = np.array(bq.gp_log_l.Kxoxo(xo), order='F')

    vals = np.empty((bq.gp_l.x.shape[0], bq.gp_l.x.shape[0], 20), order='F')
    for i in xrange(vals.shape[-1]):
        gauss_c.approx_int_int_K1_K2_K1(
            vals[:, :, i], np.array(xo[None], order='F'), 
            K1xxo, K2xoxo, x_mean, x_cov)

    assert (vals[:, :, [0]] == vals).all()


def test_int_int_K1_K2():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    approx_int = gauss_c.approx_int_int_K1_K2(
        xo, bq.gp_l, bq.gp_log_l, x_mean, x_cov)

    calc_int = np.empty(bq.gp_log_l.x.shape[0])
    gauss_c.int_int_K1_K2(
        calc_int, np.array(bq.gp_log_l.x[:, None]),
        bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
        x_mean, x_cov)

    assert np.allclose(calc_int, approx_int, atol=1e-5)


def test_int_int_K1_K2_same():
    util.npseed()
    bq = util.make_bq()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    vals = np.empty((20, bq.gp_log_l.x.shape[0]))
    for i in xrange(vals.shape[0]):
        gauss_c.int_int_K1_K2(
            vals[i], np.array(bq.gp_log_l.x[:, None]),
            bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            bq.gp_log_l.K.h, np.array([bq.gp_log_l.K.w]),
            x_mean, x_cov)

    assert (vals[0] == vals).all()


def test_int_int_K():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    approx_int = gauss_c.approx_int_int_K(xo, bq.gp_l, x_mean, x_cov)
    calc_int = gauss_c.int_int_K(
        1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
        x_mean, x_cov)
    assert np.allclose(calc_int, approx_int, atol=1e-6)


def test_int_int_K_same():
    util.npseed()
    bq = util.make_bq()

    x_mean = bq.options['x_mean']
    x_cov = bq.options['x_cov']

    vals = np.empty(20)
    for i in xrange(vals.shape[0]):
        vals[i] = gauss_c.int_int_K(
            1, bq.gp_l.K.h, np.array([bq.gp_l.K.w]),
            x_mean, x_cov)

    assert (vals[0] == vals).all()
