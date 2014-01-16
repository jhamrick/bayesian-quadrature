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
