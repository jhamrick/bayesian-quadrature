import numpy as np
import scipy.stats
import pytest

from .. import linalg_c as la
from . import util

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = util.DTYPE
options = util.options


def rand_mat(n):
    A = np.random.rand(n, n)
    A = np.array(A + A.T + np.eye(n) * n, order='F')
    return A


def test_cho_factor():
    for i in xrange(1, 11):
        A = rand_mat(i)
        A2 = A.copy()
        L1 = np.linalg.cholesky(A)
        L2 = np.empty_like(A, order='F')

        la.cho_factor(A, L2)
        L2 *= np.tri(i)
        assert np.allclose(L1, L2)
        assert (A == A2).all()

        la.cho_factor(A, A)
        A *= np.tri(i)
        assert np.allclose(L1, A)
        assert not (A == A2).all()
        

def test_cho_solve_vec():
    for i in xrange(1, 11):
        A = rand_mat(i)
        b = np.array(np.random.rand(i), order='F')
        b2 = b.copy()

        L = np.empty_like(A, order='F')
        x = np.empty(i, order='F')
        la.cho_factor(A, L)

        la.cho_solve_vec(L, b, x)
        assert np.allclose(np.dot(A, x), b)
        assert (b == b2).all()

        la.cho_solve_vec(L, b, b)
        assert np.allclose(np.dot(A, b), b2)
        assert not (b == b2).all()


def test_cho_solve_mat():
    for i in xrange(1, 11):
        A = rand_mat(i)
        B = np.array(np.random.rand(i, i), order='F')
        B2 = B.copy()

        L = np.empty_like(A, order='F')
        X = np.empty((i, i), order='F')
        la.cho_factor(A, L)

        la.cho_solve_mat(L, B, X)
        assert np.allclose(np.dot(A, X), B)
        assert (B == B2).all()

        la.cho_solve_mat(L, B, B)
        assert np.allclose(np.dot(A, B), B2)
        assert not (B == B2).all()


def test_logdet():
    for i in xrange(1, 11):
        A = rand_mat(i)
        ld1 = np.linalg.slogdet(A)[1]
        la.cho_factor(A, A)
        ld2 = la.logdet(A)
        assert np.allclose(ld1, ld2)


def test_dot11():
    for i in xrange(1, 11):
        x = np.array(np.random.rand(i), order='F')
        y = np.array(np.random.rand(i), order='F')

        d1 = np.dot(x, y)
        d2 = la.dot11(x, y)
        
        assert np.allclose(d1, d2)


def test_dot12():
    m = 10
    for i in xrange(1, 11):
        x = np.array(np.random.rand(i), order='F')
        y = np.array(np.random.rand(i, m), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty(m, order='F')
        la.dot12(x, y, d2)
        
        assert np.allclose(d1, d2)


def test_dot21():
    n = 10
    for i in xrange(1, 11):
        x = np.array(np.random.rand(n, i), order='F')
        y = np.array(np.random.rand(i), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty(n, order='F')
        la.dot21(x, y, d2)
        
        assert np.allclose(d1, d2)


def test_dot22():
    m = 10
    p = 15
    for i in xrange(1, 11):
        x = np.array(np.random.rand(m, i), order='F')
        y = np.array(np.random.rand(i, p), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty((m, p), order='F')
        la.dot22(x, y, d2)
        
        assert np.allclose(d1, d2)
