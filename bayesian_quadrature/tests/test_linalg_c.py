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


def test_cho_factor():
    pass


def test_cho_solve_vec():
    pass


def test_cho_solve_mat():
    pass


def test_logdet():
    pass


def test_dot11():
    n = 10
    for i in xrange(10):
        x = np.array(np.random.rand(n), order='F')
        y = np.array(np.random.rand(n), order='F')

        d1 = np.dot(x, y)
        d2 = la.dot11(x, y)
        
        assert np.allclose(d1, d2)


def test_dot12():
    n = 10
    m = 15
    for i in xrange(10):
        x = np.array(np.random.rand(n), order='F')
        y = np.array(np.random.rand(n, m), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty(m, order='F')
        la.dot12(x, y, d2)
        
        assert np.allclose(d1, d2)


def test_dot21():
    n = 10
    m = 15
    for i in xrange(10):
        x = np.array(np.random.rand(n, m), order='F')
        y = np.array(np.random.rand(m), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty(n, order='F')
        la.dot21(x, y, d2)
        
        assert np.allclose(d1, d2)


def test_dot22():
    n = 10
    m = 15
    for i in xrange(10):
        x = np.array(np.random.rand(n, m), order='F')
        y = np.array(np.random.rand(m, n), order='F')

        d1 = np.dot(x, y)
        d2 = np.empty((n, n), order='F')
        la.dot22(x, y, d2)
        
        assert np.allclose(d1, d2)
