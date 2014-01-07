# cython: profile=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
cimport linalg_c as la
cimport gauss_c as ga

import scipy.stats
import scipy.linalg

from numpy.linalg import LinAlgError
from libc.math cimport exp, log, fmax, copysign, fabs, M_PI
from cpython cimport bool
from warnings import warn

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t integer

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps
cdef DTYPE_t NAN = np.nan


def improve_covariance_conditioning(np.ndarray[DTYPE_t, mode='c', ndim=2] M, np.ndarray[DTYPE_t, mode='c', ndim=1] jitters, np.ndarray[long, mode='c', ndim=1] idx):
    cdef DTYPE_t sqd_jitter = fmax(EPS, np.max(M)) * 1e-4
    cdef int i

    for i in xrange(len(idx)):
        jitters[idx[i]] += sqd_jitter
        M[idx[i], idx[i]] += sqd_jitter


def remove_jitter(np.ndarray[DTYPE_t, mode='c', ndim=2] M, np.ndarray[DTYPE_t, mode='c', ndim=1] jitters, np.ndarray[long, mode='c', ndim=1] idx):
    cdef int i
    for i in xrange(len(idx)):
        M[idx[i], idx[i]] -= jitters[idx[i]]
        jitters[idx[i]] = 0



def Z_mean(np.ndarray[DTYPE_t, mode='c', ndim=2] x_sc, np.ndarray[DTYPE_t, mode='c', ndim=1] alpha_l, DTYPE_t h_l, np.ndarray[DTYPE_t, mode='c', ndim=1] w_l, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):

    cdef int nc = x_sc.shape[0]
    cdef int d = x_sc.shape[1]

    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fx_sc = np.empty((nc, d), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fcov = np.empty((d, d), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] int_K_l = np.empty(nc, dtype=DTYPE, order='F')
    cdef DTYPE_t m_Z

    for i in xrange(nc):
        for j in xrange(d):
            fx_sc[i, j] = x_sc[i, j]

    for i in xrange(d):
        for j in xrange(d):
            fcov[i, j] = cov[i, j]

    # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
    ga.int_K(int_K_l, fx_sc, h_l, w_l, mu, fcov)
    m_Z = la.dot11(int_K_l, alpha_l)
    if m_Z <= 0:
        warn("m_Z = %s" % m_Z)

    return m_Z


def Z_var(np.ndarray[DTYPE_t, mode='c', ndim=2] x_s, np.ndarray[DTYPE_t, mode='c', ndim=2] x_sc, np.ndarray[DTYPE_t, mode='c', ndim=1] alpha_l, np.ndarray[DTYPE_t, mode='fortran', ndim=2] L_tl, DTYPE_t h_l, np.ndarray[DTYPE_t, mode='c', ndim=1] w_l, DTYPE_t h_tl, np.ndarray[DTYPE_t, mode='c', ndim=1] w_tl, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):

    cdef int ns = x_s.shape[0]
    cdef int nc = x_sc.shape[0]
    cdef int d = x_sc.shape[1]

    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fx_s = np.empty((ns, d), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fx_sc = np.empty((nc, d), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fcov = np.empty((d, d), dtype=DTYPE, order='F')

    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] int_K_l_K_tl_K_l = np.empty((nc, nc), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] int_K_tl_K_l_mat = np.empty((ns, nc), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] beta = np.empty(ns, dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] L_tl_beta = np.empty(ns, dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] alpha_int = np.empty(nc, dtype=DTYPE, order='F')

    cdef DTYPE_t beta2, alpha_int_alpha, V_Z
    cdef int i, j

    for i in xrange(ns):
        for j in xrange(d):
            fx_s[i, j] = x_s[i, j]

    for i in xrange(nc):
        for j in xrange(d):
            fx_sc[i, j] = x_sc[i, j]

    for i in xrange(d):
        for j in xrange(d):
            fcov[i, j] = cov[i, j]

    # E[m_l C_tl m_l | x_sc] = alpha_l(x_sc)' *
    #    int int K_l(x_sc, x) K_tl(x, x') K_l(x', x_sc) p(x) p(x') dx dx' *
    #    alpha_l(x_sc) - beta(x_sc)'beta(x_sc)
    # Where beta is defined as:
    # beta(x_sc) = inv(L_tl(x_s, x_s)) *
    #    int K_tl(x_s, x) K_l(x, x_sc) p(x) dx *
    #    alpha_l(x_sc)
    ga.int_int_K1_K2_K1(int_K_l_K_tl_K_l, fx_sc, h_l, w_l, h_tl, w_tl, mu, fcov)
    la.dot12(alpha_l, int_K_l_K_tl_K_l, alpha_int)
    alpha_int_alpha = la.dot11(alpha_int, alpha_l)

    ga.int_K1_K2(int_K_tl_K_l_mat, fx_s, fx_sc, h_tl, w_tl, h_l, w_l, mu, fcov)
    la.dot21(int_K_tl_K_l_mat, alpha_l, beta)
    la.cho_solve_vec(L_tl, beta, L_tl_beta)
    beta2 = la.dot11(beta, L_tl_beta)

    V_Z = alpha_int_alpha - beta2
    if V_Z <= 0:
        warn("V_Z = %s" % V_Z)

    return V_Z


def expected_squared_mean(np.ndarray[DTYPE_t, mode='fortran', ndim=1] int_K_l, np.ndarray[DTYPE_t, mode='c', ndim=1] l_sc, np.ndarray[DTYPE_t, mode='c', ndim=2] K_l, DTYPE_t tm_a, DTYPE_t tC_a):
    cdef int n = K_l.shape[0]

    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] fK_l = np.empty((n, n), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] L = np.empty((n, n), dtype=DTYPE, order='F')
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] A_sca = np.empty(n, dtype=DTYPE, order='F')

    cdef DTYPE_t A_a, A_sc_l, e1, e2, E_m2
    cdef int i, j

    for i in xrange(n):
        for j in xrange(n):
            fK_l[i, j] = K_l[i, j]

    la.cho_factor(fK_l, L)
    la.cho_solve_vec(L, int_K_l, A_sca)

    A_a = A_sca[-1]
    A_sc_l = la.dot11(A_sca[:-1], l_sc)

    e1 = ga.int_exp_norm(1, tm_a, tC_a)
    e2 = ga.int_exp_norm(2, tm_a, tC_a)

    E_m2 = (A_sc_l**2) + (2*A_sc_l*A_a * e1) + (A_a**2 * e2)

    return E_m2


def filter_candidates(np.ndarray[DTYPE_t, mode='c', ndim=1] x_c, np.ndarray[DTYPE_t, mode='c', ndim=1] x_s, DTYPE_t thresh):
    cdef int nc = x_c.shape[0]
    cdef int ns = x_s.shape[0]
    cdef int i, j
    cdef bool done = False
    cdef DTYPE_t diff

    while not done:
        done = True
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                if np.isnan(x_c[i]) or np.isnan(x_c[j]):
                    continue

                diff = fabs(x_c[i] - x_c[j])
                if diff < thresh:
                    x_c[i] = (x_c[i] + x_c[j]) / 2.0
                    x_c[j] = NAN
                    done = False

        for i in xrange(nc):
            for j in xrange(ns):
                diff = fabs(x_c[i] - x_s[j])
                if diff < thresh:
                    x_c[i] = NAN

