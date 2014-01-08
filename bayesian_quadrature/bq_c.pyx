# cython: profile=True

import numpy as np
import scipy.stats
import scipy.linalg
from numpy.linalg import LinAlgError
from warnings import warn
from numpy import float64, int32
from numpy import empty

######################################################################

from libc.math cimport exp, log, fmax, fabs
from numpy cimport float64_t

cimport cython
cimport linalg_c as la
cimport gauss_c as ga

######################################################################

cdef float64_t MIN = log(np.exp2(float64(np.finfo(float64).minexp + 4)))
cdef float64_t EPS = np.finfo(float64).eps
cdef float64_t NAN = np.nan

######################################################################

def improve_covariance_conditioning(float64_t[:, ::1] M, float64_t[::1] jitters, long[::1] idx):
    cdef float64_t sqd_jitter = fmax(EPS, np.max(M)) * 1e-4
    cdef int i
    for i in xrange(len(idx)):
        jitters[idx[i]] += sqd_jitter
        M[idx[i], idx[i]] += sqd_jitter


def remove_jitter(float64_t[:, ::1] M, float64_t[::1] jitters, long[::1] idx):
    cdef int i
    for i in xrange(len(idx)):
        M[idx[i], idx[i]] -= jitters[idx[i]]
        jitters[idx[i]] = 0


def Z_mean(float64_t[::1, :] x_sc, float64_t[::1] alpha_l, float64_t h_l, float64_t[::1] w_l, float64_t[::1] mu, float64_t[::1, :] cov):

    cdef int nc = x_sc.shape[1]
    cdef int d = x_sc.shape[0]

    cdef float64_t[::1] int_K_l = empty(nc, dtype=float64, order='F')
    cdef float64_t m_Z

    if alpha_l.shape[0] != nc:
        la.value_error("alpha_l has invalid shape")
    if w_l.shape[0] != d:
        la.value_error("w_l has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
    ga.int_K(int_K_l, x_sc, h_l, w_l, mu, cov)
    m_Z = la.dot11(int_K_l, alpha_l)
    if m_Z <= 0:
        warn("m_Z = %s" % m_Z)

    return m_Z


def approx_Z_mean(float64_t[::1, :] xo, float64_t[::1] l, float64_t[::1] mu, float64_t[::1, :] cov):
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t logdet, Kp1, Kp2
    cdef int i

    if l.shape[0] != n:
        la.value_error("l has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(ga.mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # compute approximate integral with trapezoidal rule
    out = 0
    for i in xrange(n-1):
        Kp1 = l[i] * p_xo[i]
        Kp2 = l[i+1] * p_xo[i+1]
        out += diff[i] * (Kp1 + Kp2) / 2.0
    
    return out


def Z_var(float64_t[::1, :] x_s, float64_t[::1, :] x_sc, float64_t[::1] alpha_l, float64_t[::1, :] L_tl, float64_t h_l, float64_t[::1] w_l, float64_t h_tl, float64_t[::1] w_tl, float64_t[::1] mu, float64_t[::1, :] cov):

    cdef int ns = x_s.shape[1]
    cdef int nc = x_sc.shape[1]
    cdef int d = x_sc.shape[0]

    cdef float64_t[::1, :] int_K_l_K_tl_K_l = empty((nc, nc), dtype=float64, order='F')
    cdef float64_t[::1, :] int_K_tl_K_l_mat = empty((ns, nc), dtype=float64, order='F')
    cdef float64_t[::1] beta = empty(ns, dtype=float64, order='F')
    cdef float64_t[::1] L_tl_beta = empty(ns, dtype=float64, order='F')
    cdef float64_t[::1] alpha_int = empty(nc, dtype=float64, order='F')

    cdef float64_t beta2, alpha_int_alpha, V_Z
    cdef int i, j

    if x_s.shape[0] != d:
        la.value_error("x_s has invalid shape")
    if alpha_l.shape[0] != nc:
        la.value_error("alpha_l has invalid shape")
    if L_tl.shape[0] != ns or L_tl.shape[1] != ns:
        la.value_error("L_tl has invalid shape")
    if w_l.shape[0] != d:
        la.value_error("w_l has invalid shape")
    if w_tl.shape[0] != d:
        la.value_error("w_tl has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    # E[m_l C_tl m_l | x_sc] = alpha_l(x_sc)' *
    #    int int K_l(x_sc, x) K_tl(x, x') K_l(x', x_sc) p(x) p(x') dx dx' *
    #    alpha_l(x_sc) - beta(x_sc)'beta(x_sc)
    # Where beta is defined as:
    # beta(x_sc) = inv(L_tl(x_s, x_s)) *
    #    int K_tl(x_s, x) K_l(x, x_sc) p(x) dx *
    #    alpha_l(x_sc)
    ga.int_int_K1_K2_K1(int_K_l_K_tl_K_l, x_sc, h_l, w_l, h_tl, w_tl, mu, cov)
    la.dot12(alpha_l, int_K_l_K_tl_K_l, alpha_int)
    alpha_int_alpha = la.dot11(alpha_int, alpha_l)

    ga.int_K1_K2(int_K_tl_K_l_mat, x_s, x_sc, h_tl, w_tl, h_l, w_l, mu, cov)
    la.dot21(int_K_tl_K_l_mat, alpha_l, beta)
    la.cho_solve_vec(L_tl, beta, L_tl_beta)
    beta2 = la.dot11(beta, L_tl_beta)

    V_Z = alpha_int_alpha - beta2
    if V_Z <= 0:
        warn("V_Z = %s" % V_Z)

    return V_Z


def approx_Z_var(float64_t[::1, :] xo, float64_t[::1] m_l, float64_t[::1, :] C_tl, float64_t[::1] mu, float64_t[::1, :] cov):
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t[::1] buf = empty(n, dtype=float64, order='F')
    cdef float64_t logdet, Kp1, Kp2
    cdef int i, j

    if m_l.shape[0] != n:
        la.value_error("m_l has invalid shape")
    if C_tl.shape[0] != n or C_tl.shape[1] != n:
        la.value_error("C_tl has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(ga.mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # inner integral
    for i in xrange(n):
        buf[i] = 0
        for j in xrange(n-1):
            Kp1 = C_tl[i, j] * m_l[j] * p_xo[j]
            Kp2 = C_tl[i, j+1] * m_l[j+1] * p_xo[j+1]
            buf[i] += diff[j] * (Kp1 + Kp2) / 2.0

    # outer integral
    out = 0
    for i in xrange(n-1):
        Kp1 = buf[i] * m_l[i] * p_xo[i]
        Kp2 = buf[i+1] * m_l[i+1] * p_xo[i+1]
        out += diff[i] * (Kp1 + Kp2) / 2.0

    return out


cdef float64_t _esm(float64_t[::1] int_K_l, float64_t[::1] l_sc, float64_t[::1, :] K_l, float64_t tm_a, float64_t tC_a) except? -1:
    cdef int nca = K_l.shape[0]
    cdef int nc = nca - 1

    cdef float64_t[::1, :] L = empty((nca, nca), dtype=float64, order='F')
    cdef float64_t[::1] A_sca = empty(nca, dtype=float64, order='F')

    cdef float64_t A_a, A_sc_l, e1, e2, E_m2
    cdef int i, j

    if K_l.shape[1] != nca:
        la.value_error("K_l is not square")
    if int_K_l.shape[0] != nca:
        la.value_error("int_K_l has invalid shape")
    if l_sc.shape[0] != nc:
        la.value_error("l_sc has invalid shape")

    la.cho_factor(K_l, L)
    la.cho_solve_vec(L, int_K_l, A_sca)

    A_a = A_sca[-1]
    A_sc_l = la.dot11(A_sca[:-1], l_sc)

    e1 = ga.int_exp_norm(1, tm_a, tC_a)
    e2 = ga.int_exp_norm(2, tm_a, tC_a)

    E_m2 = (A_sc_l**2) + (2*A_sc_l*A_a * e1) + (A_a**2 * e2)

    return E_m2


def expected_squared_mean(float64_t[::1] l_sc, float64_t[::1, :] K_l, float64_t tm_a, float64_t tC_a, float64_t[::1, :] x_sca, float64_t h_l, float64_t[::1] w_l, float64_t[::1] mu, float64_t[::1, :] cov):
    cdef int n = x_sca.shape[1]
    cdef float64_t[::1] int_K_l = empty(n, dtype=float64, order='F')
    ga.int_K(int_K_l, x_sca, h_l, w_l, mu, cov)
    esm = _esm(int_K_l, l_sc, K_l, tm_a, tC_a)
    return esm


def approx_expected_squared_mean(float64_t[::1] l_sc, float64_t[::1, :] K_l, float64_t tm_a, float64_t tC_a, float64_t[::1, :] xo, float64_t[::1, :] Kxxo, float64_t[::1] mu, float64_t[::1, :] cov):
    cdef int m = Kxxo.shape[0]
    cdef float64_t[::1] int_K_l = empty(m, dtype=float64, order='F')
    ga.approx_int_K(int_K_l, xo, Kxxo, mu, cov)
    esm = _esm(int_K_l, l_sc, K_l, tm_a, tC_a)
    return esm


def filter_candidates(float64_t[::1] x_c, float64_t[::1] x_s, float64_t thresh):
    cdef int nc = x_c.shape[0]
    cdef int ns = x_s.shape[0]
    cdef int i, j
    cdef int done = 0
    cdef float64_t diff

    while not done:
        done = 1
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                if np.isnan(x_c[i]) or np.isnan(x_c[j]):
                    continue

                diff = fabs(x_c[i] - x_c[j])
                if diff < thresh:
                    x_c[i] = (x_c[i] + x_c[j]) / 2.0
                    x_c[j] = NAN
                    done = 1

        for i in xrange(nc):
            for j in xrange(ns):
                diff = fabs(x_c[i] - x_s[j])
                if diff < thresh:
                    x_c[i] = NAN

