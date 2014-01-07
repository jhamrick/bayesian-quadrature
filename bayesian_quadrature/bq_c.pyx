# cython: profile=True

from __future__ import division

import numpy as np
cimport numpy as np

import scipy.stats
import scipy.linalg

from numpy.linalg import LinAlgError
from libc.math cimport exp, log, fmax, copysign, fabs, M_PI
from cpython cimport bool
from warnings import warn

cdef dot = np.dot

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t integer

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps
cdef DTYPE_t NAN = np.nan
cdef char UPLO = 'L'

cdef extern from "clapack.h":
    integer dpotrf_(char *uplo, integer *n, DTYPE_t *a, integer *lda, integer *info)
    integer dpotrs_(char *uplo, integer *n, integer *nrhs, DTYPE_t *a, integer *lda, DTYPE_t *b, integer *ldb, integer *info)
    integer dgetrf_(integer *m, integer *n, DTYPE_t *a, integer *lda, integer *ipiv, integer *info)

cdef cho_factor(np.ndarray[DTYPE_t, mode='c', ndim=2] C):
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] L = np.array(C, dtype=DTYPE, copy=True, order='F')
    cdef integer n = C.shape[0]
    cdef integer info

    if C.shape[1] != n:
        raise ValueError("C is not square")

    dpotrf_(&UPLO, &n, &L[0, 0], &n, &info)

    if info < 0:
        raise ValueError("illegal value in argument %d" % (-info,))
    elif info > 0:
        raise LinAlgError("leading minor of order %d is not positive definite" % info)

    return L


cdef cho_solve_vec(np.ndarray[DTYPE_t, mode='fortran', ndim=2] L, np.ndarray[DTYPE_t, mode='c', ndim=1] b):
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=1] xf = np.array(b, dtype=DTYPE, copy=True, order='F')
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] xc
    cdef integer n = L.shape[0]
    cdef integer nrhs = 1
    cdef integer info

    if L.shape[1] != n:
        raise ValueError("L is not square")
    if b.shape[0] != n:
        raise ValueError("b has invalid size %s" % b.shape[0])

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &xf[0], &n, &info)

    if info < 0:
        raise ValueError("illegal value in argument %d" % (-info,))

    xc = np.array(xf, dtype=DTYPE, order='C')
    return xc


cdef cho_solve_mat(np.ndarray[DTYPE_t, mode='fortran', ndim=2] L, np.ndarray[DTYPE_t, mode='c', ndim=2] b):
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] xf = np.array(b, dtype=DTYPE, copy=True, order='F')
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] xc
    cdef integer n = L.shape[0]
    cdef integer nrhs = 2
    cdef integer info

    if L.shape[1] != n:
        raise ValueError("L is not square")
    if b.shape[0] != n or b.shape[1] != n:
        raise ValueError("b has invalid shape (%d, %d)" % (b.shape[0], b.shape[1]))

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &xf[0, 0], &n, &info)

    if info < 0:
        raise ValueError("illegal value in argument %d" % (-info,))

    xc = np.array(xf, dtype=DTYPE, order='C')
    return xc


cdef logdet(np.ndarray[DTYPE_t, mode='c', ndim=2] A):
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] LU
    cdef np.ndarray[integer, mode='fortran', ndim=1] P
    cdef integer n = A.shape[0]
    cdef integer info
    cdef DTYPE_t logdet
    cdef int i

    if A.shape[1] != n:
        raise ValueError("A is not square")

    LU = np.array(A, dtype=DTYPE, copy=True, order='F')
    P = np.empty(n, dtype=np.int32, order='F')

    dgetrf_(&n, &n, &LU[0, 0], &n, &P[0], &info)

    if info < 0:
        raise ValueError("illegal value in argument %d" % (-info,))
    elif info > 0:
        raise LinAlgError("U(%d, %d) is zero; matrix is singular" % (info, info))
    
    logdet = 0
    for i in xrange(n):
        logdet += log(LU[i, i])

    return logdet


def mvn_logpdf(np.ndarray[DTYPE_t, mode='c', ndim=1] out, np.ndarray[DTYPE_t, mode='c', ndim=2] x, np.ndarray[DTYPE_t, mode='c', ndim=1] m, np.ndarray[DTYPE_t, mode='c', ndim=2] C):
    """Computes the logpdf for a multivariate normal distribution:

    out[i] = N(x_i | m, C)
           = -0.5*log(2*pi)*d - 0.5*(x_i-m)*C^-1*(x_i-m) - 0.5*log(|C|)

    """
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] L
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] b
    cdef int n, d, i, j, k
    cdef DTYPE_t c

    n = x.shape[0]
    d = x.shape[1]

    b = np.empty(d, dtype=DTYPE)

    L = cho_factor(C)
    c = log(2 * M_PI) * (-d / 2.) -0.5 * logdet(C)

    for i in xrange(n):
        b[:] = cho_solve_vec(L, x[i] - m)
        out[i] = c - 0.5 * dot(x[i] - m, b)


cpdef int_exp_norm(DTYPE_t c, DTYPE_t m, DTYPE_t S):
    """Computes integrals of the form:

    int exp(cx) N(x | m, S) = exp(cm + (1/2) c^2 S)

    """
    return exp((c * m) + (0.5 * c ** 2 * S))


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


def int_K(np.ndarray[DTYPE_t, mode='c', ndim=1] out, np.ndarray[DTYPE_t, mode='c', ndim=2] x, DTYPE_t h, np.ndarray[DTYPE_t, mode='c', ndim=1] w, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):
    """Computes integrals of the form:

    int K(x', x) N(x' | mu, cov) dx'

    where K is a Gaussian kernel matrix parameterized by `h` and `w`.

    The result is:

    out[i] = h^2 N(x_i | mu, W + cov)

    """

    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] W
    cdef int n, d, i, j
    cdef DTYPE_t h_2

    n = x.shape[0]
    d = x.shape[1]
    h_2 = h ** 2

    W = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W[i, j] = cov[i, j] + w[i] ** 2
            else:
                W[i, j] = cov[i, j]

    mvn_logpdf(out, x, mu, W)
    for i in xrange(n):
        out[i] = h_2 * exp(out[i])


def approx_int_K(xo, gp, mu, cov):
    Kxxo = gp.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    approx_int = np.trapz(Kxxo * p_xo, xo)
    return approx_int


def int_K1_K2(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=2] x1, np.ndarray[DTYPE_t, mode='c', ndim=2] x2, DTYPE_t h1, np.ndarray[DTYPE_t, mode='c', ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, mode='c', ndim=1] w2, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):
    """Computes integrals of the form:

    int K_1(x1, x') K_2(x', x2) N(x' | mu, cov) dx'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^2 h2^2 N([x1_i, x2_j] | [mu, mu], [W1 + cov, cov; cov, W2 + cov])

    """
    
    cdef np.ndarray[DTYPE_t, mode='c', ndim=3] x
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] m
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] C
    cdef int n1, n2, d, i, j, k
    cdef DTYPE_t ha, hb

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]

    x = np.empty((n1, n2, 2 * d), dtype=DTYPE)
    m = np.empty(2 * d, dtype=DTYPE)
    C = np.empty((2 * d, 2 * d), dtype=DTYPE)

    h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)

    # compute concatenated means [mu, mu]
    for i in xrange(d):
        m[i] = mu[i]
        m[i + d] = mu[i]

    # compute concatenated covariances [W1 + cov, cov; cov; W2 + cov]
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w1[i] ** 2 + cov[i, j]
                C[i + d, j + d] = w2[i] ** 2 + cov[i, j]
            else:
                C[i, j] = cov[i, j]
                C[i + d, j + d] = cov[i, j]

            C[i, j + d] = cov[i, j]
            C[i + d, j] = cov[i, j]

    # compute concatenated x
    for i in xrange(n1):
        for j in xrange(n2):
            for k in xrange(d):
                x[i, j, k] = x1[i, k]
                x[i, j, k + d] = x2[j, k]

    # compute pdf
    for i in xrange(n1):
        mvn_logpdf(out[i], x[i], m, C)
        for j in xrange(n2):
            out[i, j] = h1_2_h2_2 * exp(out[i, j])


def approx_int_K1_K2(xo, gp1, gp2, mu, cov):
    K1xxo = gp1.Kxxo(xo)
    K2xxo = gp2.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    approx_int = np.trapz(K1xxo[:, None] * K2xxo[None, :] * p_xo, xo)
    return approx_int


def int_int_K1_K2_K1(np.ndarray[DTYPE_t, mode='c', ndim=2] out, np.ndarray[DTYPE_t, mode='c', ndim=2] x, DTYPE_t h1, np.ndarray[DTYPE_t, mode='c', ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, mode='c', ndim=1] w2, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x, x1') K_2(x1', x2') K_1(x2', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^4 h2^2 |G|^-1 N(x_i | mu, W1 + cov) N(x_j | mu, W1 + cov) N(x_i | x_j, G^-1 (W2 + 2*cov - 2*G*cov) G^-1)

    where G = cov(W1 + cov)^-1

    """

    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] W1_cov
    cdef np.ndarray[DTYPE_t, mode='fortran', ndim=2] L
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] C
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] cLc
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] cx
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] N1
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] N2
    cdef int n, d, i, j
    cdef DTYPE_t h1_4, h2_2, Gdeti

    n = x.shape[0]
    d = x.shape[1]

    h1_4_h2_2 = (h1 ** 4) * (h2 ** 2)

    # compute W1 + cov
    W1_cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W1_cov[i, j] = cov[i, j] + w1[i] ** 2
            else:
                W1_cov[i, j] = cov[i, j]

    # compute cov*(W1 + cov)^-1*cov
    L = cho_factor(W1_cov)
    cLc = dot(cov, cho_solve_mat(L, cov))

    # compute C = W2 + 2*cov - 2*cov*(W1 + cov)^-1*cov
    C = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w2[i] ** 2 + 2*cov[i, j] - 2*cLc[i, j]
            else:
                C[i, j] = 2*cov[i, j] - 2*cLc[i, j]

    # compute N(x | mu, W1 + cov)
    N1 = np.empty(n, dtype=DTYPE)
    mvn_logpdf(N1, x, mu, W1_cov)

    # compute cov*(W1 + cov)^-1 * x
    cx = np.empty((n, d), dtype=DTYPE)
    for i in xrange(n):
        cx[i] = dot(cov, cho_solve_vec(L, x[i]))

    # compute N(cov*(W1 + cov)^-1 * x_i | cov*(W1 + cov)^-1 * x_j, C)
    N2 = np.empty((n, n), dtype=DTYPE)
    for j in xrange(n):
        mvn_logpdf(N2[j], cx, cx[j], C)

    # put it all together
    for i in xrange(n):
        for j in xrange(n):
            out[i, j] = h1_4_h2_2 * exp(N1[i] + N1[j] + N2[j, i])


def approx_int_int_K1_K2_K1(xo, gp1, gp2, mu, cov):
    K1xxo = gp1.Kxxo(xo)
    K2xoxo = gp2.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    int1 = np.trapz(K1xxo[:, None, :] * K2xoxo * p_xo, xo)
    approx_int = np.trapz(K1xxo[:, None] * int1[None] * p_xo, xo)
    return approx_int


def int_int_K1_K2(np.ndarray[DTYPE_t, mode='c', ndim=1] out, np.ndarray[DTYPE_t, mode='c', ndim=2] x, DTYPE_t h1, np.ndarray[DTYPE_t, mode='c', ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, mode='c', ndim=1] w2, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x2', x1') K_2(x1', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i] = h1^2 h2^2 N(0 | 0, W1 + 2*cov) N(x_i | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)

    """

    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] W1_2cov
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] C
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] N1
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] N2
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] zx
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] zm
    cdef int n, d, i, j
    cdef DTYPE_t h1_2, h2_2

    n = x.shape[0]
    d = x.shape[1]

    h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)

    # compute W1 + 2*cov
    W1_2cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W1_2cov[i, j] = 2*cov[i, j] + w1[i] ** 2
            else:
                W1_2cov[i, j] = 2*cov[i, j]

    # compute N(0 | 0, W1 + 2*cov)
    N1 = np.empty(1, dtype=DTYPE)
    zx = np.zeros((1, d), dtype=DTYPE)
    zm = np.zeros(d, dtype=DTYPE)
    mvn_logpdf(N1, zx, zm, W1_2cov)

    # compute W2 + cov - cov*(W1 + 2*cov)^-1*cov
    C = dot(cov, cho_solve_mat(cho_factor(W1_2cov), cov))
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w2[i] ** 2 + cov[i, j] - C[i, j]
            else:
                C[i, j] = cov[i, j] - C[i, j]

    # compute N(x | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)
    N2 = np.empty(n, dtype=DTYPE)
    mvn_logpdf(N2, x, mu, C)

    for i in xrange(n):
        out[i] = h1_2_h2_2 * exp(N1[0] + N2[i])


def approx_int_int_K1_K2(xo, gp1, gp2, mu, cov):
    K1xoxo = gp1.Kxoxo(xo)
    K2xxo = gp2.Kxxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    int1 = np.trapz(K1xoxo * K2xxo[:, :, None] * p_xo, xo)
    approx_int = np.trapz(int1 * p_xo, xo)
    return approx_int


def int_int_K(int d, DTYPE_t h, np.ndarray[DTYPE_t, mode='c', ndim=1] w, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):
    """Computes integrals of the form:

    int int K(x1', x2') N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K is a Gaussian kernel parameterized by `h` and `w`.

    The result is:

    out = h^2 N(0 | 0, W + 2*cov)

    """

    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] W_2cov
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] N
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] zx
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] zm
    cdef int i, j

    # compute W + 2*cov
    W_2cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W_2cov[i, j] = 2*cov[i, j] + w[i] ** 2
            else:
                W_2cov[i, j] = 2*cov[i, j]

    # compute N(0 | 0, W1 + 2*cov)
    N = np.empty(1, dtype=DTYPE)
    zx = np.zeros((1, d), dtype=DTYPE)
    zm = np.zeros(d, dtype=DTYPE)
    mvn_logpdf(N, zx, zm, W_2cov)

    return (h ** 2) * exp(N[0])


def approx_int_int_K(xo, gp, mu, cov):
    Kxoxo = gp.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    approx_int = np.trapz(np.trapz(Kxoxo * p_xo, xo) * p_xo, xo)
    return approx_int


def Z_mean(np.ndarray[DTYPE_t, mode='c', ndim=2] x_sc, np.ndarray[DTYPE_t, mode='c', ndim=1] alpha_l, DTYPE_t h_l, np.ndarray[DTYPE_t, mode='c', ndim=1] w_l, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):

    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] int_K_l
    cdef int nc, d
    cdef DTYPE_t m_Z

    nc = x_sc.shape[0]
    d = x_sc.shape[1]

    # E[m_l | x_s] = (int K_l(x, x_s) p(x) dx) alpha_l(x_s)
    int_K_l = np.empty(nc, dtype=DTYPE)
    int_K(int_K_l, x_sc, h_l, w_l, mu, cov)
    m_Z = dot(int_K_l, alpha_l)
    if m_Z <= 0:
        warn("m_Z = %s" % m_Z)

    return m_Z


def Z_var(np.ndarray[DTYPE_t, mode='c', ndim=2] x_s, np.ndarray[DTYPE_t, mode='c', ndim=2] x_sc, np.ndarray[DTYPE_t, mode='c', ndim=1] alpha_l, np.ndarray[DTYPE_t, mode='fortran', ndim=2] L_tl, DTYPE_t h_l, np.ndarray[DTYPE_t, mode='c', ndim=1] w_l, DTYPE_t h_tl, np.ndarray[DTYPE_t, mode='c', ndim=1] w_tl, np.ndarray[DTYPE_t, mode='c', ndim=1] mu, np.ndarray[DTYPE_t, mode='c', ndim=2] cov):

    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] int_K_l_K_tl_K_l
    cdef np.ndarray[DTYPE_t, mode='c', ndim=2] int_K_tl_K_l_mat
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] beta
    cdef DTYPE_t beta2, alpha_int_alpha, V_Z
    cdef int ns, nc, d

    ns = x_s.shape[0]
    nc = x_sc.shape[0]
    d = x_sc.shape[1]

    # E[m_l C_tl m_l | x_sc] = alpha_l(x_sc)' *
    #    int int K_l(x_sc, x) K_tl(x, x') K_l(x', x_sc) p(x) p(x') dx dx' *
    #    alpha_l(x_sc) - beta(x_sc)'beta(x_sc)
    # Where beta is defined as:
    # beta(x_sc) = inv(L_tl(x_s, x_s)) *
    #    int K_tl(x_s, x) K_l(x, x_sc) p(x) dx *
    #    alpha_l(x_sc)
    int_K_l_K_tl_K_l = np.empty((nc, nc), dtype=DTYPE)
    int_int_K1_K2_K1(int_K_l_K_tl_K_l, x_sc, h_l, w_l, h_tl, w_tl, mu, cov)

    int_K_tl_K_l_mat = np.empty((ns, nc), dtype=DTYPE)
    int_K1_K2(int_K_tl_K_l_mat, x_s, x_sc, h_tl, w_tl, h_l, w_l, mu, cov)

    beta = dot(int_K_tl_K_l_mat, alpha_l)
    beta2 = dot(beta, cho_solve_vec(L_tl, beta))
    alpha_int_alpha = dot(dot(alpha_l, int_K_l_K_tl_K_l), alpha_l)
    V_Z = alpha_int_alpha - beta2
    if V_Z <= 0:
        warn("V_Z = %s" % V_Z)

    return V_Z


def expected_squared_mean(np.ndarray[DTYPE_t, mode='c', ndim=1] int_K_l, np.ndarray[DTYPE_t, mode='c', ndim=1] l_sc, np.ndarray[DTYPE_t, mode='c', ndim=2] K_l, DTYPE_t tm_a, DTYPE_t tC_a):
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] A_sca
    cdef DTYPE_t A_a, A_sc_l, e1, e2, E_m2

    A_sca = cho_solve_vec(cho_factor(K_l), int_K_l)
    A_a = A_sca[-1]
    A_sc_l = dot(A_sca[:-1], l_sc)

    e1 = int_exp_norm(1, tm_a, tC_a)
    e2 = int_exp_norm(2, tm_a, tC_a)

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

