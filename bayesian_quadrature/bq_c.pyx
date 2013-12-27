from __future__ import division

import numpy as np
cimport numpy as np

import scipy.stats

from libc.math cimport exp, log, fmax, copysign, fabs, M_PI
from cpython cimport bool
from warnings import warn

cdef inv = np.linalg.inv
cdef slogdet = np.linalg.slogdet
cdef dot = np.dot

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps


def mvn_logpdf(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, np.ndarray[DTYPE_t, ndim=1] m, np.ndarray[DTYPE_t, ndim=2] C):
    """Computes the logpdf for a multivariate normal distribution:

    out[i] = N(x_i | m, C)
           = -0.5*log(2*pi)*d - 0.5*(x_i-m)*C^-1*(x_i-m) - 0.5*log(|C|)

    """
    cdef np.ndarray[DTYPE_t, ndim=2] Ci
    cdef int n, d, i, j, k
    cdef DTYPE_t c

    n = x.shape[0]
    d = x.shape[1]
    Ci = inv(C)
    c = log(2 * M_PI) * (-d / 2.) -0.5 * slogdet(C)[1]

    for i in xrange(n):
        out[i] = c - 0.5 * dot(dot(x[i] - m, Ci), x[i] - m)


def improve_covariance_conditioning(np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[long, ndim=1] idx):
    cdef DTYPE_t sqd_jitter = fmax(EPS, np.max(M)) * 1e-3
    cdef int n, i
    for i in xrange(len(idx)):
        M[idx[i], idx[i]] += sqd_jitter


def int_K(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h, np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int K(x', x) N(x' | mu, cov) dx'

    where K is a Gaussian kernel matrix parameterized by `h` and `w`.

    The result is:

    out[i] = h^2 N(x_i | mu, W + cov)

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W
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


def int_K1_K2(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2, DTYPE_t h1, np.ndarray[DTYPE_t, ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, ndim=1] w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int K_1(x1, x') K_2(x', x2) N(x' | mu, cov) dx'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^2 h2^2 N([x1_i, x2_j] | [mu, mu], [W1 + cov, cov; cov, W2 + cov])

    """
    
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef np.ndarray[DTYPE_t, ndim=1] m
    cdef np.ndarray[DTYPE_t, ndim=2] C
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


def int_int_K1_K2_K1(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h1, np.ndarray[DTYPE_t, ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, ndim=1] w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x, x1') K_2(x1', x2') K_1(x2', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i, j] = h1^4 h2^2 |G|^-1 N(x_i | mu, W1 + cov) N(x_j | mu, W1 + cov) N(x_i | x_j, G^-1 (W2 + 2*cov - 2*G*cov) G^-1)

    where G = cov(W1 + cov)^-1

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W1_cov
    cdef np.ndarray[DTYPE_t, ndim=2] G
    cdef np.ndarray[DTYPE_t, ndim=2] Gi
    cdef np.ndarray[DTYPE_t, ndim=2] GWG
    cdef np.ndarray[DTYPE_t, ndim=2] GiGWGGi
    cdef np.ndarray[DTYPE_t, ndim=1] N1
    cdef np.ndarray[DTYPE_t, ndim=2] N2
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

    # compute G = cov*(W1 + cov)^-1
    G = dot(cov, inv(W1_cov))
    Gi = inv(G)
    Gcov = dot(G, cov)
    Gdeti = -slogdet(G)[1]

    # compute G^-1 (W2 + 2*cov - 2*G*cov) G^-1
    GWG = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                GWG[i, j] = w2[i] ** 2 + 2*cov[i, j] - 2*Gcov[i, j]
            else:
                GWG[i, j] = 2*cov[i, j] - 2*Gcov[i, j]

    GiGWGGi = dot(dot(Gi, GWG), Gi)

    # compute N(x | mu, W1 + cov)
    N1 = np.empty(n, dtype=DTYPE)
    mvn_logpdf(N1, x, mu, W1_cov)

    # compute N(x_i | x_j, G^-1 (W2 + 2*cov - 2*G*cov) G^-1)
    N2 = np.empty((n, n), dtype=DTYPE)
    for j in xrange(n):
        mvn_logpdf(N2[:, j], x, x[j], GiGWGGi)

    # put it all together
    for i in xrange(n):
        for j in xrange(n):
            out[i, j] = h1_4_h2_2 * exp(Gdeti + N1[i] + N1[j] + N2[i, j])


def approx_int_int_K1_K2_K1(xo, gp1, gp2, mu, cov):
    K1xxo = gp1.Kxxo(xo)
    K2xoxo = gp2.Kxoxo(xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    int1 = np.trapz(K1xxo[:, None, :] * K2xoxo * p_xo, xo)
    approx_int = np.trapz(K1xxo[:, None] * int1[None] * p_xo, xo)
    return approx_int


def int_int_K1_K2(np.ndarray[DTYPE_t, ndim=1] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h1, np.ndarray[DTYPE_t, ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, ndim=1] w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int int K_1(x2', x1') K_2(x1', x) N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K_1 is a Gaussian kernel matrix parameterized by `h1` and
    `w1`, and K_2 is a Gaussian kernel matrix parameterized by `h2`
    and `w2`.

    The result is:

    out[i] = h1^2 h2^2 N(0 | 0, W1 + 2*cov) N(x_i | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W1_2cov
    cdef np.ndarray[DTYPE_t, ndim=2] C
    cdef np.ndarray[DTYPE_t, ndim=1] N1
    cdef np.ndarray[DTYPE_t, ndim=1] N2
    cdef np.ndarray[DTYPE_t, ndim=2] zx
    cdef np.ndarray[DTYPE_t, ndim=1] zm
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
    C = dot(dot(cov, inv(W1_2cov)), cov)
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


def int_int_K(int d, DTYPE_t h, np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int int K(x1', x2') N(x1' | mu, cov) N(x2' | mu, cov) dx1' dx2'

    where K is a Gaussian kernel parameterized by `h` and `w`.

    The result is:

    out = h^2 N(0 | 0, W + 2*cov)

    """

    cdef np.ndarray[DTYPE_t, ndim=2] W_2cov
    cdef np.ndarray[DTYPE_t, ndim=1] N
    cdef np.ndarray[DTYPE_t, ndim=2] zx
    cdef np.ndarray[DTYPE_t, ndim=1] zm
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


def int_K1_dK2(np.ndarray[DTYPE_t, ndim=3] out, np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2, DTYPE_t h1, np.ndarray[DTYPE_t, ndim=1] w1, DTYPE_t h2, np.ndarray[DTYPE_t, ndim=1] w2, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):
    """Computes integrals of the form:

    int K1(x1, x') dK2(x', x2)/dw2 N(x' | mu, cov) dx'
    
    where K1 is a Gaussian kernel parameterized by `h1` and `w1`, and
    K2 is a Gaussian kernel parameterized by `h2` and `w2`.

    """

    cdef np.ndarray[DTYPE_t, ndim=2] int_K1_K2_mat
    cdef np.ndarray[DTYPE_t, ndim=2] W2_cov
    cdef np.ndarray[DTYPE_t, ndim=2] A
    cdef np.ndarray[DTYPE_t, ndim=2] B
    cdef np.ndarray[DTYPE_t, ndim=2] C
    cdef np.ndarray[DTYPE_t, ndim=2] D
    cdef np.ndarray[DTYPE_t, ndim=2] x1submu
    cdef np.ndarray[DTYPE_t, ndim=2] x2submu
    cdef np.ndarray[DTYPE_t, ndim=2] S
    cdef np.ndarray[DTYPE_t, ndim=2] m
    cdef int n1, n2, d, i, j, k

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    d = x1.shape[1]

    # compute int K_1(x1, x') K_2(x', x2) N(x' | mu, cov) dx'
    int_K1_K2_mat = np.empty((n1, n2), dtype=DTYPE)
    int_K1_K2(int_K1_K2_mat, x1, x2, h1, w1, h2, w2, mu, cov)

    # compute W2 + cov
    W2_cov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W2_cov[i, j] = w2[i] ** 2 + cov[i, j]
            else:
                W2_cov[i, j] = cov[i, j]

    # compute A = cov * (W2 + cov)^-1
    A = dot(cov, inv(W2_cov))
    # compute B = cov - A*cov
    B = cov - dot(A, cov)
    
    # compute B + W1
    B_W1 = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                B_W1[i, j] = B[i, j] + w1[i] ** 2
            else:
                B_W1[i, j] = B[i, j]

    # compute C = B * (B + W1)^-1
    C = dot(B, inv(B_W1))
    # compute D = A - CA - 1
    D = A - dot(C, A) - 1

    # compute x1 - mu
    x1submu = np.empty((n1, d), dtype=DTYPE)
    for i in xrange(n1):
        for j in xrange(d):
            x1submu[i, j] = x1[i, j] - mu[j]

    # compute x2 - mu
    x2submu = np.empty((n2, d), dtype=DTYPE)
    for i in xrange(n2):
        for j in xrange(d):
            x2submu[i, j] = x2[i, j] - mu[j]

    # compute S = B - BC
    S = B - dot(B, C)

    # compute the final values
    m = np.empty((d, 1), dtype=DTYPE)
    for i in xrange(n1):
        for j in xrange(n2):
            m[:] = dot(D, x1submu[i]) + dot(C, x2submu[j])
            for k in xrange(d):
                out[i, j, k] = int_K1_K2_mat[i, j] * (((S[k, k] + m[k] ** 2) / w2[k]**3) - (1.0 / w2[k]))
    

def approx_int_K1_dK2(xo, gp1, gp2, mu, cov):
    K1xxo = gp1.Kxxo(xo)
    dK2xxo = gp2.K.dK_dw(gp2._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    approx_int = np.trapz(
        K1xxo[:, None] * dK2xxo[None, :] * p_xo, xo)[..., None]
    return approx_int


def int_dK(np.ndarray[DTYPE_t, ndim=2] out, np.ndarray[DTYPE_t, ndim=2] x, DTYPE_t h, np.ndarray[DTYPE_t, ndim=1] w, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):

    cdef np.ndarray[DTYPE_t, ndim=1] int_K_vec
    cdef np.ndarray[DTYPE_t, ndim=2] Wcov
    cdef np.ndarray[DTYPE_t, ndim=2] Wcovi
    cdef np.ndarray[DTYPE_t, ndim=2] xsubmu
    cdef np.ndarray[DTYPE_t, ndim=2] A
    cdef np.ndarray[DTYPE_t, ndim=2] m
    cdef np.ndarray[DTYPE_t, ndim=2] S
    cdef int n1, n2, d, i, j

    n = x.shape[0]
    d = x.shape[1]

    # compute int K(x', x) N(x' | mu, cov) dx'
    int_K_vec = np.empty(n, dtype=DTYPE)
    int_K(int_K_vec, x, h, w, mu, cov)

    # compute (W + cov)^-1
    Wcov = np.empty((d, d), dtype=DTYPE)
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                Wcov[i, j] = w[i] ** 2 + cov[i, j]
            else:
                Wcov[i, j] = cov[i, j]
    Wcovi = inv(Wcov)

    # compute x - mu
    xsubmu = np.empty((n, d), dtype=DTYPE)
    for i in xrange(n):
        for j in xrange(d):
            xsubmu[i, j] = x[i, j] - mu[j]

    # compute m = (cov*(w + cov)^-1 - 1)(x - mu)
    A = dot(cov, Wcovi) - 1
    # compute S = cov - cov*(W + cov)^-1*cov
    S = cov - dot(dot(cov, Wcovi), cov)

    # compute final values
    m = np.empty((d, 1), dtype=DTYPE)
    for i in xrange(n):
        m[:] = dot(A, xsubmu[i])
        for j in xrange(d):
            out[i, j] = int_K_vec[i] * (((S[j, j] + m[j] ** 2) / w[j]**3) - (1.0 / w[j]))


def approx_int_dK(xo, gp, mu, cov):
    dKxxo = gp.K.dK_dw(gp._x, xo)
    p_xo = scipy.stats.norm.pdf(xo, mu[0], np.sqrt(cov[0, 0]))
    approx_int = np.trapz(dKxxo * p_xo, xo)[..., None]
    return approx_int


def Z_mean(np.ndarray[DTYPE_t, ndim=2] x_sc, np.ndarray[DTYPE_t, ndim=1] alpha_l, DTYPE_t h_l, np.ndarray[DTYPE_t, ndim=1] w_l, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):

    cdef np.ndarray[DTYPE_t, ndim=1] int_K_l
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


def Z_var(np.ndarray[DTYPE_t, ndim=2] x_sc, np.ndarray[DTYPE_t, ndim=1] alpha_l, np.ndarray[DTYPE_t, ndim=2] inv_L_tl, DTYPE_t h_l, np.ndarray[DTYPE_t, ndim=1] w_l, DTYPE_t h_tl, np.ndarray[DTYPE_t, ndim=1] w_tl, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):

    cdef np.ndarray[DTYPE_t, ndim=2] int_K_l_K_tl_K_l
    cdef np.ndarray[DTYPE_t, ndim=2] int_K_tl_K_l_mat
    cdef np.ndarray[DTYPE_t, ndim=1] beta
    cdef DTYPE_t beta2, alpha_int_alpha, V_Z
    cdef int nc, d

    nc = x_sc.shape[0]
    d = x_sc.shape[1]

    # E[m_l C_tl m_l | x_sc] = alpha_l(x_sc)' *
    #    int int K_l(x_sc, x) K_tl(x, x') K_l(x', x_sc) p(x) p(x') dx dx' *
    #    alpha_l(x_sc) - beta(x_sc)'beta(x_sc)
    # Where beta is defined as:
    # beta(x_sc) = inv(L_tl(x_sc, x_sc)) *
    #    int K_tl(x_sc, x) K_l(x, x_sc) p(x) dx *
    #    alpha_l(x_sc)
    int_K_l_K_tl_K_l = np.empty((nc, nc), dtype=DTYPE)
    int_int_K1_K2_K1(int_K_l_K_tl_K_l, x_sc, h_l, w_l, h_tl, w_tl, mu, cov)

    int_K_tl_K_l_mat = np.empty((nc, nc), dtype=DTYPE)
    int_K1_K2(int_K_tl_K_l_mat, x_sc, x_sc, h_tl, w_tl, h_l, w_l, mu, cov)

    beta = dot(dot(inv_L_tl, int_K_tl_K_l_mat), alpha_l)
    beta2 = dot(beta, beta)
    alpha_int_alpha = dot(dot(alpha_l, int_K_l_K_tl_K_l), alpha_l)
    V_Z = alpha_int_alpha - beta2
    if V_Z <= 0:
        warn("V_Z = %s" % V_Z)

    return V_Z


def expected_squared_mean(np.ndarray[DTYPE_t, ndim=2] x_sca, np.ndarray[DTYPE_t, ndim=1] l_sc, np.ndarray[DTYPE_t, ndim=2] inv_K_l, DTYPE_t tm_a, DTYPE_t tC_a, DTYPE_t h_l, np.ndarray[DTYPE_t, ndim=1] w_l, np.ndarray[DTYPE_t, ndim=1] mu, np.ndarray[DTYPE_t, ndim=2] cov):

    cdef np.ndarray[DTYPE_t, ndim=1] int_K_l
    cdef np.ndarray[DTYPE_t, ndim=1] nlsa
    cdef DTYPE_t nla, nls, nlsnla, nla2, expected_sqd_mean
    cdef int nsca

    nsca = x_sca.shape[0]
    
    # int K_l(x, x_s) p(x) dx inv(K_l(x_s, x_s))
    int_K_l = np.empty(nsca, dtype=DTYPE)
    int_K(int_K_l, x_sca, h_l, w_l, mu, cov)

    # nlsa is the vector of weights, which, when multipled with
    # the likelihoods, gives us our mean estimate for the evidence
    nlsa = dot(int_K_l, inv_K_l)
    nla = nlsa[-1]
    nls = dot(nlsa[:-1], l_sc)

    # put it all together
    nlsnla = 2 * nls * nla * exp(tm_a + 0.5 * tC_a)
    nla2 = (nla ** 2) * exp(2 * tm_a + 2 * tC_a)
    expected_sqd_mean = nls ** 2 + nlsnla + nla2

    return expected_sqd_mean

