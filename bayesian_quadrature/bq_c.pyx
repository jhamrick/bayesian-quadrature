# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True

import numpy as np
import scipy.stats
import scipy.linalg
from numpy.linalg import LinAlgError
from warnings import warn
from numpy import float64, int32
from numpy import empty

######################################################################

from libc.math cimport exp, log, fmax, fabs, M_PI, cos, INFINITY
from numpy cimport float64_t

cimport cython
cimport linalg_c as la
cimport gauss_c as ga

cdef extern from "math.h":
    double j0(double x)

######################################################################

cdef float64_t MIN = log(np.exp2(float64(np.finfo(float64).minexp + 4)))
cdef float64_t EPS = np.finfo(float64).eps
cdef float64_t NAN = np.nan

######################################################################

cpdef float64_t vonmises_logpdf(float64_t x, float64_t mu, float64_t kappa):
    r"""
    Computes the log-PDF for the Von Mises distribution at location
    :math:`x`:

    .. math ::
    
        p(x | \mu, \kappa) = \frac{\exp(\kappa\cos(x-\mu))}{2\pi I_0(\kappa)}

    Where :math:`I_0` is the Bessel function of order 0.

    Parameters
    ----------
    x : float64_t
        Input angle (radians)
    mu : float64_t
        Mean angle (radians)
    kappa : float64_t
        Spread parameter

    Returns
    -------
    out : value of the log-PDF

    """
    C = -log(2 * M_PI * j0(kappa))
    p = C + (kappa * cos(x - mu))
    return p


def p_x_gaussian(float64_t[::1] p_x, float64_t[::1, :] x, float64_t[::1] mu, float64_t[::1, :] cov):
    r"""
    Computes the Gaussian PDF of input locations :math:`x`.

    Parameters
    ----------
    p_x : float64_t[::1]
        :math:`n` output vector of probabilities
    x : float64_t[::1, :]
        :math:`d\times n` input locations
    mu : float64_t[::1]
        :math:`d` mean vector
    cov : float64_t[::1, :]
        :math:`d\times d` covariance matrix

    """

    cdef int d = x.shape[0]
    cdef int n = x.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef int i
    
    if p_x.shape[0] != n:
        la.value_error("p_x has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_x[i] = exp(ga.mvn_logpdf(x[:, i], mu, L, logdet))


def p_x_vonmises(float64_t[::1] p_x, float64_t[::1] x, float64_t mu, float64_t kappa):
    r"""
    Computes the Von Mises PDF of input locations :math:`x`.

    Parameters
    ----------
    p_x : float64_t[::1]
        :math:`n` output vector of probabilities
    x : float64_t[::1, :]
        :math:`n` input vector of locations
    mu : float64_t[::1]
        mean parameter
    kappa : float64_t[::1, :]
        spread parameter

    """

    cdef int n = x.shape[0]
    cdef int i
    
    if p_x.shape[0] != n:
        la.value_error("p_x has invalid shape")

    for i in xrange(n):
        p_x[i] = exp(vonmises_logpdf(x[i], mu, kappa))


@cython.boundscheck(True)
@cython.wraparound(True)
def improve_covariance_conditioning(float64_t[:, ::1] M, float64_t[::1] jitters, long[::1] idx):
    r"""
    Add noise to the indices `idx` along the diagonal of `M`. Update
    the corresponding locations in the `jitters` vector to include
    this new noise.

    """
    cdef float64_t sqd_jitter = fmax(EPS, np.max(M)) * 1e-4
    cdef int i
    for i in xrange(len(idx)):
        jitters[idx[i]] += sqd_jitter
        M[idx[i], idx[i]] += sqd_jitter


@cython.boundscheck(True)
@cython.wraparound(True)
def remove_jitter(float64_t[:, ::1] M, float64_t[::1] jitters, long[::1] idx):
    r"""
    Remove noise from the indices `idx` along the diagonal of `M`. Set
    the corresponding locations in the `jitters` vector to zero.

    """
    cdef int i
    for i in xrange(len(idx)):
        M[idx[i], idx[i]] -= jitters[idx[i]]
        jitters[idx[i]] = 0


def Z_mean(float64_t[::1, :] x_sc, float64_t[::1] alpha_l, float64_t h_l, float64_t[::1] w_l, float64_t[::1] mu, float64_t[::1, :] cov):
    r"""
    Compute the mean of the integral:

    .. math ::

        Z = \int \ell(x)\mathcal{N}(x \big\vert \mu, \Sigma)\ \mathrm{d}x

    where the mean is defined as:

    .. math ::
    
        m(Z) = E[Z \big\vert \bar{\ell}(x_sc)]

    Parameters
    ----------
    x_sc : float64_t[::1, :]
        :math:`d\times n` vector of observed and candidate locations
    alpha_l : float64_t[::1]
        :math:`K_\ell(x_{sc}, x_{sc})^{-1}\bar{\ell}(x_{sc})`
    h_l : float64_t
        output scale parameter for kernel :math:`K_\ell`
    w_l : float64_t[::1]
        :math:`d` vector of lengthscales for kernel :math:`K_\ell`
    mu : float64_t[::1]
        :math:`d` prior mean
    cov : float64_t[::1, :]
        :math:`d\times d` prior covariance

    Returns
    -------
    out : mean of :math:`Z`

    """

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


def approx_Z_mean(float64_t[::1, :] xo, float64_t[::1] p_xo, float64_t[::1] l):
    r"""
    Approximate the mean of the integral:

    .. math ::

        Z = \int \ell(x)p(x)\ \mathrm{d}x

    Parameters
    ----------
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    p_xo : float64_t[::1]
        :math:`n` vector prior probabilities at approximation locations
    l : float64_t[::1]
        :math:`n` vector of likelihood evaluated at approximation locations

    Returns
    -------
    out : approximate mean of :math:`Z`

    """

    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t Kp1, Kp2
    cdef int i

    if p_xo.shape[0] != n:
        la.value_error("p_xo has invalid shape")
    if l.shape[0] != n:
        la.value_error("l has invalid shape")

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
    r"""
    Compute the variance of the integral:

    .. math ::

        Z = \int \ell(x)\mathcal{N}(x \big\vert \mu, \Sigma)\ \mathrm{d}x

    where the variance is defined as:

    .. math ::
    
        V(Z) = \int\int m_\ell(x) C_{\log\ell}(x, x^\prime) m_\ell(x^\prime) p(x)p(x^\prime)\ \mathrm{d}x\ \mathrm{d}x^\prime

    Parameters
    ----------
    x_s : float64_t[::1, :]
        :math:`d\times n_s` vector of observed locations
    x_sc : float64_t[::1, :]
        :math:`d\times n_{sc}` vector of observed and candidate locations
    alpha_l : float64_t[::1]
        :math:`K_\ell(x_{sc}, x_{sc})^{-1}\bar{\ell}(x_{sc})`
    L_tl : float64_t[::1, :]
        lower-triangular Cholesky factor of :math:`K_{\log\ell}(x_{s}, x_{s})`
    h_l : float64_t
        output scale parameter for kernel :math:`K_\ell`
    w_l : float64_t[::1]
        :math:`d` vector of lengthscales for kernel :math:`K_\ell`
    h_tl : float64_t
        output scale parameter for kernel :math:`K_{\log\ell}`
    w_tl : float64_t[::1]
        :math:`d` vector of lengthscales for kernel :math:`K_{\log\ell}`
    mu : float64_t[::1]
        :math:`d` prior mean
    cov : float64_t[::1, :]
        :math:`d\times d` prior covariance

    Returns
    -------
    out : variance of :math:`Z`

    """

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


def approx_Z_var(float64_t[::1, :] xo, float64_t[::1] p_xo, float64_t[::1] m_l, float64_t[::1, :] C_tl):
    r"""
    Approximate the variance of the integral:

    .. math ::

        Z = \int \ell(x)p(x)\ \mathrm{d}x

    where the variance is defined as:

    .. math ::
    
        V(Z) = \int\int m_\ell(x) C_{\log\ell}(x, x^\prime) m_\ell(x^\prime) p(x)p(x^\prime)\ \mathrm{d}x\ \mathrm{d}x^\prime

    Parameters
    ----------
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    p_xo : float64_t[::1]
        :math:`n` vector prior probabilities at approximation locations
    m_l : float64_t[::1]
        :math:`n` vector of likelihoods evaluated at approximation locations
    C_tl : float64_t[::1, :]
        :math:`n\times n` covariance matrix for :math:`\log\ell` evaluated at approximation locations

    Returns
    -------
    out : approximate variance of :math:`Z`

    """

    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t[::1] buf = empty(n, dtype=float64, order='F')
    cdef float64_t Kp1, Kp2
    cdef int i, j

    if p_xo.shape[0] != n:
        la.value_error("p_xo has invalid shape")
    if m_l.shape[0] != n:
        la.value_error("m_l has invalid shape")
    if C_tl.shape[0] != n or C_tl.shape[1] != n:
        la.value_error("C_tl has invalid shape")

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
    r"""
    Computes the expected squared mean of :math:`Z` given a new
    observation at :math:`x_a`.

    .. math ::
    
        E[m(Z)^2 \big\vert x_a] = \int m(Z | \ell_s, \ell_a)^2 \mathcal{N}(\log\ell_a | \hat{m}_a, \hat{C}_a)\ \mathrm{d}\log\ell_a

    Parameters
    ----------
    int_K_l : float64_t[::1]
        :math:`\int K_\ell(x_{sca}, x) p(x)\ \mathrm{d}x`
    l_sc : float64_t[::1]
        :math:`n_{sc}` vector of observed and candidate locations
    K_l : float64_t[::1, :]
        :math:`n_{sca}\times n_{sca}` kernel matrix :math:`K_{\ell}(x_{sca}, x_{sca})`
    tm_a : float64_t
        prior mean of :math:`\log\ell_a`
    tC_a : float64_t
        prior variance of :math:`\log\ell_a`

    Returns
    -------
    out : expected squared mean of :math:`Z`

    """    

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

    A_a = A_sca[nca-1]
    A_sc_l = la.dot11(A_sca[:nca-1], l_sc)

    e1 = ga.int_exp_norm(1, tm_a, tC_a)
    if e1 == INFINITY:
        E_m2 = INFINITY
        return E_m2

    e2 = ga.int_exp_norm(2, tm_a, tC_a)
    if e2 == INFINITY:
        E_m2 = INFINITY
        return E_m2

    E_m2 = (A_sc_l**2) + (2*A_sc_l*A_a * e1) + (A_a**2 * e2)
    return E_m2


def expected_squared_mean(float64_t[::1] l_sc, float64_t[::1, :] K_l, float64_t tm_a, float64_t tC_a, float64_t[::1, :] x_sca, float64_t h_l, float64_t[::1] w_l, float64_t[::1] mu, float64_t[::1, :] cov):
    r"""
    Computes the expected squared mean of :math:`Z` given a new
    observation at :math:`x_a`.

    .. math ::
    
        E[m(Z)^2 \big\vert x_a] = \int m(Z | \ell_s, \ell_a)^2 \mathcal{N}(\log\ell_a | \hat{m}_a, \hat{C}_a)\ \mathrm{d}\log\ell_a

    Parameters
    ----------
    l_sc : float64_t[::1]
        :math:`n_{sc}` vector of observed and candidate locations
    K_l : float64_t[::1, :]
        :math:`n_{sca}\times n_{sca}` kernel matrix :math:`K_{\ell}(x_{sca}, x_{sca})`
    tm_a : float64_t
        prior mean of :math:`\log\ell_a`
    tC_a : float64_t
        prior variance of :math:`\log\ell_a`
    x_sca : float64_t[::1, :]
        :math:`d\times n_{sca}` input vector
    h_l : float64_t
        output scale kernel parameter for :math:`K_\ell`
    w_l : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_\ell`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : expected squared mean of :math:`Z`

    """    

    cdef int n = x_sca.shape[1]
    cdef float64_t[::1] int_K_l = empty(n, dtype=float64, order='F')
    ga.int_K(int_K_l, x_sca, h_l, w_l, mu, cov)
    esm = _esm(int_K_l, l_sc, K_l, tm_a, tC_a)
    return esm


def approx_expected_squared_mean(float64_t[::1] l_sc, float64_t[::1, :] K_l, float64_t tm_a, float64_t tC_a, float64_t[::1, :] xo, float64_t[::1] p_xo, float64_t[::1, :] Kxxo):
    r"""
    Approximates the expected squared mean of :math:`Z` given a new
    observation at :math:`x_a`.

    .. math ::
    
        E[m(Z)^2 \big\vert x_a] = \int m(Z | \ell_s, \ell_a)^2 \mathcal{N}(\log\ell_a | \hat{m}_a, \hat{C}_a)\ \mathrm{d}\log\ell_a

    Parameters
    ----------
    l_sc : float64_t[::1]
        :math:`n_{sc}` vector of observed and candidate locations
    K_l : float64_t[::1, :]
        :math:`n_{sca}\times n_{sca}` kernel matrix :math:`K_{\ell}(x_{sca}, x_{sca})`
    tm_a : float64_t
        prior mean of :math:`\log\ell_a`
    tC_a : float64_t
        prior variance of :math:`\log\ell_a`
    xo : float64_t[::1, :]
        :math:`d\times m` vector of approximation locations
    Kxxo : float64_t[::1, :]
        :math:`n_{sca}\times m` kernel matrix
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : approximate expected squared mean of :math:`Z`

    """    

    cdef int m = Kxxo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1] int_K_l = empty(m, dtype=float64, order='F')
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t Kp1, Kp2
    cdef int i, j

    if p_xo.shape[0] != n:
        la.value_error("p_xo has invalid shape")
    if Kxxo.shape[1] != n:
        la.value_error("Kxxo has invalid shape")

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # compute approximate integral with trapezoidal rule
    for i in xrange(m):
        int_K_l[i] = 0
        for j in xrange(n-1):
            Kp1 = Kxxo[i, j] * p_xo[j]
            Kp2 = Kxxo[i, j+1] * p_xo[j+1]
            int_K_l[i] += diff[j] * (Kp1 + Kp2) / 2.0

    esm = _esm(int_K_l, l_sc, K_l, tm_a, tC_a)
    return esm


def filter_candidates(float64_t[::1] x_c, float64_t[::1] x_s, float64_t thresh):
    r"""

    Given a vector of possible candidate locations, :math:`x_c`,
    filter out locations which are close to one or more observations
    :math:`x_s`, or to other candidate locations.

    Parameters
    ----------
    x_c : float64_t[::1]
        potential candidate locations
    x_s : float64_t[::1]
        observed locations
    thresh : float64_t
        minimum allowed distance

    """
    cdef int nc = x_c.shape[0]
    cdef int ns = x_s.shape[0]
    cdef int i, j
    cdef int done = 0
    cdef float64_t diff

    while not done:
        done = 1

        # find candidates that are close to each other, and replace
        # them with their average location
        for i in xrange(nc):
            for j in xrange(i+1, nc):
                if np.isnan(x_c[i]) or np.isnan(x_c[j]):
                    continue

                diff = fabs(x_c[i] - x_c[j])
                if diff < thresh:
                    x_c[i] = (x_c[i] + x_c[j]) / 2.0
                    x_c[j] = NAN
                    done = 1

        # remove candidates that are too close to an observation
        for i in xrange(nc):
            for j in xrange(ns):
                diff = fabs(x_c[i] - x_s[j])
                if diff < thresh:
                    x_c[i] = NAN

