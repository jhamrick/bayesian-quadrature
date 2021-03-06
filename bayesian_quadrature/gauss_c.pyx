# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True

from numpy.linalg import LinAlgError
from numpy import array, empty, zeros
from numpy import float64, int32
from numpy import exp2, finfo

######################################################################

from numpy cimport ndarray, float64_t, int32_t
from libc.math cimport exp, log, M_PI, INFINITY
cimport linalg_c as la

cdef float64_t MAX = log(exp2(float64(finfo(float64).maxexp - 4)))

######################################################################

cpdef float64_t mvn_logpdf(float64_t[::1] x, float64_t[::1] m, float64_t[::1, :] L, float64_t logdet) except? -1:
    r""" 
    Computes the log-PDF for a multivariate normal distribution for a
    single point :math:`x`.

    .. math ::

        \mathcal{N}(x \big\vert{} m, C) = -\frac{1}{2} \left(d\log(2\pi) + \log \big\vert{}C\big\vert{} + (x-m)C^{-1}(x-m)^T\right)

    Parameters
    ----------
    x : float64_t[::1]
        :math:`d` input location
    m : float64_t[::1]
        :math:`d` mean parameter
    L : float64_t[::1, :]
        :math:`d\times d` lower-triangular Cholesky factor of the covariance :math:`C`
    logdet : float64_t
        log determinant of the covariance :math:`C`

    Returns
    -------
    out : log PDF evaluated at :math:`x`

    """

    cdef int d = x.shape[0]
    cdef float64_t[::1] diff = empty(d, dtype=float64)
    cdef float64_t[::1] buf = empty(d, dtype=float64)
    cdef float64_t c = log(2 * M_PI) * d + logdet
    cdef float64_t out

    if m.shape[0] != d:
        la.value_error("m has invalid size")
    if L.shape[0] != d or L.shape[1] != d:
        la.value_error("C has invalid size")

    for i in xrange(d):
        diff[i] = x[i] - m[i]

    la.cho_solve_vec(L, diff, buf)
    out = -0.5 * (c + la.dot11(diff, buf))
    return out


cpdef float64_t int_exp_norm(float64_t c, float64_t m, float64_t S):
    r"""
    Computes integrals of the form:

    .. math ::

        \int \exp(cx) \mathcal{N}(x \big\vert m, S)\ \mathrm{d}x = \exp\left(cm + \frac{1}{2} c^2 S\right)

    Parameters
    ----------
    c : float64_t
        constant :math:`c`
    m : float64_t
        mean parameter
    S : float64_t
        variance parameter

    Returns
    -------
    out : value of the integral
    
    """
    cdef float64_t out = (c * m) + (0.5 * c ** 2 * S)
    if out > MAX:
        out = INFINITY
    else:
        out = exp(out)
    return out


cpdef int int_K(float64_t[::1] out, float64_t[::1, :] x, float64_t h, float64_t[::1] w, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int K(x^\prime, x) \mathcal{N}(x^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x^\prime

    where :math:`K` is a Gaussian kernel matrix parameterized by
    :math:`h` and :math:`w`.

    The result is:
    
    .. math ::

        \mathrm{out}_i = h^2 \mathcal{N}(x_i \big\vert \mu, Iw + \Sigma)

    Parameters
    ----------
    out : float64_t[::1]
        :math:`n` output vector
    x : float64_t[::1, :]
        :math:`d\times n` input vector
    h : float64_t
        output scale kernel parameter
    w : float64_t[::1]
        :math:`d` vector of lengthscales
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int d = x.shape[0]
    cdef int n = x.shape[1]

    cdef float64_t[::1, :] W = empty((d, d), dtype=float64, order='F')

    cdef float64_t h_2 = h ** 2
    cdef float64_t logdet
    cdef int i, j

    if out.shape[0] != n:
        la.value_error("out has invalid shape")
    if w.shape[0] != d:
        la.value_error("w has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W[i, j] = cov[i, j] + w[i] ** 2
            else:
                W[i, j] = cov[i, j]

    la.cho_factor(W, W)
    logdet = la.logdet(W)

    for i in xrange(n):
        out[i] = h_2 * exp(mvn_logpdf(x[:, i], mu, W, logdet))

    return 0


cpdef int approx_int_K(float64_t[::1] out, float64_t[::1, :] xo, float64_t[::1, :] Kxxo, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Approximates integrals of the form:

    .. math ::

        \int K(x^\prime, x) \mathcal{N}(x^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x^\prime

    where :math:`K` is a kernel.

    Parameters
    ----------
    out : float64_t[::1]
        :math:`m` output vector
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    Kxxo : float64_t[::1, :]
        :math:`n\times m` kernel matrix
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int m = Kxxo.shape[0]
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t logdet, Kp1, Kp2
    cdef int i, j

    if out.shape[0] != m:
        la.value_error("out has invalid shape")
    if Kxxo.shape[1] != n:
        la.value_error("Kxxo has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # compute approximate integral with trapezoidal rule
    for i in xrange(m):
        out[i] = 0
        for j in xrange(n-1):
            Kp1 = Kxxo[i, j] * p_xo[j]
            Kp2 = Kxxo[i, j+1] * p_xo[j+1]
            out[i] += diff[j] * (Kp1 + Kp2) / 2.0

    return 0


cpdef int int_K1_K2(float64_t[::1, :] out, float64_t[::1, :] x1, float64_t[::1, :] x2, float64_t h1, float64_t[::1] w1, float64_t h2, float64_t[::1] w2, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int K_1(x_1, x^\prime) K_2(x^\prime, x_2) \mathcal{N}(x^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x^\prime

    where :math:`K_1` is a Gaussian kernel matrix parameterized by
    :math:`h_1` and :math:`w_1`, and :math:`K_2` is a Gaussian kernel
    matrix parameterized by :math:`h_2` and :math:`w_2`.

    The result is:
    
    .. math ::

        \mathrm{out}_{i,j} = h_1^2 h_2^2 \mathcal{N}([x_{1,i}, x_{2,j}] \big\vert [\mu, \mu], [Iw_1 + \Sigma, \Sigma; \Sigma, Iw_2 + \Sigma])

    Parameters
    ----------
    out : float64_t[::1, :]
        :math:`n_1\times n_2` output matrix
    x1 : float64_t[::1, :]
        :math:`d\times n_1` input vector
    x2 : float64_t[::1, :]
        :math:`d\times n_2` input vector
    h1 : float64_t
        output scale kernel parameter for :math:`K_1`
    w1 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_1`
    h2 : float64_t
        output scale kernel parameter for :math:`K_2`
    w2 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """
    
    cdef int n1 = x1.shape[1]
    cdef int n2 = x2.shape[1]
    cdef int d = x1.shape[0]

    cdef float64_t[::1, :, :] x = empty((2 * d, n1, n2), dtype=float64, order='F')
    cdef float64_t[::1] m = empty(2 * d, dtype=float64, order='F')
    cdef float64_t[::1, :] C = empty((2 * d, 2 * d), dtype=float64, order='F')

    cdef float64_t h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)
    cdef float64_t logdet
    cdef int i, j, k

    if out.shape[0] != n1 or out.shape[1] != n2:
        la.value_error("out has invalid shape")
    if x2.shape[0] != d:
        la.value_error("x2 has invalid shape")
    if w1.shape[0] != d:
        la.value_error("w1 has invalid shape")
    if w2.shape[0] != d:
        la.value_error("w2 has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

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
                x[k, i, j] = x1[k, i]
                x[k + d, i, j] = x2[k, j]

    # compute cholesky factor and determinant
    la.cho_factor(C, C)
    logdet = la.logdet(C)

    # compute pdf
    for i in xrange(n1):
        for j in xrange(n2):
            out[i, j] = h1_2_h2_2 * exp(mvn_logpdf(x[:, i, j], m, C, logdet))

    return 0


cpdef int approx_int_K1_K2(float64_t[::1, :] out, float64_t[::1, :] xo, float64_t[::1, :] K1xxo, float64_t[::1, :] K2xxo, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Approximates integrals of the form:

    .. math ::

        \int K_1(x_1, x^\prime) K_2(x^\prime, x_2) \mathcal{N}(x^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x^\prime

    where :math:`K_1` and :math:`K_2` are kernels.

    Parameters
    ----------
    out : float64_t[::1, :]
        :math:`m_1\times m_2` output matrix
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    K1xxo : float64_t[::1, :]
        :math:`m_1\times n` matrix produced by :math:`K_1`
    K2xxo : float64_t[::1, :]
        :math:`m_2\times n` matrix produced by :math:`K_2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int m1 = K1xxo.shape[0]
    cdef int m2 = K2xxo.shape[0]
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t logdet, Kp1, Kp2
    cdef int i, j

    if out.shape[0] != m1 or out.shape[1] != m2:
        la.value_error("out has invalid shape")
    if K1xxo.shape[1] != n:
        la.value_error("K1xxo has invalid shape")
    if K2xxo.shape[1] != n:
        la.value_error("K2xxo has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # compute approximate integral with trapezoidal rule
    for i1 in xrange(m1):
        for i2 in xrange(m2):
            out[i1, i2] = 0
            for j in xrange(n-1):
                Kp1 = K1xxo[i1, j] * K2xxo[i2, j] * p_xo[j]
                Kp2 = K1xxo[i1, j+1] * K2xxo[i2, j] * p_xo[j+1]
                out[i1, i2] += diff[j] * (Kp1 + Kp2) / 2.0

    return 0


cpdef int int_int_K1_K2_K1(float64_t[::1, :] out, float64_t[::1, :] x, float64_t h1, float64_t[::1] w1, float64_t h2, float64_t[::1] w2, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int \int K_1(x, x_1^\prime) K_2(x_1^\prime, x_2^\prime) K_1(x_2^\prime, x) N(x_1^\prime | \mu, \Sigma) N(x_2^\prime | \mu, \Sigma) \ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` is a Gaussian kernel matrix parameterized by
    :math:`h_1` and :math:`w_1`, and :math:`K_2` is a Gaussian kernel
    matrix parameterized by :math:`h_2` and :math:`w_2`.

    The result is:
    
    .. math ::

        \mathrm{out}_{i,j} = h_1^4 h_2^2 \big\vert{}\Gamma\big\vert{}^{-1} \mathcal{N}(x_i \big\vert{} \mu, Iw_1 + \Sigma) \mathcal{N}(x_j \big\vert{} \mu, Iw_1 + \Sigma) \mathcal{N}(\Gamma x_i \big\vert{} \Gamma x_j, Iw_2 + 2\Sigma - 2\Gamma\Sigma)

    where :math:`\Gamma = \Sigma(Iw_1 + \Sigma)^{-1}`.

    Parameters
    ----------
    out : float64_t[::1, :]
        :math:`n\times n` output matrix
    x : float64_t[::1, :]
        :math:`d\times n` input vector
    h1 : float64_t
        output scale kernel parameter for :math:`K_1`
    w1 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_1`
    h2 : float64_t
        output scale kernel parameter for :math:`K_2`
    w2 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int n = x.shape[1]
    cdef int d = x.shape[0]

    cdef float64_t[::1, :] W1_cov = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1, :] A = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1, :] B = empty((d, n), dtype=float64, order='F')
    cdef float64_t[::1, :] C = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] N1 = empty(n, dtype=float64)
    cdef float64_t[::1, :] N2 = empty((n, n), dtype=float64, order='F')
    cdef float64_t[::1] buf = empty(d, dtype=float64)

    cdef float64_t h1_4_h2_2 = (h1 ** 4) * (h2 ** 2)
    cdef logdet
    cdef int i, j

    if out.shape[0] != n or out.shape[1] != n:
        la.value_error("out has invalid shape")
    if w1.shape[0] != d:
        la.value_error("w1 has invalid shape")
    if w2.shape[0] != d:
        la.value_error("w2 has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    # compute W1 + cov
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W1_cov[i, j] = cov[i, j] + w1[i] ** 2
            else:
                W1_cov[i, j] = cov[i, j]

    # compute A = cov*(W1 + cov)^-1*cov
    la.cho_factor(W1_cov, L)
    logdet = la.logdet(L)
    la.cho_solve_mat(L, cov, W1_cov)
    la.dot22(cov, W1_cov, A)

    # compute B = cov*(W1 + cov)^-1 * x
    for i in xrange(n):
        la.cho_solve_vec(L, x[:, i], buf)
        la.dot21(cov, buf, B[:, i])

    # compute N1 = N(x | mu, W1 + cov)
    for i in xrange(n):
        N1[i] = mvn_logpdf(x[:, i], mu, L, logdet)

    # compute C = W2 + 2*cov - 2*cov*(W1 + cov)^-1*cov
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w2[i] ** 2 + 2*cov[i, j] - 2*A[i, j]
            else:
                C[i, j] = 2*cov[i, j] - 2*A[i, j]

    # compute N2 = N(cov*(W1 + cov)^-1 * x_i | cov*(W1 + cov)^-1 * x_j, C)
    la.cho_factor(C, L)
    logdet = la.logdet(L)
    for i in xrange(n):
        for j in xrange(n):
            N2[i, j] = mvn_logpdf(B[:, i], B[:, j], L, logdet)

    # put it all together
    for i in xrange(n):
        for j in xrange(n):
            out[i, j] = h1_4_h2_2 * exp(N1[i] + N1[j] + N2[i, j])

    return 0


cpdef int approx_int_int_K1_K2_K1(float64_t[::1, :] out, float64_t[::1, :] xo, float64_t[::1, :] K1xxo, float64_t[::1, :] K2xoxo, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int \int K_1(x, x_1^\prime) K_2(x_1^\prime, x_2^\prime) K_1(x_2^\prime, x) N(x_1^\prime | \mu, \Sigma) N(x_2^\prime | \mu, \Sigma) \ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` and :math:`K_2` are kernels.

    Parameters
    ----------
    out : float64_t[::1, :]
        :math:`m\times m` output matrix
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    K1xxo : float64_t[::1, 1]
        :math:`m\times n` matrix produced by :math:`K1`
    K2xoxo : float64_t[::1, 1]
        :math:`n\times n` matrix produced by :math:`K2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int m = K1xxo.shape[0]
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t[::1, :] buf = empty((m, n), dtype=float64, order='F')
    cdef float64_t logdet, Kp1, Kp2
    cdef int i1, i2, j1, j2

    if out.shape[0] != m or out.shape[1] != m:
        la.value_error("out has invalid shape")
    if K1xxo.shape[1] != n:
        la.value_error("K1xxo has invalid shape")
    if K2xoxo.shape[0] != n or K2xoxo.shape[1] != n:
        la.value_error("K2xoxo has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i1 in xrange(n):
        p_xo[i1] = exp(mvn_logpdf(xo[:, i1], mu, L, logdet))

    for i1 in xrange(n-1):
        diff[i1] = la.vecdiff(xo[:, i1+1], xo[:, i1])

    # inner integral
    for i2 in xrange(m):
        for j1 in xrange(n):
            buf[i2, j1] = 0
            for j2 in xrange(n-1):
                Kp1 = K2xoxo[j1, j2] * K1xxo[i2, j2] * p_xo[j2]
                Kp2 = K2xoxo[j1, j2+1] * K1xxo[i2, j2+1] * p_xo[j2+1]
                buf[i2, j1] += diff[j2] * (Kp1 + Kp2) / 2.0

    # outer integral
    for i1 in xrange(m):
        for i2 in xrange(m):
            out[i1, i2] = 0
            for j1 in xrange(n-1):
                Kp1 = buf[i2, j1] * K1xxo[i1, j1] * p_xo[j1]
                Kp2 = buf[i2, j1+1] * K1xxo[i1, j1+1] * p_xo[j1+1]
                out[i1, i2] += diff[j1] * (Kp1 + Kp2) / 2.0

    return 0


cpdef int int_int_K1_K2(float64_t[::1] out, float64_t[::1, :] x, float64_t h1, float64_t[::1] w1, float64_t h2, float64_t[::1] w2, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int \int K_1(x_2^\prime, x_1^\prime) K_2(x_1^\prime, x) \mathcal{N}(x_1^\prime \big\vert \mu, \Sigma) N(x_2^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` is a Gaussian kernel matrix parameterized by
    :math:`h_1` and :math:`w_1`, and :math:`K_2` is a Gaussian kernel
    matrix parameterized by :math:`h_2` and :math:`w_2`.

    The result is:
    
    .. math ::

        \mathrm{out}_i = h_1^2 h_2^2 \mathcal{N}(0 \big\vert 0, Iw_1 + 2\Sigma) \mathcal{N}(x_i \big\vert \mu, Iw_2 + \Sigma - \Sigma(Iw_1 + 2\Sigma)^{-1}\Sigma)

    Parameters
    ----------
    out : float64_t[::1]
        :math:`n` output vector
    x : float64_t[::1, :]
        :math:`d\times n` input vector
    h1 : float64_t
        output scale kernel parameter for :math:`K_1`
    w1 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_1`
    h2 : float64_t
        output scale kernel parameter for :math:`K_2`
    w2 : float64_t[::1]
        :math:`d` vector of lengthscales for :math:`K_2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int n = x.shape[1]
    cdef int d = x.shape[0]

    cdef float64_t[::1, :] W1_2cov = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1, :] C = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1, :] buf = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] z = zeros(d, dtype=float64, order='F')

    cdef float64_t h1_2_h2_2 = (h1 ** 2) * (h2 ** 2)
    cdef float64_t N, logdet
    cdef int i, j

    if out.shape[0] != n:
        la.value_error("out has invalid shape")
    if w1.shape[0] != d:
        la.value_error("w1 has invalid shape")
    if w2.shape[0] != d:
        la.value_error("w2 has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    # compute W1 + 2*cov
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W1_2cov[i, j] = 2*cov[i, j] + w1[i] ** 2
            else:
                W1_2cov[i, j] = 2*cov[i, j]

    # compute N(0 | 0, W1 + 2*cov)
    la.cho_factor(W1_2cov, W1_2cov)
    N = mvn_logpdf(z, z, W1_2cov, la.logdet(W1_2cov))

    # compute C = cov*(W1 + 2*cov)^-1*cov
    la.cho_solve_mat(W1_2cov, cov, buf)
    la.dot22(cov, buf, C)

    # compute W2 + cov - cov*(W1 + 2*cov)^-1*cov
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                C[i, j] = w2[i] ** 2 + cov[i, j] - C[i, j]
            else:
                C[i, j] = cov[i, j] - C[i, j]

    # compute N(x | mu, W2 + cov - cov*(W1 + 2*cov)^-1*cov)
    la.cho_factor(C, C)
    logdet = la.logdet(C)
    for i in xrange(n):
        out[i] = h1_2_h2_2 * exp(N + mvn_logpdf(x[:, i], mu, C, logdet))

    return 0


cpdef int approx_int_int_K1_K2(float64_t[::1] out, float64_t[::1, :] xo, float64_t[::1, :] K1xoxo, float64_t[::1, :] K2xxo, float64_t[::1] mu, float64_t[::1, :] cov) except -1:
    r"""
    Approximates integrals of the form:

    .. math ::

        \int \int K_1(x_2^\prime, x_1^\prime) K_2(x_1^\prime, x) \mathcal{N}(x_1^\prime \big\vert \mu, \Sigma) N(x_2^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` and :math:`K_2` are kernels.

    Parameters
    ----------
    out : float64_t[::1]
        :math:`m` output vector
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locatinos
    K1xoxo : float64_t[::1, :]
        :math:`n\times n` matrix produced by :math:`K_1`
    K2xxo : float64_t[::1, :]
        :math:`m\times n` matrix produced by :math:`K_2`
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int m = K2xxo.shape[0]
    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t[::1] buf = empty(n, dtype=float64, order='F')
    cdef float64_t logdet, Kp1, Kp2
    cdef int i, j1, j2

    if out.shape[0] != m:
        la.value_error("out has invalid shape")
    if K1xoxo.shape[0] != n or K1xoxo.shape[1] != n:
        la.value_error("K1xoxo has invalid shape")
    if K2xxo.shape[1] != n:
        la.value_error("K2xxo has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # inner integral
    for i in xrange(m):
        for j1 in xrange(n):
            buf[j1] = 0
            for j2 in xrange(n-1):
                Kp1 = K1xoxo[j1, j2] * K2xxo[i, j2] * p_xo[j2]
                Kp2 = K1xoxo[j1, j2+1] * K2xxo[i, j2+1] * p_xo[j2+1]
                buf[j1] += diff[j2] * (Kp1 + Kp2) / 2.0

        out[i] = 0
        for j1 in xrange(n-1):
            Kp1 = buf[j1] * p_xo[j1]
            Kp2 = buf[j1+1] * p_xo[j1+1]
            out[i] += diff[j1] * (Kp1 + Kp2) / 2.0

    return 0


cpdef float64_t int_int_K(int32_t d, float64_t h, float64_t[::1] w, float64_t[::1] mu, float64_t[::1, :] cov) except? -1:
    r"""
    Computes integrals of the form:

    .. math ::

        \int \int K(x_1^\prime, x_2^\prime) \mathcal{N}(x_1^\prime \big\vert \mu, \Sigma) \mathcal{N}(x_2^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` is a Gaussian kernel matrix parameterized by
    :math:`h_1` and :math:`w_1`, and :math:`K_2` is a Gaussian kernel
    matrix parameterized by :math:`h_2` and :math:`w_2`.

    The result is:
    
    .. math ::

        \mathrm{out} = h^2 \mathcal{N}(0 \big\vert 0, Iw + 2\Sigma)

    Parameters
    ----------
    d : int32_t
        dimensionality of the Gaussians
    h : float64_t
        output scale kernel parameter
    w : float64_t[::1]
        :math:`d` vector of lengthscales
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : value of the integral

    """

    cdef float64_t[::1, :] W_2cov = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] z = zeros(d, dtype=float64)
    cdef int i, j

    if w.shape[0] != d:
        la.value_error("w has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    # compute W + 2*cov
    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                W_2cov[i, j] = 2*cov[i, j] + w[i] ** 2
            else:
                W_2cov[i, j] = 2*cov[i, j]

    # compute N(0 | 0, W1 + 2*cov)
    la.cho_factor(W_2cov, W_2cov)

    return (h ** 2) * exp(mvn_logpdf(z, z, W_2cov, la.logdet(W_2cov)))


cpdef float64_t approx_int_int_K(float64_t[::1, :] xo, float64_t[::1, :] Kxoxo, float64_t[::1] mu, float64_t[::1, :] cov) except? -1:
    r"""
    Approximates integrals of the form:

    .. math ::

        \int \int K(x_1^\prime, x_2^\prime) \mathcal{N}(x_1^\prime \big\vert \mu, \Sigma) \mathcal{N}(x_2^\prime \big\vert \mu, \Sigma)\ \mathrm{d}x_1^\prime \ \mathrm{d}x_2^\prime

    where :math:`K_1` and :math:`K_2` are kernels.

    Parameters
    ----------
    xo : float64_t[::1, :]
        :math:`d\times n` vector of approximation locations
    Kxoxo : float64_t[::1, :]
        :math:`n\times n` kernel matrix
    mu : float64_t[::1]
        :math:`d` mean
    cov : float64_t[::1, :]
        :math:`d\times d` covariance

    Returns
    -------
    out : value of the integral

    """

    cdef int d = xo.shape[0]
    cdef int n = xo.shape[1]

    cdef float64_t[::1, :] L = empty((d, d), dtype=float64, order='F')
    cdef float64_t[::1] p_xo = empty(n, dtype=float64)
    cdef float64_t[::1] diff = empty(n-1, dtype=float64)
    cdef float64_t[::1] buf = empty(n, dtype=float64, order='F')
    cdef float64_t logdet, Kp1, Kp2
    cdef int i, j

    if Kxoxo.shape[0] != n or Kxoxo.shape[1] != n:
        la.value_error("Kxoxo has invalid shape")
    if mu.shape[0] != d:
        la.value_error("mu has invalid shape")
    if cov.shape[0] != d or cov.shape[1] != d:
        la.value_error("cov has invalid shape")

    la.cho_factor(cov, L)
    logdet = la.logdet(L)

    for i in xrange(n):
        p_xo[i] = exp(mvn_logpdf(xo[:, i], mu, L, logdet))

    for i in xrange(n-1):
        diff[i] = la.vecdiff(xo[:, i+1], xo[:, i])

    # inner integral
    for i in xrange(n):
        buf[i] = 0
        for j in xrange(n-1):
            Kp1 = Kxoxo[i, j] * p_xo[j]
            Kp2 = Kxoxo[i, j+1] * p_xo[j+1]
            buf[i] += diff[j] * (Kp1 + Kp2) / 2.0

    out = 0
    for i in xrange(n-1):
        Kp1 = buf[i] * p_xo[i]
        Kp2 = buf[i+1] * p_xo[i+1]
        out += diff[i] * (Kp1 + Kp2) / 2.0

    return out
