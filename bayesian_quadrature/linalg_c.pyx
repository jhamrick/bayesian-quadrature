# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True

from numpy.linalg import LinAlgError
from numpy import array, empty
from numpy import float64, int32

######################################################################

from numpy cimport float64_t, int32_t
from libc.math cimport log, sqrt, fabs

cdef extern from "cblas.h":
    float64_t cblas_ddot(int32_t N, float64_t *X, int32_t incX, float64_t *Y, int32_t incY)

    enum CBLAS_ORDER:
        CblasRowMajor = 101
        CblasColMajor = 102

    enum CBLAS_UPLO:
        CblasUpper = 121
        CblasLower = 122

IF UNAME_SYSNAME == "Darwin":
    cdef extern from "clapack.h":
        int32_t dpotrf_(char *uplo, int32_t *n, float64_t *a, int32_t *lda, int32_t *info)
        int32_t dpotrs_(char *uplo, int32_t *n, int32_t *nrhs, float64_t *a, int32_t *lda, float64_t *b, int32_t *ldb, int32_t *info)

    cdef char UPLO = 'L'

    cdef int32_t clapack_dpotrf(int32_t Order, int32_t Uplo, int32_t N, float64_t *A, int32_t lda):
        cdef int32_t info
        dpotrf_(&UPLO, &N, A, &lda, &info)
        return info

    cdef int32_t clapack_dpotrs(int32_t Order, int32_t Uplo, int32_t N, int32_t NRHS, float64_t *A, int32_t lda, float64_t *B, int32_t ldb):
        cdef int32_t info
        dpotrs_(&UPLO, &N, &NRHS, A, &lda, B, &ldb, &info)
        return info

ELIF UNAME_SYSNAME == "Linux":
    cdef extern from "clapack.h":
        int32_t clapack_dpotrf(int32_t Order, int32_t Uplo, int32_t N, float64_t *A, int32_t lda)
        int32_t clapack_dpotrs(int32_t Order, int32_t Uplo, int32_t N, int32_t NRHS, float64_t *A, int32_t lda, float64_t *B, int32_t ldb)

######################################################################

cdef void value_error(str msg) except *:
    raise ValueError(msg)

cdef void linalg_error(str msg) except *:
    raise LinAlgError(msg)

cpdef int cho_factor(float64_t[::1, :] C, float64_t[::1, :] L) except -1:
    r"""
    Compute the Cholesky factorization of matrix :math:`C` and store
    the result in :math:`L`. The lower part will contain the
    factorization; upper values could be anything.

    Parameters
    ----------
    C : float64_t[::1, :]
        :math:`n\times n` input matrix
    L : float64_t[::1, :]
        :math:`n\times n` output matrix

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t n = C.shape[0]
    cdef int32_t info
    cdef int i, j

    if C.shape[1] != n:
        value_error("C is not square")
    if L.shape[0] != n or L.shape[1] != n:
        value_error("invalid shape for L")

    if &C[0, 0] != &L[0, 0]:
        L[:, :] = C[:, :]

    info = clapack_dpotrf(CblasColMajor, CblasLower, n, &L[0, 0], n)

    if info < 0:
        value_error("illegal value")
    elif info > 0:
        linalg_error("matrix is not positive definite")

    return 0


cpdef int cho_solve_vec(float64_t[::1, :] L, float64_t[::1] b, float64_t[::1] x) except -1:
    r"""
    Solve the equation :math:`Ax=b`, where :math:`A` is a matrix and
    :math:`x` and :math:`b` are vectors.

    Parameters
    ----------
    L : float64_t[::1, :]
        :math:`n\times n` lower-triangular Cholesky factor of :math:`A`
    b : float64_t[::1]
        :math:`n` input vector
    x : float64_t[::1]
        :math:`n` output vector

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t n = L.shape[0]
    cdef int32_t nrhs = 1
    cdef int32_t info
    cdef int i

    if L.shape[1] != n:
        value_error("L is not square")
    if b.shape[0] != n:
        value_error("b has invalid size")
    if x.shape[0] != n:
        value_error("x has invalid size")

    if &x[0] != &b[0]:
        x[:] = b[:]

    info = clapack_dpotrs(CblasColMajor, CblasLower, n, nrhs, &L[0, 0], n, &x[0], n);

    if info < 0:
        value_error("illegal value")

    return 0


cpdef int cho_solve_mat(float64_t[::1, :] L, float64_t[::1, :] B, float64_t[::1, :] X) except -1:
    r"""
    Solve the equation :math:`AX=B`, where :math:`A`, :math:`X`, and
    :math:`B` are matrices.

    Parameters
    ----------
    L : float64_t[::1, :]
        :math:`n\times n` lower-triangular Cholesky factor of :math:`A`
    B : float64_t[::1, :]
        :math:`n\times n` input matrix
    X : float64_t[::1, :]
        :math:`n\times n` output matrix

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t n = L.shape[0]
    cdef int32_t nrhs = B.shape[1]
    cdef int32_t info
    cdef int i, j

    if L.shape[1] != n:
        value_error("L is not square")
    if B.shape[0] != n or B.shape[1] != n:
        value_error("B has invalid shape")
    if X.shape[0] != n or X.shape[1] != n:
        value_error("X has invalid shape")

    if &X[0, 0] != &B[0, 0]:
        X[:, :] = B[:, :]

    info = clapack_dpotrs(CblasColMajor, CblasLower, n, nrhs, &L[0, 0], n, &X[0, 0], n);

    if info < 0:
        value_error("illegal value")

    return 0


cpdef float64_t logdet(float64_t[::1, :] L) except? -1:
    r"""
    Compute the log determinant of a matrix :math:`A` from its
    lower-triangular Cholesky factor :math:`L`.

    Parameters
    ----------
    L : float64_t[::1, :]
        :math:`n\times n` lower-triangular Cholesky factor of :math:`A`

    Returns
    -------
    out : log-determinant

    """

    cdef int32_t n = L.shape[0]
    cdef float64_t logdet
    cdef int i

    if L.shape[1] != n:
        value_error("L is not square")

    logdet = 0
    for i in xrange(n):
        logdet += log(L[i, i])
    logdet *= 2

    return logdet


cpdef float64_t dot11(float64_t[::1] x, float64_t[::1] y) except? -1:
    r"""
    Compute the dot product between vectors :math:`x` and :math:`y`.

    Parameters
    ----------
    x : float64_t[::1]
        :math:`n` left-side vector
    y : float64_t[::1]
        :math:`n` right-side vector

    Returns
    -------
    out : dot product

    """

    cdef float64_t out
    cdef int32_t n = x.shape[0]

    if y.shape[0] != n:
        value_error("shape mismatch")

    if n == 1:
        out = x[0] * y[0]
    
    elif n == 2:
        out = (x[0] * y[0]) + (x[1] * y[1])

    else:
        out = cblas_ddot(n, &x[0], 1, &y[0], 1)

    return out


cpdef int dot12(float64_t[::1] x, float64_t[::1, :] Y, float64_t[::1] xY) except -1:
    r"""
    Compute the dot product between a vector :math:`x` and a matrix :math:`Y`.

    Parameters
    ----------
    x : float64_t[::1]
        :math:`n` left-side vector
    Y : float64_t[::1, :]
        :math:`n\times p` right-side matrix
    xY : float64_t[::1]
        :math:`p` output vector

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t n = x.shape[0]
    cdef int32_t p = Y.shape[1]
    cdef int j

    if Y.shape[0] != n:
        value_error("shape mismatch")
    if xY.shape[0] != p:
        value_error("shape mismatch")

    for j in xrange(p):
        if n == 1:
            xY[j] = x[0] * Y[0, j]

        elif n == 2:
            xY[j] = (x[0] * Y[0, j]) + (x[1] * Y[1, j])

        else:
            xY[j] = cblas_ddot(n, &x[0], 1, &Y[0, j], 1)

    return 0


cpdef int dot21(float64_t[::1, :] X, float64_t[::1] y, float64_t[::1] Xy) except -1:
    r"""
    Compute the dot product between a matrix :math:`X` and a vector :math:`y`.

    Parameters
    ----------
    X : float64_t[::1, :]
        :math:`m\times n` left-side matrix
    y : float64_t[::1]
        :math:`n` right-side vector
    xY : float64_t[::1]
        :math:`m` output vector

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t m = X.shape[0]
    cdef int32_t n = X.shape[1]
    cdef int i

    if y.shape[0] != n:
        value_error("shape mismatch")
    if Xy.shape[0] != m:
        value_error("shape mismatch")

    for i in xrange(m):
        if n == 1:
            Xy[i] = X[i, 0] * y[0]

        elif n == 2:
            Xy[i] = (X[i, 0] * y[0]) + (X[i, 1] * y[1])

        else:
            Xy[i] = cblas_ddot(n, &X[i, 0], m, &y[0], 1)

    return 0


cpdef int dot22(float64_t[::1, :] X, float64_t[::1, :] Y, float64_t[::1, :] XY) except -1:
    r"""
    Compute the dot product between a matrix :math:`X` and a vector :math:`y`.

    Parameters
    ----------
    X : float64_t[::1, :]
        :math:`m\times n` left-side matrix
    Y : float64_t[::1, :]
        :math:`n\times p` right-side matrix
    XY : float64_t[::1, :]
        :math:`m\times p` output matrix

    Returns
    -------
    out : 0 on success, -1 on failure

    """

    cdef int32_t m = X.shape[0]
    cdef int32_t n = X.shape[1]
    cdef int32_t p = Y.shape[1]
    cdef int i, j

    if Y.shape[0] != n:
        value_error("shape mismatch")
    if XY.shape[0] != m or XY.shape[1] != p:
        value_error("shape mismatch")

    for i in xrange(m):
        for j in xrange(p):
            if n == 1:
                XY[i, j] = X[i, 0] * Y[0, j]

            elif n == 2:
                XY[i, j] = (X[i, 0] * Y[0, j]) + (X[i, 1] * Y[1, j])

            else:
                XY[i, j] = cblas_ddot(n, &X[i, 0], m, &Y[0, j], 1)

    return 0


cpdef float64_t vecdiff(float64_t[::1] x, float64_t[::1] y) except? -1:
    r"""
    Compute the Euclidean distance between two vectors :math:`x` and :math:`y`.

    Parameters
    ----------
    x : float64_t[::1]
        :math:`n` left-side vector
    y : float64_t[::1]
        :math:`n` right-side vector

    Returns
    -------
    out : Euclidean distance, :math:`\sqrt{\sum_i x_i^2 + y_i^2}`

    """

    cdef int n = x.shape[0]
    cdef float64_t diff = 0
    cdef int i

    if y.shape[0] != n:
        value_error("shape mismatch")

    if n == 1:
        diff = fabs(x[0] - y[0])

    elif n == 2:
        diff = sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    else:
        for i in xrange(n):
            diff += (x[i] - y[i]) ** 2
        diff = sqrt(diff)

    return diff
