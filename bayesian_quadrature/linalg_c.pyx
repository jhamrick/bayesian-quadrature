# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False

from numpy.linalg import LinAlgError
from numpy import array, empty
from numpy import float64, int32

######################################################################

from numpy cimport ndarray, float64_t, int32_t
from libc.math cimport log

cdef extern from "cblas.h":
    float64_t cblas_ddot(int32_t N, float64_t *X, int32_t incX, float64_t *Y, int32_t incY)

cdef extern from "clapack.h":
    int32_t dpotrf_(char *uplo, int32_t *n, float64_t *a, int32_t *lda, int32_t *info)
    int32_t dpotrs_(char *uplo, int32_t *n, int32_t *nrhs, float64_t *a, int32_t *lda, float64_t *b, int32_t *ldb, int32_t *info)
    int32_t dgetrf_(int32_t *m, int32_t *n, float64_t *a, int32_t *lda, int32_t *ipiv, int32_t *info)

######################################################################

cdef char UPLO = 'L'

######################################################################

cdef void value_error(str msg):
    raise ValueError(msg)

cdef void linalg_error(str msg):
    raise LinAlgError(msg)

cdef void cho_factor(ndarray[float64_t, mode='fortran', ndim=2] C, ndarray[float64_t, mode='fortran', ndim=2] L):
    cdef int32_t n = C.shape[0]
    cdef int32_t info
    cdef int i, j

    if C.shape[1] != n:
        value_error("C is not square")
    if L.shape[0] != n or L.shape[1] != n:
        value_error("invalid shape for L")

    for i in xrange(n):
        for j in xrange(n):
            L[i, j] = C[i, j]

    dpotrf_(&UPLO, &n, &L[0, 0], &n, &info)

    if info < 0:
        value_error("illegal value")
    elif info > 0:
        linalg_error("matrix is not positive definite")


cdef void cho_solve_vec(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=1] b, ndarray[float64_t, mode='fortran', ndim=1] x):
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

    for i in xrange(n):
        x[i] = b[i]

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &x[0], &n, &info)

    if info < 0:
        value_error("illegal value")


cdef void cho_solve_mat(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=2] b, ndarray[float64_t, mode='fortran', ndim=2] x):
    cdef int32_t n = L.shape[0]
    cdef int32_t nrhs = 2
    cdef int32_t info
    cdef int i, j

    if L.shape[1] != n:
        value_error("L is not square")
    if b.shape[0] != n or b.shape[1] != n:
        value_error("b has invalid shape")
    if x.shape[0] != n or x.shape[1] != n:
        value_error("x has invalid shape")

    for i in xrange(n):
        for j in xrange(n):
            x[i, j] = b[i, j]

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &x[0, 0], &n, &info)

    if info < 0:
        value_error("illegal value")


cdef float64_t logdet(ndarray[float64_t, mode='c', ndim=2] A):
    cdef ndarray[float64_t, mode='fortran', ndim=2] LU
    cdef ndarray[int32_t, mode='fortran', ndim=1] P
    cdef int32_t n = A.shape[0]
    cdef int32_t info
    cdef float64_t logdet
    cdef int i

    if A.shape[1] != n:
        value_error("A is not square")

    LU = array(A, dtype=float64, copy=True, order='F')
    P = empty(n, dtype=int32, order='F')

    dgetrf_(&n, &n, &LU[0, 0], &n, &P[0], &info)

    if info < 0:
        value_error("illegal value")
    elif info > 0:
        linalg_error("matrix is singular")
    
    logdet = 0
    for i in xrange(n):
        logdet += log(LU[i, i])

    return logdet


cdef float64_t dot11(ndarray[float64_t, mode='c', ndim=1] x, ndarray[float64_t, mode='c', ndim=1] y):
    cdef float64_t out
    cdef int32_t n = x.shape[0]

    if y.shape[0] != n:
        value_error("shape mismatch")

    out = cblas_ddot(n, &x[0], 1, &y[0], 1)
    return out


cdef ndarray[float64_t, mode='c', ndim=1] dot12(ndarray[float64_t, mode='c', ndim=1] x, ndarray[float64_t, mode='c', ndim=2] y):
    cdef ndarray[float64_t, mode='c', ndim=1] out
    cdef int32_t n = x.shape[0]
    cdef int32_t p = y.shape[1]
    cdef int j

    if y.shape[0] != n:
        value_error("shape mismatch")

    out = empty(p, dtype=float64)
    for j in xrange(p):
        out[j] = cblas_ddot(n, &x[0], 1, &y[0, j], p)

    return out


cdef ndarray[float64_t, mode='c', ndim=1] dot21(ndarray[float64_t, mode='c', ndim=2] x, ndarray[float64_t, mode='c', ndim=1] y):
    cdef ndarray[float64_t, mode='c', ndim=1] out
    cdef int32_t m = x.shape[0]
    cdef int32_t n = x.shape[1]
    cdef int i

    if y.shape[0] != n:
        value_error("shape mismatch")

    out = empty(m, dtype=float64)
    for i in xrange(m):
        out[i] = cblas_ddot(n, &x[i, 0], 1, &y[0], 1)

    return out


cdef ndarray[float64_t, mode='c', ndim=2] dot22(ndarray[float64_t, mode='c', ndim=2] x, ndarray[float64_t, mode='c', ndim=2] y):
    cdef ndarray[float64_t, mode='c', ndim=2] out
    cdef int32_t m = x.shape[0]
    cdef int32_t n = x.shape[1]
    cdef int32_t p = y.shape[1]
    cdef int i, j

    if y.shape[0] != n:
        value_error("shape mismatch")

    out = empty((m, p), dtype=float64)
    for i in xrange(m):
        for j in xrange(p):
            out[i, j] = cblas_ddot(n, &x[i, 0], 1, &y[0, j], p)

    return out
