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

cpdef cho_factor(ndarray[float64_t, mode='fortran', ndim=2] C, ndarray[float64_t, mode='fortran', ndim=2] L):
    cdef int32_t n = C.shape[0]
    cdef int32_t info
    cdef int i, j

    if C.shape[1] != n:
        value_error("C is not square")
    if L.shape[0] != n or L.shape[1] != n:
        value_error("invalid shape for L")

    if &C[0, 0] != &L[0, 0]:
        for i in xrange(n):
            for j in xrange(n):
                L[i, j] = C[i, j]

    dpotrf_(&UPLO, &n, &L[0, 0], &n, &info)

    if info < 0:
        value_error("illegal value")
    elif info > 0:
        linalg_error("matrix is not positive definite")

    return


cpdef cho_solve_vec(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=1] b, ndarray[float64_t, mode='fortran', ndim=1] x):
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
        for i in xrange(n):
            x[i] = b[i]

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &x[0], &n, &info)

    if info < 0:
        value_error("illegal value")


cpdef cho_solve_mat(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=2] B, ndarray[float64_t, mode='fortran', ndim=2] X):
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
        for i in xrange(n):
            for j in xrange(n):
                X[i, j] = B[i, j]

    dpotrs_(&UPLO, &n, &nrhs, &L[0, 0], &n, &X[0, 0], &n, &info)

    if info < 0:
        value_error("illegal value")

    return


cpdef float64_t logdet(ndarray[float64_t, mode='fortran', ndim=2] L):
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


cpdef float64_t dot11(ndarray[float64_t, mode='fortran', ndim=1] x, ndarray[float64_t, mode='fortran', ndim=1] y):
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


cpdef dot12(ndarray[float64_t, mode='fortran', ndim=1] x, ndarray[float64_t, mode='fortran', ndim=2] Y, ndarray[float64_t, mode='fortran', ndim=1] xY):
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

    return


cpdef dot21(ndarray[float64_t, mode='fortran', ndim=2] X, ndarray[float64_t, mode='fortran', ndim=1] y, ndarray[float64_t, mode='fortran', ndim=1] Xy):
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

    return


cpdef dot22(ndarray[float64_t, mode='fortran', ndim=2] X, ndarray[float64_t, mode='fortran', ndim=2] Y, ndarray[float64_t, mode='fortran', ndim=2] XY):
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

    return
