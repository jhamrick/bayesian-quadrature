from numpy cimport ndarray, float64_t, int32_t

cdef char UPLO

cdef void value_error(str msg)
cdef void linalg_error(str msg)

cpdef cho_factor(ndarray[float64_t, mode='fortran', ndim=2] C, ndarray[float64_t, mode='fortran', ndim=2] L)
cpdef cho_solve_vec(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=1] b, ndarray[float64_t, mode='fortran', ndim=1] x)
cpdef cho_solve_mat(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=2] b, ndarray[float64_t, mode='fortran', ndim=2] x)
cpdef float64_t logdet(ndarray[float64_t, mode='fortran', ndim=2] A)
cpdef float64_t dot11(ndarray[float64_t, mode='fortran', ndim=1] x, ndarray[float64_t, mode='fortran', ndim=1] y)
cpdef dot12(ndarray[float64_t, mode='fortran', ndim=1] x, ndarray[float64_t, mode='fortran', ndim=2] Y, ndarray[float64_t, mode='fortran', ndim=1] xY)
cpdef dot21(ndarray[float64_t, mode='fortran', ndim=2] X, ndarray[float64_t, mode='fortran', ndim=1] y, ndarray[float64_t, mode='fortran', ndim=1] Xy)
cpdef dot22(ndarray[float64_t, mode='fortran', ndim=2] X, ndarray[float64_t, mode='fortran', ndim=2] Y, ndarray[float64_t, mode='fortran', ndim=2] XY)
