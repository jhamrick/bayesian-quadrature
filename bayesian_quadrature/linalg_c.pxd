from numpy cimport ndarray, float64_t, int32_t

cdef char UPLO

cdef void cho_factor(ndarray[float64_t, mode='fortran', ndim=2] C, ndarray[float64_t, mode='fortran', ndim=2] L)
cdef void cho_solve_vec(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=1] b, ndarray[float64_t, mode='fortran', ndim=1] x)
cdef void cho_solve_mat(ndarray[float64_t, mode='fortran', ndim=2] L, ndarray[float64_t, mode='fortran', ndim=2] b, ndarray[float64_t, mode='fortran', ndim=2] x)
cdef float64_t logdet(ndarray[float64_t, mode='c', ndim=2] A)
cdef float64_t dot11(ndarray[float64_t, mode='c', ndim=1] x, ndarray[float64_t, mode='c', ndim=1] y)
cdef ndarray[float64_t, mode='c', ndim=1] dot12(ndarray[float64_t, mode='c', ndim=1] x, ndarray[float64_t, mode='c', ndim=2] y)
cdef ndarray[float64_t, mode='c', ndim=1] dot21(ndarray[float64_t, mode='c', ndim=2] x, ndarray[float64_t, mode='c', ndim=1] y)
cdef ndarray[float64_t, mode='c', ndim=2] dot22(ndarray[float64_t, mode='c', ndim=2] x, ndarray[float64_t, mode='c', ndim=2] y)
