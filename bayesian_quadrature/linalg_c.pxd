from numpy cimport float64_t, int32_t

cdef char UPLO

cdef void value_error(str msg) except *
cdef void linalg_error(str msg) except *

cpdef int cho_factor(float64_t[::1, :] C, float64_t[::1, :] L) except -1
cpdef int cho_solve_vec(float64_t[::1, :] L, float64_t[::1] b, float64_t[::1] x) except -1
cpdef int cho_solve_mat(float64_t[::1, :] L, float64_t[::1, :] b, float64_t[::1, :] x) except -1
cpdef float64_t logdet(float64_t[::1, :] A) except? -1
cpdef float64_t dot11(float64_t[::1] x, float64_t[::1] y) except? -1
cpdef int dot12(float64_t[::1] x, float64_t[::1, :] Y, float64_t[::1] xY) except -1
cpdef int dot21(float64_t[::1, :] X, float64_t[::1] y, float64_t[::1] Xy) except -1
cpdef int dot22(float64_t[::1, :] X, float64_t[::1, :] Y, float64_t[::1, :] XY) except -1
cpdef float64_t vecdiff(float64_t[::1] x, float64_t[::1] y) except? -1
