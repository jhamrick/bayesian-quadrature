from numpy cimport float64_t, int32_t

cdef char UPLO

cdef void value_error(str msg)
cdef void linalg_error(str msg)

cpdef cho_factor(float64_t[::1, :] C, float64_t[::1, :] L)
cpdef cho_solve_vec(float64_t[::1, :] L, float64_t[::1] b, float64_t[::1] x)
cpdef cho_solve_mat(float64_t[::1, :] L, float64_t[::1, :] b, float64_t[::1, :] x)
cpdef float64_t logdet(float64_t[::1, :] A)
cpdef float64_t dot11(float64_t[::1] x, float64_t[::1] y)
cpdef dot12(float64_t[::1] x, float64_t[::1, :] Y, float64_t[::1] xY)
cpdef dot21(float64_t[::1, :] X, float64_t[::1] y, float64_t[::1] Xy)
cpdef dot22(float64_t[::1, :] X, float64_t[::1, :] Y, float64_t[::1, :] XY)
cpdef float64_t vecdiff(float64_t[::1] x, float64_t[::1] y)
