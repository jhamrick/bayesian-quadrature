from numpy cimport ndarray, float64_t, int32_t

cpdef float64_t mvn_logpdf(float64_t[::1] x, float64_t[::1] m, float64_t[::1, :] L, float64_t logdet)

cpdef float64_t int_exp_norm(float64_t c, float64_t m, float64_t S)

cpdef int_K(float64_t[::1] out, float64_t[::1, :] x, float64_t h, float64_t[::1] w, float64_t[::1] mu, float64_t[::1, :] cov)
cpdef approx_int_K(float64_t[::1] out, float64_t[::1, :] xo, float64_t[::1, :] Kxxo, float64_t[::1] mu, float64_t[::1, :] cov)

cpdef int_K1_K2(float64_t[::1, :] out, float64_t[::1, :] x1, float64_t[::1, :] x2, float64_t h1, float64_t[::1] w1, float64_t h2, float64_t[::1] w2, float64_t[::1] mu, float64_t[::1, :] cov)
cpdef approx_int_K1_K2(float64_t[::1, :] out, float64_t[::1, :] xo, float64_t[::1, :] K1xxo, float64_t[::1, :] K2xxo, float64_t[::1] mu, float64_t[::1, :] cov)

cpdef int_int_K1_K2_K1(float64_t[::1, :] out, float64_t[::1, :] x, float64_t h1, float64_t[::1] w1, float64_t h2, float64_t[::1] w2, float64_t[::1] mu, float64_t[::1, :] cov)
cpdef approx_int_int_K1_K2_K1(float64_t[::1, :] out, float64_t[::1, :] xo, float64_t[::1, :] K1xxo, float64_t[::1, :] K2xoxo, float64_t[::1] mu, float64_t[::1, :] cov)

cpdef int_int_K1_K2(ndarray[float64_t, mode='fortran', ndim=1] out, ndarray[float64_t, mode='fortran', ndim=2] x, float64_t h1, ndarray[float64_t, mode='fortran', ndim=1] w1, float64_t h2, ndarray[float64_t, mode='fortran', ndim=1] w2, ndarray[float64_t, mode='fortran', ndim=1] mu, ndarray[float64_t, mode='fortran', ndim=2] cov)

cpdef float64_t int_int_K(int32_t d, float64_t h, ndarray[float64_t, mode='fortran', ndim=1] w, ndarray[float64_t, mode='fortran', ndim=1] mu, ndarray[float64_t, mode='fortran', ndim=2] cov)
