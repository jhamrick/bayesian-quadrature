import numpy as np
cimport numpy as np

from libc.math cimport exp, log
from libc.stdlib cimport rand, srand, RAND_MAX
from warnings import warn

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps
cdef DTYPE_t NAN = np.nan
cdef DTYPE_t FRAND_MAX = float(RAND_MAX)


cdef DTYPE_t uniform(DTYPE_t lo, DTYPE_t hi):
    return (rand() / FRAND_MAX) * (hi - lo) + lo


def slice_sample(np.ndarray[DTYPE_t, ndim=2] samples, logpdf, DTYPE_t xval, DTYPE_t w):
    cdef np.ndarray[DTYPE_t, ndim=1] yvals
    cdef DTYPE_t xpr, pr, yval, logyval, left, right
    cdef int pct, newpct, i, n

    # seed the random number generator
    srand(np.random.randint(0, RAND_MAX))

    # number of samples
    n = samples.shape[0]
    
    # keep track of progress
    pct = 0

    # sample random yvals ahead of time
    yvals = np.random.rand(n - 1)

    for i in xrange(0, n - 1):
        # compute the height of the pdf
        xpr = logpdf(samples[i])

        # sample a value somewhere in that range
        yval = uniform(0, exp(xpr))
        logyval = log(yval)

        # compute the initial window bounds
        left = samples[i] - w
        right = samples[i] + w

        # widen the bounds until they're outside the slice
        while logpdf(left) > logyval:
            left -= w
        while logpdf(right) > logyval:
            right += w

        # now sample a new x value
        while True:

            # check the window size to make sure it's not too small
            if (right - left) < 1e-9:
                raise RuntimeError("sampling window is too small")

            # choose the x and evaluate its probability
            samples[i + 1] = uniform(left, right)
            pr = logpdf(samples[i + 1])

            # if it is within the slice, then we're done
            if pr > logyval:
                break

            # otherwise, shrink the window bounds so we don't sample
            # beyond this value again
            if samples[i + 1] < samples[i]:
                left = samples[i + 1]
            else:
                right = samples[i + 1]
