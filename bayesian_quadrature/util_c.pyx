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


def slice_sample(np.ndarray[DTYPE_t, ndim=2] samples, logpdf, np.ndarray[DTYPE_t, ndim=1] xval, np.ndarray[DTYPE_t, ndim=1] w):
    cdef np.ndarray[DTYPE_t, ndim=1] dir
    cdef np.ndarray[DTYPE_t, ndim=1] left
    cdef np.ndarray[DTYPE_t, ndim=1] right
    cdef np.ndarray[DTYPE_t, ndim=1] loc
    cdef DTYPE_t xpr, pr, yval, logyval
    cdef int pct, newpct, i, n

    # seed the random number generator
    srand(np.random.randint(0, RAND_MAX))

    # number of samples
    n = samples.shape[0]
    d = samples.shape[1]
    
    # keep track of progress
    pct = 0

    # allocate direction and edge vectors
    dir = np.zeros(d, dtype=DTYPE)
    left = np.empty(d, dtype=DTYPE)
    right = np.empty(d, dtype=DTYPE)
    loc = np.empty(d, dtype=DTYPE)

    i = 0
    while i < (n - 1):
        # compute the height of the pdf
        xpr = logpdf(samples[i])

        # sample a value somewhere in that range
        yval = uniform(0, exp(xpr))
        logyval = log(yval)
        # print "logyval is %s" % logyval

        # pick a direction
        dir[:] = np.random.rand(d) - 0.5
        dir[:] /= np.linalg.norm(dir)

        # compute the initial window bounds
        left[:] = -w.copy()
        right[:] = w.copy()

        # widen the bounds until they're outside the slice
        # print "Adjusting left bound..."
        while logpdf(samples[i] + (left * dir)) > logyval:
            left -= w

        # print "Adjusting right bound..."
        while logpdf(samples[i] + (right * dir)) > logyval:
            right += w

        # now sample a new x value
        while True:

            # check the window size to make sure it's not too small
            if ((right - left) < 1e-9).any():
                warn("sampling window shrunk to zero")
                break

            # choose the x and evaluate its probability
            for j in xrange(d):
                loc[j] = uniform(left[j], right[j])

            samples[i + 1] = samples[i] + (loc * dir)
            pr = logpdf(samples[i + 1])

            # if it is within the slice, then we're done
            if pr > logyval:
                # print "Got sample %s" % samples[i + 1]
                i += 1
                break

            # otherwise, shrink the window bounds so we don't sample
            # beyond this value again
            if loc[0] < 0:
                # print "Setting left bound to %s" % loc
                left[:] = loc
            else:
                # print "Setting right bound to %s" % loc
                right[:] = loc

