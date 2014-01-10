import numpy as np
cimport numpy as np

from libc.math cimport exp, log, INFINITY
from libc.stdlib cimport rand, srand, RAND_MAX
from cpython cimport bool

import logging
logger = logging.getLogger("bayesian_quadrature.util")


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef DTYPE_t MIN = log(np.exp2(DTYPE(np.finfo(DTYPE).minexp + 4)))
cdef DTYPE_t EPS = np.finfo(DTYPE).eps
cdef DTYPE_t NAN = np.nan
cdef DTYPE_t FRAND_MAX = float(RAND_MAX)


cdef DTYPE_t uniform(DTYPE_t lo, DTYPE_t hi):
    return (rand() / FRAND_MAX) * (hi - lo) + lo


def slice_sample(np.ndarray[DTYPE_t, mode='c', ndim=2] samples, logpdf, np.ndarray[DTYPE_t, mode='c', ndim=1] xval, DTYPE_t w, bool verbose):
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] dir
    cdef np.ndarray[DTYPE_t, mode='c', ndim=1] temp
    cdef DTYPE_t left, right, loc
    cdef DTYPE_t xpr, pr, yval, logyval
    cdef int pct, newpct, i, j, n

    # seed the random number generator
    srand(np.random.randint(0, RAND_MAX))

    # number of samples
    n = samples.shape[0]
    d = samples.shape[1]
    
    # keep track of progress
    pct = 0

    # allocate direction and edge vectors
    dir = np.zeros(d, dtype=DTYPE)
    temp = np.zeros(d, dtype=DTYPE)

    if verbose:
        logger.debug("Taking %d samples", n)
            
    i = 0
    while i < (n - 1):
        if verbose:
            logger.debug("[%d] x value is %s", i, samples[i])

        # compute the height of the pdf
        xpr = logpdf(samples[i])
        if verbose:
            logger.debug("[%d] xpr is %s", i, xpr)
        
        if xpr == -INFINITY:
            raise RuntimeError("zero probability encountered")

        # sample a value somewhere in that range
        yval = uniform(0, exp(xpr))
        logyval = log(yval)
        if verbose:
            logger.debug("[%d] logyval is %s", i, logyval)

        # pick a direction
        dir[:] = np.random.rand(d) - 0.5
        dir[:] /= np.linalg.norm(dir)

        if verbose:
            logger.debug("[%d] direction is %s", i, dir)

        # compute the initial window bounds
        left = -w
        right = w

        # widen the bounds until they're outside the slice
        if verbose:
            logger.debug("[%d] Adjusting left bound...", i)
        j = 0
        while True:
            temp[:] = samples[i] + left * dir
            if logpdf(temp) < logyval:
                break

            left -= w
            j += 1
            if j > 100:
                if verbose:
                    logger.debug("[%d] Slice is too wide, stopping adjustment", i)
                break

        if verbose:
            logger.debug("[%d] Adjusting right bound...", i)
        j = 0
        while True:
            temp[:] = samples[i] + right * dir
            if logpdf(temp) < logyval:
                break

            right += w
            j += 1
            if j > 100:
                if verbose:
                    logger.debug("[%d] Slice is too wide, stopping adjustment", i)
                break

        if verbose:
            logger.debug("[%d] Left bound is %s", i, left)
            logger.debug("[%d] Right bound is %s", i, right)

        # now sample a new x value
        while True:

            # check the window size to make sure it's not too small
            if (right - left) < 1e-9:
                if verbose:
                    logger.debug("[%d] Sampling window shrunk to zero!", i)
                break

            # choose the x and evaluate its probability
            loc = uniform(left, right)
            samples[i + 1] = samples[i] + loc * dir
            pr = logpdf(samples[i + 1])

            # if it is within the slice, then we're done
            if pr > logyval:
                if verbose:
                    logger.debug("[%d] pr is %s", i, pr)
                    logger.debug("[%d] Got sample %s", i, samples[i + 1])
                i += 1
                break

            # otherwise, shrink the window bounds so we don't sample
            # beyond this value again
            if loc < 0:
                if verbose:
                    logger.debug("[%d] Setting left bound to %s", i, loc)
                left = loc
            else:
                if verbose:
                    logger.debug("[%d] Setting right bound to %s", i, loc)
                right = loc

    if verbose:
        logger.debug("Done")
