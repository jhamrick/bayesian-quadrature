import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from bayesian_quadrature import util

from . import util as tutil

def test_set_scientific():
    fig, ax = plt.subplots()
    util.set_scientific(ax, -5, 4, axis=None)
    util.set_scientific(ax, -5, 4, axis='x')
    util.set_scientific(ax, -5, 4, axis='y')
    plt.close('all')


def test_slice_sample_normal():
    tutil.npseed()

    def logpdf(x):
        return (-(x ** 2) / 2.) - 0.5 * np.log(2 * np.pi)

    samples = util.slice_sample(logpdf, 10000, 0.5, xval=0, nburn=10, freq=1)
    hist, bins = np.histogram(samples, bins=10, normed=True)
    centers = (bins[:-1] + bins[1:]) / 2.
    bin_pdf = np.exp(logpdf(centers))

    assert (np.abs(bin_pdf - hist) < 0.02).all()


def test_slice_sample_uniform():
    tutil.npseed()

    def logpdf(x):
        if x > 1 or x < 0:
            return -np.inf
        return 0

    samples = util.slice_sample(logpdf, 10000, 0.5, xval=0, nburn=10, freq=1)
    hist, bins = np.histogram(samples, bins=5, normed=True, range=[0, 1])

    assert (np.abs(hist - 1) < 0.03).all()
