import matplotlib.pyplot as plt
from bayesian_quadrature import util


def test_set_scientific():
    fig, ax = plt.subplots()
    util.set_scientific(ax, -5, 4, axis=None)
    util.set_scientific(ax, -5, 4, axis='x')
    util.set_scientific(ax, -5, 4, axis='y')
    plt.close('all')
