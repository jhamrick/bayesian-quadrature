import numpy as np
import pytest
import matplotlib.pyplot as plt
from gp import GP

from .. import BQ
from . import util

import logging
logger = logging.getLogger("bayesian_quadrature")
logger.setLevel("DEBUG")

DTYPE = util.DTYPE
options = util.options


def test_init():
    util.npseed()
    x, y = util.make_xy()
    bq = BQ(x, y, **options)
    assert (x == bq.x_s).all()
    assert (y == bq.l_s).all()
    assert (np.log(y) == bq.tl_s).all()
    assert (x.shape[0] == bq.ns)

    assert not bq.initialized
    assert bq.gp_log_l is None
    assert bq.gp_l is None
    assert bq.x_c is None
    assert bq.l_c is None
    assert bq.nc is None
    assert bq.x_sc is None
    assert bq.l_sc is None
    assert bq.nsc is None
    assert bq._approx_x is None
    assert bq._approx_px is None

    util.init_bq(bq)
    assert (x == bq.x_s).all()
    assert (y == bq.l_s).all()
    assert (np.log(y) == bq.tl_s).all()
    assert (x.shape[0] == bq.ns)

    assert bq.initialized
    assert bq.gp_log_l is not None
    assert hasattr(bq.gp_log_l, 'jitter')
    assert bq.gp_l is not None
    assert hasattr(bq.gp_l, 'jitter')
    assert bq.x_c is not None
    assert bq.l_c is not None
    assert bq.nc is not None
    assert bq.x_sc is not None
    assert bq.l_sc is not None
    assert bq.nsc is not None
    assert bq._approx_x is not None
    assert bq._approx_px is not None

    
def test_bad_init():
    util.npseed()
    x, y = util.make_xy()

    with pytest.raises(ValueError):
        BQ(x[:, None], y, **options)
    with pytest.raises(ValueError):
        BQ(x, y[:, None], **options)
    with pytest.raises(ValueError):
        BQ(x[:-1], y, **options)
    with pytest.raises(ValueError):
        BQ(x, y[:-1], **options)
    with pytest.raises(ValueError):
        BQ(x, -y, **options)


def test_choose_candidates():
    util.npseed()

    bq = util.make_bq(nc=1000)
    assert bq.x_c.ndim == 1
    assert bq.x_sc.size >= bq.x_s.size

    diff = np.abs(bq.x_sc[:, None] - bq.x_c[None])
    thresh = bq.options['candidate_thresh']
    assert ((diff > thresh) | (diff == 0)).all()


def test_l_mean():
    util.npseed()
    bq = util.make_bq()
    l = bq.l_mean(bq.x_s)
    assert np.allclose(l, bq.l_s, atol=1e-4)


def test_Z_mean():
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    approx_Z = bq._approx_Z_mean(xo)
    calc_Z = bq._exact_Z_mean()

    assert np.allclose(approx_Z, calc_Z, atol=1e-5)


def test_Z_mean_same():
    util.npseed()
    bq = util.make_bq()

    means = np.empty(100)
    for i in xrange(100):
        means[i] = bq.Z_mean()
    assert (means[0] == means).all()


@pytest.mark.xfail(reason="https://github.com/numpy/numpy/issues/661")
def test_Z_var_same():
    util.npseed()
    bq = util.make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert (vars[0] == vars).all()


def test_Z_var_close():
    util.npseed()
    bq = util.make_bq()

    vars = np.empty(100)
    for i in xrange(100):
        vars[i] = bq.Z_var()
    assert np.allclose(vars[0], vars)


def test_Z_var():
    # int int m_l(x) m_l(x') C_tl(x, x') dx dx'
    util.npseed()
    bq = util.make_bq()
    xo = util.make_xo()
    approx_var = bq._approx_Z_var(xo)
    calc_var = bq._exact_Z_var()
    assert np.allclose(approx_var, calc_var, atol=1e-4)


def test_expected_Z_var_close():
    util.npseed()
    bq = util.make_bq()
    Z_var = bq.Z_var()
    E_Z_var = bq.expected_Z_var(bq.x_s)
    assert np.allclose(E_Z_var, Z_var, atol=1e-4)


def test_expected_squared_mean_valid():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 10)
    esm = bq.expected_squared_mean(x_a)
    assert (esm >= 0).all()


def test_expected_squared_mean_params():
    util.npseed()
    bq = util.make_bq()
    with pytest.raises(ValueError):
        bq.expected_squared_mean(np.array([np.nan]))
    with pytest.raises(ValueError):
        bq.expected_squared_mean(np.array([np.inf]))
    with pytest.raises(ValueError):
        bq.expected_squared_mean(np.array([-np.inf]))


def test_expected_squared_mean():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 20)

    esm = bq.expected_squared_mean(x_a)
    
    bq.options['use_approx'] = True
    approx = bq.expected_squared_mean(x_a)

    assert np.allclose(approx, esm, rtol=1)


def test_plot_gp_log_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_log_l(ax)
    ax.cla()

    bq.plot_gp_log_l(ax, f_l=lambda x: np.log(util.f_x(x)))
    ax.cla()

    bq.plot_gp_log_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_gp_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_gp_l(ax)
    ax.cla()

    bq.plot_gp_l(ax, f_l=util.f_x)
    ax.cla()

    bq.plot_gp_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_l():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_l(ax)
    ax.cla()

    bq.plot_l(ax, f_l=util.f_x)
    ax.cla()

    bq.plot_l(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot():
    util.npseed()
    bq = util.make_bq()

    bq.plot()
    plt.close('all')

    bq.plot(f_l=util.f_x)
    plt.close('all')

    bq.plot(xmin=-10, xmax=10)
    plt.close('all')


def test_plot_expected_variance():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_variance(ax)
    ax.cla()

    bq.plot_expected_variance(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_plot_expected_squared_mean():
    util.npseed()
    bq = util.make_bq()
    fig, ax = plt.subplots()

    bq.plot_expected_squared_mean(ax)
    ax.cla()

    bq.plot_expected_squared_mean(ax, xmin=-10, xmax=10)
    ax.cla()

    plt.close('all')


def test_l():
    util.npseed()
    bq = util.make_bq()
    assert (np.log(bq.l_s) == bq.tl_s).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()
    assert (bq.l_sc[bq.ns:] == np.exp(bq.gp_log_l.mean(bq.x_c))).all()


def test_expected_squared_mean_1():
    util.npseed()
    X = np.linspace(-5, 5, 20)[:, None]
    for x in X:
        bq = util.make_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2

        E_m2 = bq.expected_squared_mean(x)
        assert np.allclose(m2, E_m2, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2_close, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-8)
        assert np.allclose(m2, E_m2_close, atol=1e-4)


def test_periodic():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    y = util.f_xp(x)
    assert np.allclose(bq.l_mean(x), y, atol=1e-3)


def test_periodic_z_mean():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    l = bq.l_mean(x)
    p_x = bq._make_approx_px(x)
    approx_z = np.trapz(l * p_x, x)
    assert np.allclose(bq.Z_mean(), approx_z)
    

def test_periodic_z_var():
    util.npseed()
    bq = util.make_periodic_bq()
    x = np.linspace(-np.pi, np.pi, 1000)
    l = bq.l_mean(x)
    C = bq.gp_log_l.cov(x)
    p_x = bq._make_approx_px(x)
    approx_z = np.trapz(np.trapz(C * l * p_x, x) * l * p_x, x)
    assert np.allclose(bq.Z_var(), approx_z)


@pytest.mark.xfail(reason="poorly conditioned matrix")
def test_periodic_expected_squared_mean():
    util.npseed()
    bq = util.make_periodic_bq(nc=0)
    x_a = np.random.uniform(-np.pi, np.pi, 20)[:, None]
    x = np.linspace(-np.pi, np.pi, 1000)

    for xa in x_a:
        esm = bq.expected_squared_mean(xa)
        approx = bq._approx_expected_squared_mean(xa, x)
        assert np.allclose(esm, approx)


def test_periodic_expected_squared_mean_1():
    util.npseed()
    X = np.linspace(-np.pi, np.pi, 20)[:, None]
    for x in X:
        bq = util.make_periodic_bq(x=x, nc=0)
        m2 = bq.Z_mean() ** 2
        E_m2 = bq.expected_squared_mean(x)
        assert np.allclose(m2, E_m2, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-10)
        assert np.allclose(m2, E_m2_close, atol=1e-4)

        E_m2_close = bq.expected_squared_mean(x - 1e-8)
        assert np.allclose(m2, E_m2_close, atol=1e-4)


def test_add_observation():
    util.npseed()
    bq = util.make_bq()
    x = bq.x_s.copy()
    l = bq.l_s.copy()
    tl = bq.tl_s.copy()

    x_a = 20
    l_a = util.f_x(x_a)
    tl_a = np.log(l_a)

    bq.add_observation(x_a, l_a)
    assert (bq.x_s == np.append(x, x_a)).all()
    assert (bq.l_s == np.append(l, l_a)).all()
    assert (bq.tl_s == np.append(tl, tl_a)).all()
        
    assert (bq.x_s == bq.x_sc[:bq.ns]).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()

    old_x_s = bq.x_s.copy()
    old_l_s = bq.l_s.copy()
    bq.add_observation(x[0], l[0])
    assert (old_x_s == bq.x_s).all()
    assert (old_l_s == bq.l_s).all()


def test_approx_add_observation():
    util.npseed()
    bq = util.make_periodic_bq(np.linspace(-np.pi, 0, 4))
    x = bq.x_s.copy()
    l = bq.l_s.copy()
    tl = bq.tl_s.copy()

    x_a = np.pi / 2.
    l_a = util.f_x(x_a)
    tl_a = np.log(l_a)

    bq.add_observation(x_a, l_a)
    assert (bq.x_s == np.append(x, x_a)).all()
    assert (bq.l_s == np.append(l, l_a)).all()
    assert (bq.tl_s == np.append(tl, tl_a)).all()

    assert (bq.x_s == bq.x_sc[:bq.ns]).all()
    assert (bq.l_s == bq.l_sc[:bq.ns]).all()

    old_x_s = bq.x_s.copy()
    old_l_s = bq.l_s.copy()
    bq.add_observation(x[0], l[0])
    assert (old_x_s == bq.x_s).all()
    assert (old_l_s == bq.l_s).all()


def test_getstate():
    util.npseed()
    bq = util.make_bq(init=False)

    # uninitialized
    state = bq.__getstate__()
    assert (state['x_s'] == bq.x_s).all()
    assert (state['l_s'] == bq.l_s).all()
    assert (state['tl_s'] == bq.tl_s).all()
    assert state['options'] == bq.options
    assert state['initialized'] == bq.initialized
    assert sorted(state.keys()) == sorted(
        ['x_s', 'l_s', 'tl_s', 'options', 'initialized'])

    util.init_bq(bq)
    state = bq.__getstate__()
    assert (state['x_s'] == bq.x_s).all()
    assert (state['l_s'] == bq.l_s).all()
    assert (state['tl_s'] == bq.tl_s).all()
    assert state['options'] == bq.options
    assert state['initialized'] == bq.initialized
    assert state['gp_log_l'] == bq.gp_log_l
    assert (state['gp_log_l_jitter'] == bq.gp_log_l.jitter).all()
    assert state['gp_l'] == bq.gp_l
    assert (state['gp_l_jitter'] == bq.gp_l.jitter).all()
    assert sorted(state.keys()) == sorted(
        ['x_s', 'l_s', 'tl_s', 'options', 'initialized',
         'gp_log_l', 'gp_log_l_jitter', 'gp_l', 'gp_l_jitter',
         '_approx_x', '_approx_px'])


def test_copy():
    util.npseed()
    bq1 = util.make_bq(init=False)
    bq2 = bq1.copy(deep=False)
    assert bq1 is not bq2

    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) == sorted(state2.keys())

    for key in state1.keys():
        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is state2[key]

    util.init_bq(bq1)
    assert bq1.initialized
    assert not bq2.initialized
    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) != sorted(state2.keys())

    for key in state1.keys():
        if key == 'initialized':
            continue
        if key not in state2:
            continue

        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is state2[key]

    bq1 = util.make_bq()
    bq2 = bq1.copy(deep=False)

    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) == sorted(state2.keys())

    for key in state1.keys():
        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is state2[key]

def test_deepcopy():
    util.npseed()
    bq1 = util.make_bq(init=False)
    bq2 = bq1.copy(deep=True)
    assert bq1 is not bq2

    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) == sorted(state2.keys())

    for key in state1.keys():
        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is not state2[key]

    util.init_bq(bq1)
    assert bq1.initialized
    assert not bq2.initialized
    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) != sorted(state2.keys())

    for key in state1.keys():
        if key == 'initialized':
            continue

        if key not in state2:
            continue

        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is not state2[key]

    bq1 = util.make_bq()
    bq2 = bq1.copy(deep=True)

    state1 = bq1.__getstate__()
    state2 = bq2.__getstate__()
    assert sorted(state1.keys()) == sorted(state2.keys())

    for key in state1.keys():
        if isinstance(state1[key], np.ndarray):
            assert (state1[key] == state2[key]).all()
        elif not isinstance(state1[key], GP):
            assert state1[key] == state2[key]

        if not isinstance(state1[key], bool):
            assert state1[key] is not state2[key]


def test_set_params():
    util.npseed()
    bq = util.make_bq()
    params_tl = bq.gp_log_l.params
    params_l = bq.gp_l.params
    x_sc = bq.x_sc.copy()
    l_sc = bq.l_sc.copy()

    bq._set_gp_log_l_params(dict(h=10, w=3.0, s=0.01))
    assert (bq.gp_log_l.params != params_tl).all()
    assert (bq.gp_l.params == params_l).all()
    assert (bq.gp_log_l.jitter == 0).all()
    assert (bq.gp_l.jitter == 0).all()
    assert (bq.x_sc == x_sc).all()
    assert not (bq.l_sc == l_sc).all()

    params_tl = bq.gp_log_l.params

    bq._set_gp_l_params(dict(h=0.3, w=1.4, s=0.01))
    assert (bq.gp_log_l.params == params_tl).all()
    assert (bq.gp_l.params != params_l).all()
    assert (bq.gp_log_l.jitter == 0).all()
    assert (bq.gp_l.jitter == 0).all()


def test_fit_hypers():
    util.npseed()
    bq = util.make_bq()
    llh = bq.gp_log_l.log_lh + bq.gp_l.log_lh
    bq.fit_hypers(['h', 'w'])
    new_llh = bq.gp_log_l.log_lh + bq.gp_l.log_lh
    assert new_llh >= llh


def test_sample_hypers():
    util.npseed()
    bq = util.make_bq()
    params = ['h', 'w']
    params_tl = {p: bq.gp_log_l.get_param(p) for p in params}
    params_l = {p: bq.gp_l.get_param(p) for p in params}

    bq.sample_hypers(params)
    assert not np.isinf(bq.gp_log_l.log_lh)
    assert not np.isinf(bq.gp_l.log_lh)

    for p in params:
        assert bq.gp_log_l.get_param(p) != params_tl[p]
        assert bq.gp_l.get_param(p) != params_l[p]

    bq = util.make_bq(init=False)
    bq.init(params_tl=(15, 2, 0), params_l=(0.2, 1.3, 0))
    bq.sample_hypers(['h', 'w'])
    assert not np.isinf(bq.gp_log_l.log_lh)
    assert not np.isinf(bq.gp_l.log_lh)

    bq = util.make_bq(init=False)
    bq.init(params_tl=(15, 2, 0), params_l=(0.00002, 1.3, 0))
    bq.sample_hypers(['h'])
    assert not np.isinf(bq.gp_log_l.log_lh)
    assert not np.isinf(bq.gp_l.log_lh)

    bq = util.make_bq(init=False)
    bq.init(params_tl=(15, 2, 0), params_l=(0.2, 5, 0))
    bq.sample_hypers(['w'])
    assert not np.isinf(bq.gp_log_l.log_lh)
    assert not np.isinf(bq.gp_l.log_lh)

    bq = util.make_bq(init=False)
    bq.init(params_tl=(15, 2, 0), params_l=(0.00000002, 1.3, 0))
    with pytest.raises(RuntimeError):
        bq.sample_hypers(['w'])


def test_marginal_mean():
    util.npseed()
    bq = util.make_bq()

    # marginal mean
    values = bq.marginalize(
        [bq.Z_mean], 20, ['h', 'w'])

    assert len(values) == 1
    assert values[0].shape == (20,)


def test_marginal_mean_and_variance():
    util.npseed()
    bq = util.make_bq()

    # marginal mean and variance
    values = bq.marginalize(
        [bq.Z_mean, bq.Z_var], 20, ['h', 'w'])

    assert len(values) == 2
    assert values[0].shape == (20,)
    assert values[1].shape == (20,)


def test_marginal_loss():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 5)

    # setting params
    llh = bq.gp_log_l.log_lh + bq.gp_l.log_lh
    f = lambda: bq.expected_squared_mean(x_a)
    values = bq.marginalize([f], 20, ['h', 'w'])

    assert len(values) == 1
    assert values[0].shape == (20, 5)


def test_choose_next():
    util.npseed()
    bq = util.make_bq()
    x_a = np.random.uniform(-10, 10, 5)

    bq.choose_next(x_a, 20, ['h', 'w'])
    bq.choose_next(x_a, 20, ['h', 'w'], plot=True)
