import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy.stats

from copy import copy, deepcopy
from gp import GP, GaussianKernel, PeriodicKernel

from . import bq_c
from . import util

logger = logging.getLogger("bayesian_quadrature")
DTYPE = np.dtype('float64')
MIN = np.log(np.exp2(np.float64(np.finfo(np.float64).minexp + 4)))
MAX = np.log(np.exp2(np.float64(np.finfo(np.float64).maxexp - 4)))


class BQ(object):
    r"""
    Estimate an integral of the following form using Bayesian
    Quadrature with a Gaussian Process prior:

    .. math::

        Z = \int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x

    See :meth:`~bayesian_quadrature.bq.BQ.load_options` for details on
    allowable options.

    Parameters
    ----------
    x : numpy.ndarray
        size :math:`s` array of sample locations
    l : numpy.ndarray
        size :math:`s` array of sample observations
    options : dict
        Options dictionary

    Notes
    -----
    This algorithm is an updated version of the one described in
    [OD12]_. The overall idea is:

    1. Estimate :math:`\log\ell` using a GP.
    2. Estimate :math:`\bar{\ell}=\exp(\log\ell)` using second GP.
    3. Integrate exactly under :math:`\bar{\ell}`.

    """

    ##################################################################
    # Initialization                                                 #
    ##################################################################

    def __init__(self, x, l, **options):
        """Initialize the Bayesian Quadrature object."""

        #: Vector of observed locations
        self.x_s = np.array(x, dtype=DTYPE)
        #: Vector of observed values
        self.l_s = np.array(l, dtype=DTYPE)

        if (self.l_s <= 0).any():
            raise ValueError("l_s contains zero or negative values")
        if self.x_s.ndim > 1:
            raise ValueError("invalid number of dimensions for x")
        if self.l_s.ndim > 1:
            raise ValueError("invalid number of dimensions for l")
        if self.x_s.shape != self.l_s.shape:
            raise ValueError("shape mismatch for x and l")

        #: Vector of log-transformed observed values
        self.tl_s = np.log(self.l_s)
        #: Number of observations
        self.ns = self.x_s.shape[0]

        self.load_options(**options)
        self.initialized = False
        
        self.gp_log_l = None #: Gaussian process over log(l)
        self.gp_l = None     #: Gaussian process over exp(log(l))
        
        self.x_c = None #: Vector of candidate locations
        self.l_c = None #: Vector of candidate values
        self.nc = None  #: Number of candidate points

        self.x_sc = None #: Vector of observed plus candidate locations
        self.l_sc = None #: Vector of observed plus candidate values
        self.nsc = None  #: Number of observations plus candidates

        self._approx_x = None
        self._approx_px = None

    def load_options(self, kernel, n_candidate, candidate_thresh, x_mean, x_var):
        r"""
        Load options.

        Parameters
        ----------
        kernel : Kernel
            The type of kernel to use. Note that if the kernel is not
            Gaussian, slow approximate (rather than analytic)
            solutions will be used.
        n_candidate : int
            The (maximum) number of candidate points.
        candidate_thresh : float
            Minimum allowed space between candidates.
        x_mean : float
            Prior mean, :math:`\mu`.
        x_var : float
            Prior variance, :math:`\sigma^2`.

        """
        # store the options dictionary for future use
        self.options = {
            'kernel': kernel,
            'n_candidate': int(n_candidate),
            'candidate_thresh': float(candidate_thresh),
            'x_mean': np.array([x_mean], dtype=DTYPE, order='F'),
            'x_cov': np.array([[x_var]], dtype=DTYPE, order='F'),
            'use_approx': not (kernel is GaussianKernel),
            'wrapped': kernel is PeriodicKernel
        }

        if self.options['use_approx']:
            logger.warn("Using approximate solutions for non-Gaussian kernel")

    def init(self, params_tl, params_l):
        """Initialize the GPs.

        Parameters
        ----------
        params_tl : np.ndarray
            initial parameters for GP over :math:`\log\ell`
        params_l : np.ndarray
            initial parameters for GP over :math:`\exp(\log\ell)`

        """

        kernel = self.options['kernel']

        # create the gaussian process over log(l)
        self.gp_log_l = GP(
            kernel(*params_tl[:-1]),
            self.x_s, self.tl_s, 
            s=params_tl[-1])

        # TODO: improve matrix conditioning for log(l)
        self.gp_log_l.jitter = np.zeros(self.ns, dtype=DTYPE)

        # pick candidate points
        self._choose_candidates()

        # create the gaussian process over exp(log(l))
        self.gp_l = GP(
            kernel(*params_l[:-1]),
            self.x_sc, self.l_sc,
            s=params_l[-1])

        # TODO: improve matrix conditioning for exp(log(l))
        self.gp_l.jitter = np.zeros(self.nsc, dtype=DTYPE)

        # make the vector of locations for approximations
        self._approx_x = self._make_approx_x()
        self._approx_px = self._make_approx_px()

        self.initialized = True

    ##################################################################
    # Mean and variance of l                                         #
    ##################################################################

    def l_mean(self, x):
        r"""
        Mean of the final approximation to :math:`\ell`.

        Parameters
        ----------
        x : numpy.ndarray
            :math:`m` array of new sample locations.

        Returns
        -------
        mean : numpy.ndarray
            :math:`m` array of predictive means

        Notes
        -----
        This is just the mean of the GP over :math:`\exp(\log\ell)`, i.e.:

        .. math::

            \mathbb{E}[\bar{\ell}(\mathbf{x})] = \mathbb{E}_{\mathrm{GP}(\exp(\log\ell))}(\mathbf{x})

        """
        return self.gp_l.mean(x)

    def l_var(self, x):
        r"""
        Marginal variance of the final approximation to :math:`\ell`.

        Parameters
        ----------
        x : numpy.ndarray
            :math:`m` array of new sample locations.

        Returns
        -------
        mean : numpy.ndarray
            :math:`m` array of predictive variances

        Notes
        -----
        This is just the diagonal of the covariance of the GP over
        :math:`\log\ell` multiplied by the squared mean of the GP over
        :math:`\exp(\log\ell)`, i.e.:

        .. math::

            \mathbb{V}[\bar{\ell}(\mathbf{x})] = \mathbb{V}_{\mathrm{GP}(\log\ell)}(\mathbf{x})\mathbb{E}_{\mathrm{GP}(\exp(\log\ell))}(\mathbf{x})^2

        """
        v_log_l = np.diag(self.gp_log_l.cov(x)).copy()
        m_l = self.gp_l.mean(x)
        l_var = v_log_l * m_l ** 2
        l_var[l_var < 0] = 0
        return l_var

    ##################################################################
    # Mean of Z                                                      #
    ##################################################################

    def Z_mean(self):
        r"""
        Computes the mean of :math:`Z`, which is defined as:

        .. math ::

            \mathbb{E}[Z]=\int \bar{\ell}(x)p(x)\ \mathrm{d}x

        Returns
        -------
        mean : float

        """

        if self.options['use_approx']:
            return self._approx_Z_mean()
        else:
            return self._exact_Z_mean()

    def _approx_Z_mean(self, xo=None):
        if xo is None:
            xo = self._approx_x
            p_xo = self._approx_px
        else:
            p_xo = self._make_approx_px(xo)

        approx = bq_c.approx_Z_mean(
            np.array(xo[None], order='F'), p_xo, self.l_mean(xo))

        return approx

    def _exact_Z_mean(self):
        r"""
        Equivalent to:

        .. math::

            \begin{align*}
            \mathbb{E}[Z]&\approx \int\bar{\ell}(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x \\
            &= \left(\int K_{\exp(\log\ell)}(x, \mathbf{x}_c)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x\right)K_{\exp(\log\ell)}(\mathbf{x}_c, \mathbf{x}_c)^{-1}\ell(\mathbf{x}_c)
            \end{align*}

        """

        x_sc = np.array(self.x_sc[None], order='F')
        alpha_l = self.gp_l.inv_Kxx_y
        h_s, w_s = self.gp_l.K.params
        w_s = np.array([w_s], order='F')

        m_Z = bq_c.Z_mean(
            x_sc, alpha_l, h_s, w_s,
            self.options['x_mean'], 
            self.options['x_cov'])

        return m_Z

    ##################################################################
    # Variance of Z                                                  #
    ##################################################################

    def Z_var(self):
        r"""
        Computes the variance of :math:`Z`, which is defined as:

        .. math::

            \mathbb{V}(Z)\approx \int\int \mathrm{Cov}_{\log\ell}(x, x^\prime)\bar{\ell}(x)\bar{\ell}(x^\prime)p(x)p(x^\prime)\ \mathrm{d}x\ \mathrm{d}x^\prime

        Returns
        -------
        var : float

        """
        if self.options['use_approx']:
            return self._approx_Z_var()
        else:
            return self._exact_Z_var()

    def _approx_Z_var(self, xo=None):
        if xo is None:
            xo = self._approx_x
            p_xo = self._approx_px
        else:
            p_xo = self._make_approx_px(xo)

        approx = bq_c.approx_Z_var(
            np.array(xo[None], order='F'), p_xo,
            np.array(self.l_mean(xo), order='F'),
            np.array(self.gp_log_l.cov(xo), order='F'))

        return approx

    def _exact_Z_var(self):
        # values for the GPs over l(x) and log(l(x))
        x_s = np.array(self.x_s[None], order='F')
        x_sc = np.array(self.x_sc[None], order='F')

        alpha_l = self.gp_l.inv_Kxx_y
        L_tl = self.gp_log_l.Lxx

        h_l, w_l = self.gp_l.K.params
        w_l = np.array([w_l])
        h_tl, w_tl = self.gp_log_l.K.params
        w_tl = np.array([w_tl])

        V_Z = bq_c.Z_var(
            x_s, x_sc, alpha_l, L_tl,
            h_l, w_l, h_tl, w_tl,
            self.options['x_mean'], 
            self.options['x_cov'])

        return V_Z

    ##################################################################
    # Expected variance of Z                                         #
    ##################################################################

    def expected_Z_var(self, x_a):
        r"""
        Computes the expected variance of :math:`Z` given a new
        observation :math:`x_a`. This is defined as:

        .. math ::

            \mathbb{E}[V(Z)\ |\ \ell_s, \ell_a] = \mathbb{E}[Z\ |\ \ell_s]^2 + V(Z\ |\ \ell_s) - \int \mathbb{E}[Z\ |\ \ell_s, \ell_a]^2 \mathcal{N}(\ell_a\ |\ \hat{m}_a, \hat{C}_a)\ \mathrm{d}\ell_a

        Parameters
        ----------
        x_a : numpy.ndarray
            vector of points for which to (independently) compute the
            expected variance

        Returns
        -------
        out : expected variance for each point in `x_a`

        """
        mean_second_moment = self.Z_mean() ** 2 + self.Z_var()
        expected_squared_mean = self.expected_squared_mean(x_a)
        expected_var = mean_second_moment - expected_squared_mean
        return expected_var

    def expected_squared_mean(self, x_a):
        r"""
        Computes the expected variance of :math:`Z` given a new
        observation :math:`x_a`. This is defined as:

        .. math ::

            \mathbb{E}[\mathbb{E}[Z]^2 |\ \ell_s] = \int \mathbb{E}[Z\ |\ \ell_s, \ell_a]^2 \mathcal{N}(\ell_a\ |\ \hat{m}_a, \hat{C}_a)\ \mathrm{d}\ell_a

        Parameters
        ----------
        x_a : numpy.ndarray
            vector of points for which to (independently) compute the
            expected squared mean

        Returns
        -------
        out : expected squared mean for each point in `x_a`

        """
        esm = np.empty(x_a.shape[0])
        for i in xrange(x_a.shape[0]):
            esm[i] = self._esm(x_a[[i]])
        return esm

    def _esm(self, x_a):
        """Computes the expected square mean for a single point `x_a`."""

        # check for invalid inputs
        if x_a is None or np.isnan(x_a) or np.isinf(x_a):
            raise ValueError("invalid value for x_a: %s", x_a)

        # don't do the heavy computation if the point is close to one
        # we already have
        if np.isclose(x_a, self.x_s, atol=1e-2).any():
            return self.Z_mean() ** 2

        # include new x_a
        x_sca = np.concatenate([self.x_sc, x_a])

        # compute K_l(x_sca, x_sca)
        K_l = self.gp_l.Kxoxo(x_sca)
        jitter = np.zeros(self.nsc + 1)

        # add noise to the candidate points closest to x_a, since they
        # are likely to change
        close = np.abs(self.x_c - x_a) < self.options['candidate_thresh']
        if close.any():
            idx = np.array(np.nonzero(close)[0]) + self.ns
            bq_c.improve_covariance_conditioning(K_l, jitter, idx)

        # also add noise to the new point
        bq_c.improve_covariance_conditioning(K_l, jitter, np.array([self.nsc]))

        # compute expected transformed mean
        tm_a = self.gp_log_l.mean(x_a)

        # compute expected transformed covariance
        tC_a = self.gp_log_l.cov(x_a)

        if self.options['use_approx']:
            xo = self._approx_x
            p_xo = self._approx_px
            Kxxo = np.array(self.gp_l.K(x_sca, xo), order='F')
            try:
                esm = bq_c.approx_expected_squared_mean(
                    self.l_sc, np.array(K_l, order='F'), tm_a, tC_a,
                    np.array(xo[None], order='F'), p_xo, Kxxo)
            except np.linalg.LinAlgError:
                logger.error(
                    "could not compute expected squared mean for x_a=%s" % x_a)
                raise

        else:
            try:
                esm = bq_c.expected_squared_mean(
                    self.l_sc, np.array(K_l, order='F'), tm_a, tC_a,
                    np.array(x_sca[None], order='F'),
                    self.gp_l.K.h, np.array([self.gp_l.K.w]),
                    self.options['x_mean'],
                    self.options['x_cov'])
            except np.linalg.LinAlgError:
                logger.error(
                    "could not compute expected squared mean for x_a=%s" % x_a)
                raise

        if np.isnan(esm) or esm < 0:
            raise RuntimeError(
                "invalid expected squared mean for x_a=%s: %s" % (
                    x_a, esm))

        if np.isinf(esm):
            logger.warn("expected squared mean for x_a=%s is infinity!", x_a)

        return esm

    ##################################################################
    # Hyperparameter optimization/marginalization                    #
    ##################################################################

    def _make_llh_params(self, params):
        nparam = len(params)

        def f(x):
            if x is None or np.isnan(x).any():
                return -np.inf
            try:
                self._set_gp_log_l_params(dict(zip(params, x[:nparam])))
                self._set_gp_l_params(dict(zip(params, x[nparam:])))
            except (ValueError, np.linalg.LinAlgError):
                return -np.inf

            try:
                llh = self.gp_log_l.log_lh + self.gp_l.log_lh
            except (ValueError, np.linalg.LinAlgError):
                return -np.inf

            return llh

        return f

    def fit_hypers(self, params):
        p0_tl = [self.gp_log_l.get_param(p) for p in params]
        p0_l = [self.gp_l.get_param(p) for p in params]
        p0 = np.array(p0_tl + p0_l)

        f = self._make_llh_params(params)
        p0 = util.find_good_parameters(f, p0)
        if p0 is None:
            raise RuntimeError("couldn't find good parameters")
        

    def sample_hypers(self, params, n=1, nburn=10):
        r"""
        Use slice sampling to samples new hyperparameters for the two
        GPs. Note that this will probably cause
        :math:`\bar{ell}(x_{sc})` to change.

        Parameters
        ----------
        params : list of strings
            Dictionary of parameter names to be sampled
        nburn : int
            Number of burn-in samples

        """

        # TODO: should the window be a parameter that is set by the
        # user? is there a way to choose a sane window size
        # automatically?
        nparam = len(params)
        window = 2 * nparam

        p0_tl = [self.gp_log_l.get_param(p) for p in params]
        p0_l = [self.gp_l.get_param(p) for p in params]
        p0 = np.array(p0_tl + p0_l)

        f = self._make_llh_params(params)
        if f(p0) < MIN:
            p0 = util.find_good_parameters(f, p0)
            if p0 is None:
                raise RuntimeError("couldn't find good starting parameters")

        hypers = util.slice_sample(f, nburn+n, window, p0, nburn=nburn, freq=1)
        return hypers[:, :nparam], hypers[:, nparam:]

    ##################################################################
    # Active sampling                                                #
    ##################################################################

    def marginalize(self, funs, n, params):
        r"""
        Compute the approximate marginal functions `funs` by
        approximately marginalizing over the GP hyperparameters.

        Parameters
        ----------
        funs : list
            List of functions for which to compute the marginal.
        n : int
            Number of samples to take when doing the approximate
            marginalization
        params : list or tuple
            List of parameters to marginalize over

        """

        # cache state
        state = deepcopy(self.__getstate__())

        # allocate space for the function values
        values = []
        for fun in funs:
            value = fun()
            try: 
                m = len(value)
            except TypeError:
                values.append(np.empty(n))
            else:
                values.append(np.empty((n, m)))

        # do all the sampling at once, because it is faster
        hypers_tl, hypers_l = self.sample_hypers(params, n=n, nburn=1)

        # compute values for the functions based on the parameters we
        # just sampled
        for i in xrange(n):
            params_tl = dict(zip(params, hypers_tl[i]))
            params_l = dict(zip(params, hypers_l[i]))
            self._set_gp_log_l_params(params_tl)
            self._set_gp_l_params(params_l)
            
            for j, fun in enumerate(funs):
                try:
                    values[j][i] = fun()
                except:
                    logger.error(
                        "error with parameters %s and %s", params_tl, params_l)
                    raise

        # restore state
        self.__setstate__(state)

        return values

    def choose_next(self, x_a, n, params, plot=False):
        f = lambda: -self.expected_squared_mean(x_a)
        values = self.marginalize([f], n, params)
        loss = values[0].mean(axis=0)
        best = np.min(loss)
        close = np.nonzero(np.isclose(loss, best))[0]
        choice = np.random.choice(close)
        best = x_a[choice]

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            self.plot_l(ax1, xmin=x_a.min(), xmax=x_a.max())
            util.vlines(ax1, best, color='g', linestyle='--', lw=2)

            ax2.plot(x_a, loss, 'k-', lw=2)
            util.vlines(ax2, best, color='g', linestyle='--', lw=2)
            ax2.set_title("Negative expected sq. mean")

            fig.set_figwidth(10)
            fig.set_figheight(3.5)

        return best

    def add_observation(self, x_a, l_a):
        if np.isclose(x_a, self.x_s).any():
            raise ValueError("point already sampled")

        self.x_s = np.append(self.x_s, float(x_a))
        self.l_s = np.append(self.l_s, float(l_a))
        self.tl_s = np.append(self.tl_s, np.log(float(l_a)))
        self.ns += 1

        # reinitialize the bq object
        self.init(self.gp_log_l.params, self.gp_l.params)

    ##################################################################
    # Plotting methods                                               #
    ##################################################################

    def plot_gp_log_l(self, ax, f_l=None, xmin=None, xmax=None):
        x = self._make_approx_x(xmin=xmin, xmax=xmax, n=1000)

        if f_l is not None:
            l = np.log(f_l(x))
            ax.plot(x, l, 'k-', lw=2)

        self.gp_log_l.plot(ax, xlim=[x.min(), x.max()], color='r')
        ax.plot(
            self.x_c, np.log(self.l_c),
            'bs', markersize=4, 
            label="$m_{\log\ell}(x_c)$")

        ax.set_title(r"GP over $\log\ell$")
        util.set_scientific(ax, -5, 4)

    def plot_gp_l(self, ax, f_l=None, xmin=None, xmax=None):
        x = self._make_approx_x(xmin=xmin, xmax=xmax, n=1000)

        if f_l is not None:
            l = f_l(x)
            ax.plot(x, l, 'k-', lw=2)

        self.gp_l.plot(ax, xlim=[x.min(), x.max()], color='r')
        ax.plot(
            self.x_c, self.l_c,
            'bs', markersize=4, 
            label="$\exp(m_{\log\ell}(x_c))$")

        ax.set_title(r"GP over $\exp(\log\ell)$")
        util.set_scientific(ax, -5, 4)

    def plot_l(self, ax, f_l=None, xmin=None, xmax=None, legend=True):
        x = self._make_approx_x(xmin=xmin, xmax=xmax, n=1000)

        if f_l is not None:
            l = f_l(x)
            ax.plot(x, l, 'k-', lw=2, label="$\ell(x)$")

        l_mean = self.l_mean(x)
        l_var = np.sqrt(self.l_var(x))
        lower = l_mean - l_var
        upper = l_mean + l_var

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(
            x, l_mean,
            'r-', lw=2, 
            label="final approx")
        ax.plot(
            self.x_s, self.l_s,
            'ro', markersize=5, 
            label="$\ell(x_s)$")
        ax.plot(
            self.x_c, self.l_c,
            'bs', markersize=4, 
            label="$\exp(m_{\log\ell}(x_c))$")

        ax.set_title("Final Approximation")
        ax.set_xlim(x.min(), x.max())
        util.set_scientific(ax, -5, 4)

        if legend:
            ax.legend(loc=0, fontsize=10)

    def plot_expected_squared_mean(self, ax, xmin=None, xmax=None):
        x_a = self._make_approx_x(xmin=xmin, xmax=xmax, n=1000)
        exp_sq_m = self.expected_squared_mean(x_a)

        # plot the expected variance
        ax.plot(x_a, exp_sq_m,
                label=r"$E[\mathrm{m}(Z)^2]$",
                color='k', lw=2)
        ax.set_xlim(x_a.min(), x_a.max())

        # plot a line for the current variance
        util.hlines(
            ax, self.Z_mean() ** 2,
            color="#00FF00", lw=2, 
            label=r"$\mathrm{m}(Z)^2$")

        # plot lines where there are observatiosn
        util.vlines(ax, self.x_sc, color='k', linestyle='--', alpha=0.5)

        util.set_scientific(ax, -5, 4)
        ax.legend(loc=0, fontsize=10)
        ax.set_title(r"Expected squared mean of $Z$")

    def plot_expected_variance(self, ax, xmin=None, xmax=None):
        x_a = self._make_approx_x(xmin=xmin, xmax=xmax, n=1000)
        exp_Z_var = self.expected_Z_var(x_a)

        # plot the expected variance
        ax.plot(x_a, exp_Z_var,
                label=r"$E[\mathrm{Var}(Z)]$",
                color='k', lw=2)
        ax.set_xlim(x_a.min(), x_a.max())

        # plot a line for the current variance
        util.hlines(
            ax, self.Z_var(), 
            color="#00FF00", lw=2, 
            label=r"$\mathrm{Var}(Z)$")

        # plot lines where there are observations
        util.vlines(ax, self.x_sc, color='k', linestyle='--', alpha=0.5)

        util.set_scientific(ax, -5, 4)
        ax.legend(loc=0, fontsize=10)
        ax.set_title(r"Expected variance of $Z$")

    def plot(self, f_l=None, xmin=None, xmax=None):
        fig, axes = plt.subplots(1, 3)

        self.plot_gp_log_l(axes[0], f_l=f_l, xmin=xmin, xmax=xmax)
        self.plot_gp_l(axes[1], f_l=f_l, xmin=xmin, xmax=xmax)
        self.plot_l(axes[2], f_l=f_l, xmin=xmin, xmax=xmax)

        ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes[1:]])
        ymin = min(ymins)
        ymax = max(ymaxs)
        for ax in axes[1:]:
            ax.set_ylim(ymin, ymax)

        fig.set_figwidth(14)
        fig.set_figheight(3.5)

        return fig, axes

    ##################################################################
    # Saving and restoring                                           #
    ##################################################################

    def __getstate__(self):
        state = {}

        state['x_s'] = self.x_s
        state['l_s'] = self.l_s
        state['tl_s'] = self.tl_s

        state['options'] = self.options
        state['initialized'] = self.initialized

        if self.initialized:
            state['gp_log_l'] = self.gp_log_l
            state['gp_log_l_jitter'] = self.gp_log_l.jitter

            state['gp_l'] = self.gp_l
            state['gp_l_jitter'] = self.gp_l.jitter

            state['_approx_x'] = self._approx_x
            state['_approx_px'] = self._approx_px

        return state

    def __setstate__(self, state):
        self.x_s = state['x_s']
        self.l_s = state['l_s']
        self.tl_s = state['tl_s']
        self.ns = self.x_s.shape[0]

        self.options = state['options']
        self.initialized = state['initialized']

        if self.initialized:
            self.gp_log_l = state['gp_log_l']
            self.gp_log_l.jitter = state['gp_log_l_jitter']

            self.gp_l = state['gp_l']
            self.gp_l.jitter = state['gp_l_jitter']

            self.x_sc = self.gp_l._x
            self.l_sc = self.gp_l._y
            self.nsc = self.x_sc.shape[0]

            self.x_c = self.x_sc[self.ns:]
            self.l_c = self.l_sc[self.ns:]
            self.nc = self.nsc - self.ns

            self._approx_x = state['_approx_x']
            self._approx_px = state['_approx_px']

        else:
            self.gp_log_l = None
            self.gp_l = None

            self.x_c = None
            self.l_c = None
            self.nc = None

            self.x_sc = None
            self.l_sc = None
            self.nsc = None

            self._approx_x = None
            self._approx_px = None

    ##################################################################
    # Copying                                                        #
    ##################################################################

    def __copy__(self):
        state = self.__getstate__()
        cls = type(self)
        bq = cls.__new__(cls)
        bq.__setstate__(state)
        return bq

    def __deepcopy__(self, memo):
        state = deepcopy(self.__getstate__(), memo)
        cls = type(self)
        bq = cls.__new__(cls)
        bq.__setstate__(state)
        return bq

    def copy(self, deep=True):
        if deep:
            out = deepcopy(self)
        else:
            out = copy(self)
        return out

    ##################################################################
    # Helper methods                                                 #
    ##################################################################

    def _set_gp_log_l_params(self, params):
        # set the parameter values
        for p, v in params.iteritems():
            self.gp_log_l.set_param(p, v)

        # TODO: improve matrix conditioning for log(l)
        self.gp_log_l.jitter.fill(0)

        # update values of candidate points
        m = self.gp_log_l.mean(self.x_c)
        V = np.diag(self.gp_log_l.cov(self.x_c))
        tl_c = m + 2 * np.sqrt(V)
        if (tl_c > MAX).any():
            raise np.linalg.LinAlgError("GP mean is too large")

        self.l_c = np.exp(m)
        self.l_sc = np.concatenate([self.l_s, self.l_c], axis=0)

        # update the locations and values for exp(log(l))
        self.gp_l.x = self.x_sc
        self.gp_l.y = self.l_sc

        # TODO: improve matrix conditioning for exp(log(l))
        self.gp_l.jitter.fill(0)

    def _set_gp_l_params(self, params):
        # set the parameter values
        for p, v in params.iteritems():
            self.gp_l.set_param(p, v)

        # TODO: improve matrix conditioning for exp(log(l))
        self.gp_l.jitter.fill(0)

    def _choose_candidates(self):
        logger.debug("Choosing candidate points")

        if self.options['wrapped']:
            xmin = -np.pi * self.gp_log_l.K.p
            xmax = np.pi * self.gp_log_l.K.p
        else:
            xmin = self.x_s.min() - self.gp_log_l.K.w
            xmax = self.x_s.max() + self.gp_log_l.K.w

        # compute the candidate points
        xc = np.random.uniform(xmin, xmax, self.options['n_candidate'])

        # make sure they don't overlap with points we already have
        bq_c.filter_candidates(xc, self.x_s, self.options['candidate_thresh'])

        # save the locations and compute the values
        self.x_c = np.sort(xc[~np.isnan(xc)])
        self.l_c = np.exp(self.gp_log_l.mean(self.x_c))
        self.nc = self.x_c.shape[0]

        # concatenate with the observations we already have
        self.x_sc = np.concatenate([self.x_s, self.x_c], axis=0)
        self.l_sc = np.concatenate([self.l_s, self.l_c], axis=0)
        self.nsc = self.ns + self.nc

    def _make_approx_x(self, xmin=None, xmax=None, n=1000):
        if xmin is None:
            if self.options['wrapped']:
                xmin = -np.pi * self.gp_log_l.K.p
            else:
                xmin = self.x_sc.min() - self.gp_log_l.K.w

        if xmax is None:
            if self.options['wrapped']:
                xmax = np.pi * self.gp_log_l.K.p
            else:
                xmax = self.x_sc.max() + self.gp_log_l.K.w

        return np.linspace(xmin, xmax, n)

    def _make_approx_px(self, x=None):
        if x is None:
            x = self._approx_x

        p = np.empty(x.size, order='F')

        if self.options['wrapped']:
            bq_c.p_x_vonmises(
                p, x,
                float(self.options['x_mean']),
                1. / float(self.options['x_cov']))

        else:
            bq_c.p_x_gaussian(
                p, np.array(x[None], order='F'),
                self.options['x_mean'],
                self.options['x_cov'])

        return p
