import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize as optim
from gp import GP, GaussianKernel, PeriodicKernel

from . import bq_c
from . import util

logger = logging.getLogger("bayesian_quadrature")

DTYPE = np.dtype('float64')
EPS = np.finfo(DTYPE).eps
PREC = np.finfo(DTYPE).precision / 2.0


class BQ(object):
    r"""
    Estimate an integral of the following form using Bayesian
    Quadrature with a Gaussian Process prior:

    .. math::

        Z = \int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x

    Parameters
    ----------
    x : numpy.ndarray
        size :math:`s` array of sample locations
    l : numpy.ndarray
        size :math:`s` array of sample observations

    Other options are:

    kernel : Kernel
        the type of kernel to use
    n_candidate : int
        (maximum) number of candidate points
    candidate_thresh : float
        minimum allowed space between candidates
    x_mean : float
        prior mean, :math:`\mu`
    x_var : float
        prior variance, :math:`\sigma^2`

    Notes
    -----
    This algorithm is an updated version of the one described in
    [OD12]_. The overall idea is:

    1. Estimate :math:`\log\ell` using a GP.
    2. Estimate :math:`\bar{\ell}=\exp(\log\ell)` using second GP.
    3. Integrate exactly under :math:`\bar{\ell}`.

    """

    def __init__(self, x, l, **options):
        """Initialize the Bayesian Quadrature object."""

        # save the given parameters
        self.n_candidate = int(options['n_candidate'])
        self.candidate_thresh = float(options['candidate_thresh'])
        self.x_mean = np.array([options['x_mean']], dtype=DTYPE)
        self.x_cov = np.array([[options['x_var']]], dtype=DTYPE)
        self.kernel = options.get('kernel', GaussianKernel)

        # we only have an analytic solution for Gaussian kernels, so
        # we have to use a slower approximation for any other type of
        # kernel
        self.use_approx = not (self.kernel is GaussianKernel)
        # x points to use if using an approximation
        self._approx_x = None

        if self.use_approx:
            logger.warn("Using approximate solutions for non-Gaussian kernel")

        self.x_s = np.array(x, dtype=DTYPE, copy=True)
        self.l_s = np.array(l, dtype=DTYPE, copy=True)

        if (self.l_s <= 0).any():
            raise ValueError("l_s contains zero or negative values")

        self.ns = self.x_s.shape[0]

        if self.x_s.ndim > 1:
            raise ValueError("invalid number of dimensions for x")
        if self.l_s.ndim > 1:
            raise ValueError("invalid number of dimensions for l")
        if self.x_s.shape != self.l_s.shape:
            raise ValueError("shape mismatch for x and l")

        self.tl_s = np.log(self.l_s)

    def init(self, params_tl, params_l):
        """Initialize the GPs.

        Parameters
        ----------
        params_tl : np.ndarray
            initial parameters for GP over :math:`\log\ell`
        params_l : np.ndarray
            initial parameters for GP over :math:`\exp(\log\ell)`

        """

        self.gp_log_l = GP(
            self.kernel(*params_tl[:-1]),
            self.x_s, self.tl_s, 
            s=params_tl[-1])

        self._fit_log_l()

        self.gp_l = GP(
            self.kernel(*params_l[:-1]),
            self.x_sc, self.l_sc, 
            s=params_l[-1])

        self._fit_l()

    def _choose_candidates(self):
        logger.debug("Choosing candidate points")

        if self.kernel is PeriodicKernel:
            xmin = -np.pi * self.gp_log_l.K.p
            xmax = np.pi * self.gp_log_l.K.p
        else:
            xmin = self.x_s.min() - self.gp_log_l.K.w
            xmax = self.x_s.max() + self.gp_log_l.K.w

        # compute the candidate points
        xc = np.random.uniform(xmin, xmax, self.n_candidate)

        # make sure they don't overlap with points we already have
        bq_c.filter_candidates(xc, self.x_s, self.candidate_thresh)
        xc = np.sort(xc[~np.isnan(xc)])
        return xc

    def _improve_gp_conditioning(self, gp):
        Kxx = gp.Kxx
        cond = np.linalg.cond(Kxx)
        logger.debug("Kxx conditioning number is %s", cond)

        if hasattr(gp, "jitter"):
            jitter = gp.jitter
        else:
            jitter = np.zeros(Kxx.shape[0], dtype=DTYPE)
            gp.jitter = jitter

        # the conditioning is really bad -- just increase the variance
        # a little for all the elements until it's less bad
        idx = np.arange(Kxx.shape[0])
        while np.log10(cond) > PREC:
            logger.debug("Adding jitter to all elements")
            bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)
            cond = np.linalg.cond(Kxx)
            logger.debug("Kxx conditioning number is now %s", cond)

        # now improve just for those elements which result in a
        # negative variance, until there are no more negative elements
        # in the diagonal
        gp._memoized = {'Kxx': Kxx}
        var = np.diag(gp.cov(gp._x))
        while (var < 0).any():
            idx = np.nonzero(var < 0)[0]

            logger.debug("Adding jitter to indices %s", idx)
            bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)

            Kxx = gp.Kxx
            gp._memoized = {'Kxx': Kxx}
            var = np.diag(gp.cov(gp._x))
            cond = np.linalg.cond(Kxx)
            logger.debug("Kxx conditioning number is now %s", cond)

    def _improve_tail_covariance(self):
        Kxx = self.gp_log_l.Kxx
        self.gp_log_l._memoized = {'Kxx': Kxx}
        max_jitter = np.diag(Kxx).max() * 1e-2
        jitter = np.clip(-self.tl_s * 1e-4, 0, max_jitter)
        Kxx += np.eye(self.ns) * jitter
        self.gp_log_l.jitter = jitter

    def _make_approx_x(self, xmin=None, xmax=None, n=300):
        if xmin is None:
            if self.kernel is PeriodicKernel:
                xmin = -np.pi * self.gp_log_l.K.p
            else:
                xmin = self.x_sc.min() - self.gp_log_l.K.w

        if xmax is None:
            if self.kernel is PeriodicKernel:
                xmax = np.pi * self.gp_log_l.K.p
            else:
                xmax = self.x_sc.max() + self.gp_log_l.K.w

        return np.linspace(xmin, xmax, n)

    def _make_approx_px(self, x):
        if self.kernel is PeriodicKernel:
            mu = float(self.x_mean)
            kappa = 1. / float(self.x_cov)
            C = -np.log(2 * np.pi * scipy.special.iv(0, kappa))
            p = np.exp(C + (kappa * np.cos(x - mu)))

        else:
            mu = float(self.x_mean)
            sigma = np.sqrt(float(self.x_cov))
            p = scipy.stats.norm.pdf(x, mu, sigma)

        return p

    def _fit_log_l(self):
        logger.debug("Setting parameters for GP over log(l)")
        self._improve_tail_covariance()
        self._improve_gp_conditioning(self.gp_log_l)
        self._log_l_hypers = self._sample_gp_hypers(self.gp_log_l)

        # choose candidate points
        self.x_c = self._choose_candidates()
        self.x_sc = np.concatenate([self.x_s, self.x_c], axis=0)

        # approximately integrate over lengthscales to get marginal
        # mean function
        gp = self.gp_log_l.copy()
        m_l = np.empty((self._log_l_hypers.shape[0], self.x_c.shape[0]))
        for i in xrange(m_l.shape[0]):
            gp.set_param('w', self._log_l_hypers[i])
            m_l[i] = gp.mean(self.x_c)

        self.l_c = np.exp(np.mean(m_l, axis=0))
        self.l_sc = np.concatenate([self.l_s, self.l_c], axis=0)
        self.nsc = self.ns + self.x_c.shape[0]

    def _fit_l(self):
        logger.debug("Setting parameters for GP over exp(log(l))")
        self._improve_gp_conditioning(self.gp_l)
        self._l_hypers = self._sample_gp_hypers(self.gp_l)

        if self.use_approx:
            self._approx_x = self._make_approx_x()

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
        # the estimated mean of l
        l_mean = self.gp_l.mean(x)
        return l_mean

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
        # the estimated variance of l
        v_log_l = np.diag(self.gp_log_l.cov(x)).copy()
        m_l = self.l_mean(x)
        l_var = v_log_l * m_l ** 2
        l_var[l_var < 0] = 0
        return l_var

    def _approx_Z_mean(self, xo):
        l = self.l_mean(xo)
        p_xo = self._make_approx_px(xo)
        approx = np.trapz(l * p_xo, xo)
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

        x_sc = self.x_sc[:, None]
        alpha_l = self.gp_l.inv_Kxx_y
        h_s, w_s = self.gp_l.K.params
        w_s = np.array([w_s])

        m_Z = bq_c.Z_mean(
            x_sc, alpha_l, h_s, w_s,
            self.x_mean, self.x_cov)

        return m_Z

    def Z_mean(self):
        r"""
        Approximate mean of :math:`Z=\int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x`.

        Returns
        -------
        mean : float
            approximate mean

        """

        if self.use_approx:
            return self._approx_Z_mean(self._approx_x)
        else:
            return self._exact_Z_mean()

    def _approx_Z_var(self, xo):
        m_l = self.l_mean(xo)
        C_tl = self.gp_log_l.cov(xo)
        p_xo = self._make_approx_px(xo)
        approx = np.trapz(
            np.trapz(C_tl * m_l * p_xo, xo) * m_l * p_xo, xo)
        return approx

    def _exact_Z_var(self):
        r"""
        Equivalent to:

        .. math::

            \mathbb{V}(Z)\approx \int\int \mathrm{Cov}_{\log\ell}(x, x^\prime)\bar{\ell}(x)\bar{\ell}(x^\prime)\mathcal{N}(x\ |\ \mu, \sigma^2)\mathcal{N}(x^\prime\ |\ \mu, \sigma^2)\ \mathrm{d}x\ \mathrm{d}x^\prime

        """

        # values for the GPs over l(x) and log(l(x))
        x_s = self.x_s[:, None]
        x_sc = self.x_sc[:, None]

        alpha_l = self.gp_l.inv_Kxx_y
        inv_L_tl = self.gp_log_l.inv_Lxx

        h_l, w_l = self.gp_l.K.params
        w_l = np.array([w_l])
        h_tl, w_tl = self.gp_log_l.K.params
        w_tl = np.array([w_tl])

        V_Z = bq_c.Z_var(
            x_s, x_sc, alpha_l, inv_L_tl,
            h_l, w_l, h_tl, w_tl,
            self.x_mean, self.x_cov)

        return V_Z

    def Z_var(self):
        r"""
        Approximate variance of :math:`Z=\int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x`.

        Returns
        -------
        var : float
            approximate variance

        """

        if self.use_approx:
            return self._approx_Z_var(self._approx_x)
        else:
            return self._exact_Z_var()

    def _expected_inv_K_l(self, x_a):
        # update gp over l
        Kxx = np.empty((self.nsc + 1, self.nsc + 1), dtype=DTYPE)
        Kxx[:-1, :-1] = self.gp_l.Kxx.copy()
        Kxx[:-1, -1:] = self.gp_l.Kxxo(x_a)
        Kxx[-1:, :-1] = self.gp_l.Kxox(x_a)
        Kxx[-1:, -1:] = self.gp_l.Kxoxo(x_a)

        jitter = np.empty(self.nsc + 1, dtype=DTYPE)
        jitter[:-1] = self.gp_l.jitter
        jitter[-1] = 0

        # remove jitter from the x_sc which are close to x_a
        close = np.isclose(Kxx[:self.ns, :self.ns], Kxx[-1, -1], atol=1e-6)
        if close.any():
            idx = np.nonzero(close)[0]
            bq_c.remove_jitter(Kxx, jitter, idx)

        # apply jitter to the x_a -- this is so that when x_a
        # is close to some x_s, our matrix will (hopefully) still be
        # well-conditioned
        idx = np.array([-1])
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)

        # apply jitter to the x_c -- this is because the x_c will
        # probably change with the addition of x_a
        idx = np.arange(self.ns, self.nsc - 1)
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)

        try:
            inv_K_l = np.linalg.inv(Kxx)
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "cannot invert covariance matrix with x_a=%s" % x_a)

        return inv_K_l

    def _approx_expected_squared_mean(self, x_a, xo):
        # include new x_a
        x_sca = np.concatenate([self.x_sc, x_a])

        # compute K_l(x_sca, x_sca)^-1
        inv_K_l = self._expected_inv_K_l(x_a)

        # compute expected transformed mean
        tm_a = self.gp_log_l.mean(x_a)

        # compute expected transformed covariance
        tC_a = self.gp_log_l.cov(x_a)

        p_xo = self._make_approx_px(xo)
        int_K_l = np.trapz(self.gp_l.K(x_sca, xo) * p_xo, xo)

        A_sca = np.dot(int_K_l, inv_K_l)
        A_a = A_sca[-1]
        A_sc_l = np.dot(A_sca[:-1], self.l_sc)

        e1 = bq_c.int_exp_norm(1, tm_a, tC_a)
        e2 = bq_c.int_exp_norm(2, tm_a, tC_a)

        E_m2 = (A_sc_l ** 2) + (2 * A_sc_l * A_a * e1) + (A_a ** 2 * e2)
        return E_m2

    def _exact_expected_squared_mean(self, x_a):
        # include new x_a
        x_sca = np.concatenate([self.x_sc, x_a])

        # compute K_l(x_sca, x_sca)^-1
        inv_K_l = self._expected_inv_K_l(x_a)

        # compute expected transformed mean
        tm_a = self.gp_log_l.mean(x_a)

        # compute expected transformed covariance
        tC_a = self.gp_log_l.cov(x_a)

        expected_sqd_mean = bq_c.expected_squared_mean(
            x_sca[:, None], self.l_sc,
            inv_K_l, tm_a, tC_a,
            self.gp_l.K.h, np.array([self.gp_l.K.w]),
            self.x_mean, self.x_cov)

        if np.isnan(expected_sqd_mean) or expected_sqd_mean < 0:
            raise RuntimeError(
                "invalid expected squared mean for x_a=%s: %s" % (
                    x_a, expected_sqd_mean))

        return expected_sqd_mean

    def expected_squared_mean(self, x_a):
        if self.use_approx:
            f = lambda x: self._approx_expected_squared_mean(
                x, self._approx_x)
        else:
            f = self._exact_expected_squared_mean

        esm = np.empty(x_a.shape[0])
        for i in xrange(x_a.shape[0]):
            esm[i] = f(x_a[[i]])
        return esm

    def expected_Z_var(self, x_a):
        mean_second_moment = self.Z_mean() ** 2 + self.Z_var()
        expected_squared_mean = self.expected_squared_mean(x_a)
        expected_var = mean_second_moment - expected_squared_mean
        return expected_var

    def plot_gp_log_l(self, ax, f_l=None, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x_s.min()
        if xmax is None:
            xmax = self.x_s.max()

        x = np.linspace(xmin, xmax, 1000)
        l_mean = self.gp_log_l.mean(x)
        l_var = np.diag(self.gp_log_l.cov(x)).copy()
        l_var[l_var < 0] = 0
        l_int = 1.96 * np.sqrt(l_var)
        lower = l_mean - l_int
        upper = l_mean + l_int

        if f_l is not None:
            l = np.log(f_l(x))
            ax.plot(x, l, 'k-', lw=2)

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(x, l_mean, 'r-', lw=2)
        ax.plot(self.x_s, self.tl_s, 'ro', markersize=5)
        ax.plot(self.x_c, np.log(self.l_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\log(\ell)$")
        ax.set_xlim(xmin, xmax)
        util.set_scientific(ax, -5, 4)

    def plot_gp_l(self, ax, f_l=None, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x_s.min()
        if xmax is None:
            xmax = self.x_s.max()

        x = np.linspace(xmin, xmax, 1000)
        l_mean = self.l_mean(x)
        l_var = np.diag(self.gp_l.cov(x)).copy()
        l_var[l_var < 0] = 0
        l_int = 1.96 * np.sqrt(l_var)
        lower = l_mean - l_int
        upper = l_mean + l_int

        if f_l is not None:
            l = f_l(x)
            ax.plot(x, l, 'k-', lw=2)

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(x, l_mean, 'r-', lw=2)
        ax.plot(self.x_s, self.l_s, 'ro', markersize=5)
        ax.plot(self.x_c, self.l_mean(self.x_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\exp(\log(\ell))$")
        ax.set_xlim(xmin, xmax)
        util.set_scientific(ax, -5, 4)

    def plot_l(self, ax, f_l=None, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x_s.min()
        if xmax is None:
            xmax = self.x_s.max()

        x = np.linspace(xmin, xmax, 1000)
        l_mean = self.l_mean(x)
        l_var = 1.96 * np.sqrt(self.l_var(x))
        lower = l_mean - l_var
        upper = l_mean + l_var

        if f_l is not None:
            l = f_l(x)
            ax.plot(x, l, 'k-', lw=2, label=r"$\ell(x)$")

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(
            x, l_mean,
            'r-', lw=2, label="final approx")
        ax.plot(
            self.x_s, self.l_s,
            'ro', markersize=5, label="$\ell(x_s)$")
        ax.plot(
            self.x_c, self.l_mean(self.x_c),
            'bs', markersize=4, label="$\exp(m_{\log(\ell)}(x_c))$")

        ax.set_title("Final Approximation")
        ax.set_xlim(xmin, xmax)
        util.set_scientific(ax, -5, 4)

        ax.legend(loc=0, fontsize=10)

    def plot_expected_squared_mean(self, ax, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x_s.min()
        if xmax is None:
            xmax = self.x_s.max()

        x_a = np.linspace(xmin, xmax, 1000)
        exp_sq_m = self.expected_squared_mean(x_a)

        # plot the expected variance
        ax.plot(x_a, exp_sq_m,
                label=r"$E[\mathrm{m}(Z)^2]$",
                color='k', lw=2)

        # plot a line for the current variance
        ax.hlines(
            self.Z_mean() ** 2, xmin, xmax,
            color="#00FF00", lw=2, label=r"$\mathrm{m}(Z)^2$")

        ymin, ymax = ax.get_ylim()

        # plot lines where there are observatiosn
        ax.vlines(
            self.x_sc, ymin, ymax,
            color='k', linestyle='--', alpha=0.5)

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        util.set_scientific(ax, -5, 4)

        ax.legend(loc=0, fontsize=10)
        ax.set_title(r"Expected squared mean of $Z$")

    def plot_expected_variance(self, ax, xmin=None, xmax=None):
        if xmin is None:
            xmin = self.x_s.min()
        if xmax is None:
            xmax = self.x_s.max()

        x_a = np.linspace(xmin, xmax, 1000)
        exp_Z_var = self.expected_Z_var(x_a)

        # plot the expected variance
        ax.plot(x_a, exp_Z_var,
                label=r"$E[\mathrm{Var}(Z)]$",
                color='k', lw=2)

        # plot a line for the current variance
        ax.hlines(
            self.Z_var(), xmin, xmax,
            color="#00FF00", lw=2, label=r"$\mathrm{Var}(Z)$")

        ymin, ymax = ax.get_ylim()

        # plot lines where there are observatiosn
        ax.vlines(
            self.x_sc, ymin, ymax,
            color='k', linestyle='--', alpha=0.5)

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
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

    def add_observation(self, x_a, l_a):
        self.x_s = np.concatenate([self.x_s, x_a])
        self.l_s = np.concatenate([self.l_s, l_a])
        self.ns += x_a.shape[0]
        self.tl_s = np.concatenate([self.tl_s, np.log(l_a)])

        self.gp_log_l.x = self.x_s
        self.gp_log_l.y = self.tl_s
        self.gp_log_l.jitter = np.zeros(self.ns, dtype=DTYPE)
        self._fit_log_l()

        self.gp_l.x = self.x_sc
        self.gp_l.y = self.l_sc
        self.gp_l.jitter = np.zeros(self.nsc, dtype=DTYPE)
        self._fit_l()

    def choose_next(self, cost_fun=None, n=5):
        x = np.empty(n)
        y = np.empty(n)

        if cost_fun is None:
            def cost(x):
                if np.isnan(x):
                    return np.inf
                return -self.expected_squared_mean(x)

        else:
            def cost(x):
                if np.isnan(x):
                    return np.inf
                nesm = -self.expected_squared_mean(x)
                return nesm * cost_fun(x)

        for i in xrange(n):
            xmin, xmax = self.x_s.min(), self.x_s.max()
            x0 = np.array([np.random.uniform(xmin, xmax)])
            res = optim.minimize(
                fun=cost,
                x0=x0,
                tol=1e-10,
                method='Anneal')

            x[i] = float(res['x'])
            y[i] = float(res['fun'])

        target = x[np.argmin(y)]
        return target


    def _sample_gp_hypers(self, gp):
        g = gp.copy()

        def logpdf(w):
            if w < EPS:
                return -np.inf

            g.set_param('w', float(w))
            llh = g.log_lh
            return llh

        window = np.ones(1)
        w0 = np.array([g.get_param('w')])

        logger.info("sampling lengthscales...")
        samps = util.slice_sample(
            logpdf, 110,
            window, w0, 
            nburn=10, freq=1)
        logger.info("done sampling lengthscales")

        return samps
