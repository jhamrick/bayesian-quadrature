import numpy as np
import logging
import matplotlib.pyplot as plt
import scipy.stats
from gp import GP, GaussianKernel

from . import bq_c

logger = logging.getLogger("bayesian_quadrature")

DTYPE = np.dtype('float64')
EPS = np.finfo(DTYPE).eps
PREC = np.finfo(DTYPE).precision


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

    ntry : int
        number of times to try fitting MLII parameters for GPs
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
        self.ntry = int(options['ntry'])
        self.n_candidate = int(options['n_candidate'])
        self.candidate_thresh = float(options['candidate_thresh'])
        self.x_mean = np.array([options['x_mean']], dtype=DTYPE)
        self.x_cov = np.array([[options['x_var']]], dtype=DTYPE)

        # default kernel parameter values
        self.default_params = dict(
            h=options.get('h', None),
            w=options.get('w', None),
            s=options.get('s', None))

        self.x_s = np.array(x, dtype=DTYPE, copy=True)
        self.l_s = np.array(l, dtype=DTYPE, copy=True)

        self.ns = self.x_s.shape[0]

        if self.x_s.ndim > 1:
            raise ValueError("invalid number of dimensions for x")
        if self.l_s.ndim > 1:
            raise ValueError("invalid number of dimensions for l")
        if self.x_s.shape != self.l_s.shape:
            raise ValueError("shape mismatch for x and l")

        self.tl_s = np.log(self.l_s)
        self.n_sample = self.x_s.size

    def _fit_gp(self, x, y, **kwargs):
        # figure out which parameters we are fitting and how to
        # generate them
        randf = []
        fitmask = np.empty(3, dtype=bool)
        for i, p in enumerate(['h', 'w', 's']):
            # are we fitting the parameter?
            p_v = kwargs.get(p, self.default_params.get(p, None))
            if p_v is None:
                fitmask[i] = True

            # what should we use as an initial parameter?
            p0 = kwargs.get('%s0' % p, p_v)
            if p0 is None:
                randf.append(lambda: np.abs(np.random.normal()))
            else:
                # need to use keyword argument, because python does
                # not assign new values to closure variables in loop
                def f(p=p0):
                    return p
                randf.append(f)

        # generate initial parameter values
        randf = np.array(randf)
        h, w, s = [f() for f in randf]

        # number of restarts
        ntry = kwargs.get('ntry', self.ntry)

        # create the GP object
        gp = GP(GaussianKernel(h, w), x, y, s=s)

        # fit the parameters
        if fitmask.any():
            gp.fit_MLII(fitmask, randf=randf[fitmask], nrestart=ntry)

        return gp

    def _choose_candidates(self):
        # compute the candidate points
        w = self.gp_log_l.K.w
        # TODO: make this threshold be a configuration parameter
        thresh = 1e-1
        xmin = (self.x_s.min() - w) * thresh
        xmax = (self.x_s.max() + w) * thresh
        xc_all = np.random.uniform(xmin, xmax, self.n_candidate) / thresh

        # make sure they don't overlap with points we already have
        xc = []
        for i in xrange(self.n_candidate):
            if (np.abs(xc_all[i] - self.x_s) >= thresh).all():
                xc.append(xc_all[i])
        xc = np.array(sorted(xc))
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

    def _fit_log_l(self, params=None):
        logger.debug("Fitting parameters for GP over log(l)")
        self.gp_log_l = self._fit_gp(self.x_s, self.tl_s)
        if params is not None:
            self.gp_log_l.params = params
        self._improve_gp_conditioning(self.gp_log_l)

    def _fit_l(self, params=None):
        self.x_c = self._choose_candidates()
        self.x_sc = np.concatenate([self.x_s, self.x_c], axis=0)
        self.l_sc = np.concatenate([
            self.l_s,
            np.exp(self.gp_log_l.mean(self.x_c))
        ], axis=0)

        self.nsc = self.ns + self.x_c.shape[0]

        logger.debug("Fitting parameters for GP over exp(log(l))")
        self.gp_l = self._fit_gp(self.x_sc, self.l_sc)
        if params is not None:
            self.gp_l.params = params
        self._improve_gp_conditioning(self.gp_l)

    def fit(self):
        """Run the GP regressions to fit the likelihood function."""

        logger.info("Fitting likelihood")

        self._fit_log_l()
        self._fit_l()

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

    def Z_mean(self):
        r"""
        Approximate mean of :math:`Z=\int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x`.

        Returns
        -------
        mean : float
            approximate mean

        Notes
        -----
        This is equivalent to:

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

    def approx_Z_mean(self, xo):
        p_xo = scipy.stats.norm.pdf(
            xo, self.x_mean[0], np.sqrt(self.x_cov[0, 0]))
        l = self.l_mean(xo)
        approx = np.trapz(l * p_xo, xo)
        return approx

    def Z_var(self):
        r"""
        Approximate variance of :math:`Z=\int \ell(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x`.

        Returns
        -------
        var : float
            approximate variance

        Notes
        -----
        This is equivalent to:

        .. math::

            \mathbb{V}(Z)\approx \int\int \mathrm{Cov}_{\log\ell}(x, x^\prime)\bar{\ell}(x)\bar{\ell}(x^\prime)\mathcal{N}(x\ |\ \mu, \sigma^2)\mathcal{N}(x^\prime\ |\ \mu, \sigma^2)\ \mathrm{d}x\ \mathrm{d}x^\prime

        """

        # values for the GPs over l(x) and log(l(x))
        x_sc = self.x_sc[:, None]

        alpha_l = self.gp_l.inv_Kxx_y
        gp_tl = self.gp_log_l.copy()
        gp_tl.x = self.x_sc
        gp_tl.y = np.log(self.l_sc)
        gp_tl.Kxx[:self.ns, :self.ns] = self.gp_log_l.Kxx.copy()
        self._improve_gp_conditioning(gp_tl)
        inv_L_tl = gp_tl.inv_Lxx

        h_l, w_l = self.gp_l.K.params
        w_l = np.array([w_l])
        h_tl, w_tl = self.gp_log_l.K.params
        w_tl = np.array([w_tl])

        V_Z = bq_c.Z_var(
            x_sc, alpha_l, inv_L_tl,
            h_l, w_l, h_tl, w_tl,
            self.x_mean, self.x_cov)

        return V_Z

    def approx_Z_var(self, xo):
        p_xo = scipy.stats.norm.pdf(
            xo, self.x_mean[0], np.sqrt(self.x_cov[0, 0]))
        m_l = self.l_mean(xo)
        C_tl = self.gp_log_l.cov(xo)
        approx = np.trapz(
            np.trapz(C_tl * m_l * p_xo, xo) * m_l * p_xo, xo)
        return approx

    def _expected_squared_mean(self, x_a):
        # include new x_a
        x_sca = np.concatenate([self.x_sc, x_a])

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
        inv_K_l = np.linalg.inv(Kxx)

        # apply jitter to the x_c -- this is because the x_c will
        # probably change with the addition of x_a
        idx = np.arange(self.ns, self.nsc - 1)
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)
        bq_c.improve_covariance_conditioning(Kxx, jitter, idx=idx)
        inv_K_l = np.linalg.inv(Kxx)

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
            logger.error(
                "invalid expected squared mean: %s",
                expected_sqd_mean)

        return expected_sqd_mean

    def expected_squared_mean(self, x_a):
        esm = np.empty(x_a.shape[0])
        for i in xrange(x_a.shape[0]):
            esm[i] = self._expected_squared_mean(x_a[[i]])
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
        ax.plot(self.x_c, self.gp_log_l.mean(self.x_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\log(\ell)$")
        ax.set_xlim(xmin, xmax)

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
        ax.plot(self.x_sc, self.l_sc, 'ro', markersize=5)
        ax.plot(self.x_c, self.l_mean(self.x_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\exp(\log(\ell))$")
        ax.set_xlim(xmin, xmax)

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
            self.x_sc, self.l_sc,
            'ro', markersize=5, label="$\ell(x_s)$")
        ax.plot(
            self.x_c, self.l_mean(self.x_c),
            'bs', markersize=4, label="$\exp(m_{\log(\ell)}(x_c))$")

        ax.set_title("Final Approximation")
        ax.set_xlim(xmin, xmax)

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
