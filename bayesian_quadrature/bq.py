import numpy as np
import logging
import matplotlib.pyplot as plt
from gp import GP, GaussianKernel

from . import bq_c

logger = logging.getLogger("bayesian_quadrature")

DTYPE = np.dtype('float64')
EPS = np.finfo(DTYPE).eps


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

    def __init__(self, x, l,
                 ntry, n_candidate,
                 x_mean, x_var,
                 h=None, w=None, s=None):
        """Initialize the Bayesian Quadrature object."""

        # save the given parameters
        self.ntry = int(ntry)
        self.n_candidate = int(n_candidate)
        self.x_mean = np.array([x_mean], dtype=DTYPE)
        self.x_cov = np.array([[x_var]], dtype=DTYPE)

        # default kernel parameter values
        self.default_params = dict(h=h, w=w, s=s)

        self.x = np.array(x, dtype=DTYPE, copy=True)
        self.l = np.array(l, dtype=DTYPE, copy=True)

        if self.x.ndim > 1:
            raise ValueError("invalid number of dimensions for x")
        if self.l.ndim > 1:
            raise ValueError("invalid number of dimensions for l")
        if self.x.shape != self.l.shape:
            raise ValueError("shape mismatch for x and l")

        self.log_l = np.log(self.l)
        self.n_sample = self.x.size

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
        thresh = 1e-1
        xmin = (self.x.min() - w) * thresh
        xmax = (self.x.max() + w) * thresh
        xc_all = np.random.uniform(xmin, xmax, self.n_candidate) / thresh

        # make sure they don't overlap with points we already have
        xc = []
        for i in xrange(self.n_candidate):
            if (np.abs(xc_all[i] - self.x) >= thresh).all():
                xc.append(xc_all[i])
        xc = np.array(xc)
        return xc

    def _fit_log_l(self):
        logger.info("Fitting parameters for GP over log(l)")
        self.gp_log_l = self._fit_gp(self.x, self.log_l)

    def _fit_l(self):
        self.x_c = self._choose_candidates()
        self.x_sc = np.concatenate([self.x, self.x_c], axis=0)

        m_log_l = self.gp_log_l.mean(self.x_sc)
        v_log_l = np.diag(self.gp_log_l.cov(self.x_sc))
        self.l_sc = np.exp(m_log_l + 0.5 * v_log_l)

        logger.info("Fitting parameters for GP over exp(log(l))")
        self.gp_l = self._fit_gp(self.x_sc, self.l_sc)

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
        v_log_l = np.diag(self.gp_log_l.cov(x))
        m_l = self.gp_l.mean(x)
        l_var = v_log_l * m_l ** 2
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

            \mathbb{E}[Z]\approx \int\bar{\ell}(x)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x = \left(\int K_{\exp(\log\ell)}(x, \mathbf{x}_c)\mathcal{N}(x\ |\ \mu, \sigma^2)\ \mathrm{d}x\right)K_{\exp(\log\ell)}(\mathbf{x}_c, \mathbf{x}_c)^{-1}\ell(\mathbf{x}_c)

        """

        x_sc = self.x_sc[:, None]
        alpha_l = self.gp_l.inv_Kxx_y
        h_s, w_s = self.gp_l.K.params
        w_s = np.array([w_s])

        m_Z = bq_c.Z_mean(
            x_sc, alpha_l, h_s, w_s,
            self.x_mean, self.x_cov)

        return m_Z

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
        Kxcxc = self.gp_log_l.Kxoxo(self.x_sc)
        inv_L_tl = np.linalg.inv(np.linalg.cholesky(Kxcxc))

        h_l, w_l = self.gp_l.K.params
        w_l = np.array([w_l])
        h_tl, w_tl = self.gp_log_l.K.params
        w_tl = np.array([w_tl])

        V_Z = bq_c.Z_var(
            x_sc, alpha_l, inv_L_tl,
            h_l, w_l, h_tl, w_tl,
            self.x_mean, self.x_cov)

        return V_Z

    def expected_squared_mean(self, x_a):
        if np.abs((x_a - self.x_c) < 1e-3).any():
            return self.Z_mean() ** 2

        x_s = self.x
        x_c = self.x_c

        ns, = x_s.shape

        # include new x_a
        x_sa = np.concatenate([x_s, x_a])
        x_sca = np.concatenate([x_c, x_a])

        tl_a = self.gp_log_l.mean(x_a)
        tl_s = self.log_l

        l_a = self.gp_l.mean(x_a)
        l_sc = self.l

        # update gp over log(l)
        gp_log_la = self.gp_log_l.copy()
        gp_log_la.x = x_sa
        gp_log_la.y = np.concatenate([tl_s, tl_a])

        # update gp over l
        gp_la = self.gp_l.copy()
        gp_la.x = x_sca
        gp_la.y = np.concatenate([l_sc, l_a])

        # exp(log(l_c)) (not exp(log(l_s))) will probably change with
        # the addition of x_a. We can't recompute them (because we
        # don't know l_a) but we can add noise to the diagonal of the
        # covariance matrix to allow room for error for the x_c
        # closest to x_a
        K_l = gp_la.Kxx
        bq_c.improve_covariance_conditioning(
            K_l, idx=np.array([x_sca.shape[0] - 1], dtype=DTYPE))

        try:
            inv_K_l = gp_la.inv_Kxx
        except np.linalg.LinAlgError:
            return self.mean() ** 2

        # compute expected transformed mean
        tm_a = float(self.gp_log_l.mean(x_a))

        # compute expected transformed covariance
        tC_a = float(self.gp_log_l.cov(x_a))

        expected_sqd_mean = bq_c.expected_squared_mean(
            x_sca[:, None], l_sc,
            inv_K_l,
            tm_a, tC_a,
            gp_la.K.h, np.array([gp_la.K.w]),
            self.x_mean, self.x_cov)

        if np.isnan(expected_sqd_mean) or expected_sqd_mean < 0:
            raise RuntimeError(
                "invalid expected squared mean: %s" % expected_sqd_mean)

        return expected_sqd_mean

    def expected_Z_var(self, x_a):
        mean_second_moment = self.Z_mean() ** 2 + self.Z_var()
        expected_sqd_mean = self.expected_squared_mean(x_a)
        expected_var = mean_second_moment - expected_sqd_mean
        return expected_var

    def plot_gp_log_l(self, ax, f_l=None, xmin=None, xmax=None):
        x_s = self.x
        l_s = self.log_l

        if xmin is None:
            xmin = x_s.min()
        if xmax is None:
            xmax = x_s.max()

        x = np.linspace(xmin, xmax, 1000)
        l_mean = self.gp_log_l.mean(x)
        l_var = 1.96 * np.sqrt(np.diag(self.gp_log_l.cov(x)))
        lower = l_mean - l_var
        upper = l_mean + l_var

        if f_l is not None:
            l = np.log(f_l(x))
            ax.plot(x, l, 'k-', lw=2)

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(x, l_mean, 'r-', lw=2)
        ax.plot(x_s, l_s, 'ro', markersize=5)
        ax.plot(self.x_c, self.gp_log_l.mean(self.x_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\log(\ell)$")
        ax.set_xlim(xmin, xmax)

    def plot_gp_l(self, ax, f_l=None, xmin=None, xmax=None):
        x_s = self.x
        l_s = self.l

        if xmin is None:
            xmin = x_s.min()
        if xmax is None:
            xmax = x_s.max()

        x = np.linspace(xmin, xmax, 1000)
        l_mean = self.gp_l.mean(x)
        l_var = 1.96 * np.sqrt(np.diag(self.gp_l.cov(x)))
        lower = l_mean - l_var
        upper = l_mean + l_var

        if f_l is not None:
            l = f_l(x)
            ax.plot(x, l, 'k-', lw=2)

        ax.fill_between(x, lower, upper, color='r', alpha=0.2)
        ax.plot(x, l_mean, 'r-', lw=2)
        ax.plot(x_s, l_s, 'ro', markersize=5)
        ax.plot(self.x_c, self.gp_l.mean(self.x_c), 'bs', markersize=4)

        ax.set_title(r"GP over $\exp(\log(\ell))$")
        ax.set_xlim(xmin, xmax)

    def plot_l(self, ax, f_l=None, xmin=None, xmax=None):
        x_s = self.x
        l_s = self.l

        if xmin is None:
            xmin = x_s.min()
        if xmax is None:
            xmax = x_s.max()

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
            x_s, l_s,
            'ro', markersize=5, label="$\ell(x_s)$")
        ax.plot(
            self.x_c, self.l_mean(self.x_c),
            'bs', markersize=4, label="$\exp(m_{\log(\ell)}(x_c))$")

        ax.set_title("Final Approximation")
        ax.set_xlim(xmin, xmax)

        ax.legend(loc=0, fontsize=10)

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

        fig.set_figwidth(12)
        fig.set_figheight(3.5)
        plt.tight_layout()

        return fig, axes
