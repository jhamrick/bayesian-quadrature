import numpy as np
import logging
import matplotlib.pyplot as plt
from gp import GP, GaussianKernel

from . import bq_c

logger = logging.getLogger("bayesian_quadrature")

DTYPE = np.dtype('float64')
EPS = np.finfo(DTYPE).eps


class BQ(object):
    """Estimate a likelihood function, S(y|x) using Gaussian Process
    regressions, as in Osborne et al. (2012):

    1) Estimate S using a GP
    2) Estimate log(S) using second GP
    3) Estimate delta_C using a third GP

    References
    ----------
    Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
        Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
        Model Evidence Using Bayesian Quadrature. *Advances in Neural
        Information Processing Systems*, 25.

    """

    def __init__(self, R, S,
                 gamma, ntry, n_candidate,
                 R_mean, R_var,
                 h=None, w=None, s=None):

        # save the given parameters
        self.gamma = float(gamma)
        self.ntry = int(ntry)
        self.n_candidate = int(n_candidate)
        self.R_mean = np.array([R_mean], dtype=DTYPE)
        self.R_cov = np.array([[R_var]], dtype=DTYPE)

        # default kernel parameter values
        self.default_params = dict(h=h, w=w, s=s)

        self.R = np.array(R, dtype=DTYPE, copy=True)
        self.S = np.array(S, dtype=DTYPE, copy=True)

        if self.R.ndim > 1:
            raise ValueError("invalid number of dimensions for R")
        if self.S.ndim > 1:
            raise ValueError("invalid number of dimensions for S")
        if self.R.shape != self.S.shape:
            raise ValueError("shape mismatch for R and S")

        self.log_S = self.log_transform(self.S)
        self.n_sample = self.R.size

        self.improve_covariance_conditioning = False

    def log_transform(self, x):
        return np.log(x + self.gamma) - np.log(self.gamma)

    def inv_log_transform(self, xt):
        return np.exp(xt + np.log(self.gamma)) - self.gamma

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
        w = self.gp_log_S.K.w
        xmin = self.R.min() - w
        xmax = self.R.max() + w
        Rc_all = np.random.uniform(xmin, xmax, self.n_candidate)

        # make sure they don't overlap with points we already have
        Rc = []
        for i in xrange(self.n_candidate):
            if (np.abs(Rc_all[i] - np.array(Rc)) >= 1e-4).all():
                if (np.abs(Rc_all[i] - self.R) >= 1e-4).all():
                    Rc.append(Rc_all[i])
        Rc = np.array(Rc)
        return Rc

    def _fit_log_S(self):
        logger.info("Fitting parameters for GP over log(S)")
        self.gp_log_S = self._fit_gp(self.R, self.log_S)

    def _fit_S(self):
        self.Rc = self._choose_candidates()
        self.Rsc = np.concatenate([self.R, self.Rc], axis=0)

        m_log_S = self.gp_log_S.mean(self.Rsc)
        v_log_S = np.diag(self.gp_log_S.cov(self.Rsc))
        self.Sc = self.inv_log_transform(m_log_S + 0.5 * v_log_S)

        logger.info("Fitting parameters for GP over exp(log(S))")
        self.gp_S = self._fit_gp(self.Rsc, self.Sc)

    def fit(self):
        """Run the GP regressions to fit the likelihood function.

        References
        ----------
        Osborne, M. A., Duvenaud, D., Garnett, R., Rasmussen, C. E.,
            Roberts, S. J., & Ghahramani, Z. (2012). Active Learning of
            Model Evidence Using Bayesian Quadrature. *Advances in Neural
            Information Processing Systems*, 25.

        """

        logger.info("Fitting likelihood")

        self._fit_log_S()
        self._fit_S()

    def S_mean(self, R):
        # the estimated mean of S
        S_mean = self.gp_S.mean(R)
        return S_mean

    def S_var(self, R):
        # the estimated variance of S
        v_log_S = np.diag(self.gp_log_S.cov(R))
        m_S = self.gp_S.mean(R)
        S_var = v_log_S * m_S ** 2
        return S_var

    def Z_mean(self):

        # values for the GP over l(x)
        x_s = self.R[:, None]
        alpha_l = self.gp_S.inv_Kxx_y
        h_s, w_s = self.gp_S.K.params
        w_s = np.array([w_s])

        # values for the GP of Delta(x)
        x_sc = self.gp_Dc.x[:, None]
        alpha_del = self.gp_Dc.inv_Kxx_y
        h_dc, w_dc = self.gp_Dc.K.params
        w_dc = np.array([w_dc])

        m_Z = bq_c.Z_mean(
            x_s, x_sc, alpha_l, alpha_del,
            h_s, w_s, h_dc, w_dc,
            self.R_mean, self.R_cov, self.gamma)

        return m_Z

    def _Z_var_and_eps(self):
        # values for the GPs over l(x) and log(l(x))
        x_s = self.R

        alpha_l = self.gp_S.inv_Kxx_y
        alpha_tl = self.gp_log_S.inv_Kxx_y
        inv_L_tl = self.gp_log_S.inv_Lxx
        inv_K_tl = self.gp_log_S.inv_Kxx

        h_l, w_l = self.gp_S.K.params
        w_l = np.array([w_l])
        h_tl, w_tl = self.gp_log_S.K.params
        w_tl = np.array([w_tl])

        dK_tl_dw = self.gp_log_S.K.dK_dw(x_s, x_s)[..., None]
        Cw = np.array([[self.Cw(self.gp_log_S)]])

        V_Z, V_Z_eps = bq_c.Z_var(
            x_s[:, None], alpha_l, alpha_tl,
            inv_L_tl, inv_K_tl, dK_tl_dw, Cw,
            h_l, w_l, h_tl, w_tl,
            self.R_mean, self.R_cov, self.gamma)

        return V_Z, V_Z_eps

    def Z_var(self):
        V_Z, V_Z_eps = self._Z_var_and_eps()
        return V_Z + V_Z_eps

    def dm_dw(self, x):
        """Compute the partial derivative of a GP mean with respect to
        w, the input scale parameter.

        """
        dm_dtheta = self.gp_log_S.dm_dtheta(x)
        # XXX: fix this slicing
        dm_dw = dm_dtheta[1]
        return dm_dw

    def Cw(self, gp):
        """The variances of our posteriors over our input scale. We assume the
        covariance matrix has zero off-diagonal elements; the posterior
        is spherical.

        """
        # H_theta is the diagonal of the hessian of the likelihood of
        # the GP over the log-likelihood with respect to its log input
        # scale.
        H_theta = gp.d2lh_dtheta2
        # XXX: fix this slicing
        Cw = -1. / H_theta[1, 1]
        return Cw

    def expected_squared_mean(self, x_a):
        if np.abs((x_a - self.Rc) < 1e-3).any():
            return self.Z_mean() ** 2

        x_s = self.R
        x_c = self.Rc

        ns, = x_s.shape

        # include new x_a
        x_sa = np.concatenate([x_s, x_a])

        l_a = self.gp_S.mean(x_a)
        l_s = self.S

        tl_a = self.gp_log_S.mean(x_a)
        tl_s = self.log_S

        # update gp over S
        gp_Sa = self.gp_S.copy()
        gp_Sa.x = x_sa
        gp_Sa.y = np.concatenate([l_s, l_a])

        # update gp over log(S)
        gp_log_Sa = self.gp_log_S.copy()
        gp_log_Sa.x = x_sa
        gp_log_Sa.y = np.concatenate([tl_s, tl_a])

        # add jitter to the covariance matrix where our new point is,
        # because if it's close to other x_s then it will cause problems
        if self.improve_covariance_conditioning:
            idx = np.array([ns], dtype=DTYPE)

            gp_Sa._memoized = {}
            bq_c.improve_covariance_conditioning(gp_Sa.Kxx, idx=idx)

            gp_log_Sa._memoized = {}
            bq_c.improve_covariance_conditioning(gp_log_Sa.Kxx, idx=idx)

        try:
            inv_K_l = gp_Sa.inv_Kxx
        except np.linalg.LinAlgError:
            return self.Z_mean() ** 2

        # update gp over delta
        del_sc = self.Dc
        if (np.abs(x_c - x_a) < 1e-3).any():
            x_sca = x_c.copy()
            del_sca = del_sc.copy()
            gp_Dca = self.gp_Dc.copy()

        else:
            x_sca = np.concatenate([x_c, x_a])
            del_sca = np.concatenate([del_sc, [0]])
            gp_Dca = self.gp_Dc.copy()
            gp_Dca.x = x_sca
            gp_Dca.y = del_sca

            # the delta_c (not delta_s) will probably change with
            # the addition of x_a. We can't recompute them
            # (because we don't know l_a) but we can add noise
            # to the diagonal of the covariance matrix to allow room
            # for error for the x_c closest to x_a
            gp_Dca._memoized = {}
            K_del = gp_Dca.Kxx
            bq_c.improve_covariance_conditioning(
                K_del, idx=np.array([x_sca.shape[0] - 1], dtype=DTYPE))

        try:
            alpha_del = gp_Dca.inv_Kxx_y
        except np.linalg.LinAlgError:
            return self.mean() ** 2

        # compute expected transformed mean
        tm_a = float(self.gp_log_S.mean(x_a))

        # compute expected transformed covariance
        dm_dw = self.dm_dw(x_a)
        Cw = self.Cw(gp_log_Sa)
        C_a = float(self.gp_log_S.cov(x_a))
        tC_a = C_a + float(dm_dw ** 2 * Cw)

        expected_sqd_mean = bq_c.expected_squared_mean(
            x_sa[:, None], x_sca[:, None], l_s,
            alpha_del,
            inv_K_l,
            tm_a, tC_a,
            gp_Sa.K.h, np.array([gp_Sa.K.w]),
            gp_Dca.K.h, np.array([gp_Dca.K.w]),
            self.R_mean, self.R_cov, self.gamma)

        if np.isnan(expected_sqd_mean) or expected_sqd_mean < 0:
            raise RuntimeError(
                "invalid expected squared mean: %s" % expected_sqd_mean)

        return expected_sqd_mean

    def expected_Z_var(self, x_a):
        mean_second_moment = self.Z_mean() ** 2 + self.Z_var()
        expected_sqd_mean = self.expected_squared_mean(x_a)
        expected_var = mean_second_moment - expected_sqd_mean
        return expected_var

    def plot_gp_log_S(self, ax, f_S=None, xmin=None, xmax=None):
        Ri = self.R
        Si = self.log_S

        if xmin is None:
            xmin = Ri.min()
        if xmax is None:
            xmax = Ri.max()

        R = np.linspace(xmin, xmax, 1000)
        S_mean = self.gp_log_S.mean(R)
        S_var = 1.96 * np.sqrt(np.diag(self.gp_log_S.cov(R)))
        lower = S_mean - S_var
        upper = S_mean + S_var

        if f_S is not None:
            S = self.log_transform(f_S(R))
            ax.plot(R, S, 'k-', lw=2)

        ax.fill_between(R, lower, upper, color='r', alpha=0.2)
        ax.plot(R, S_mean, 'r-', lw=2)
        ax.plot(Ri, Si, 'ro', markersize=5)
        ax.plot(self.Rc, self.gp_log_S.mean(self.Rc), 'bs', markersize=4)

        ax.set_xlabel("R")
        ax.set_ylabel("log(S)")
        ax.set_title("GP over log(S)")
        ax.set_xlim(xmin, xmax)

    def plot_gp_S(self, ax, f_S=None, xmin=None, xmax=None):
        Ri = self.R
        Si = self.S

        if xmin is None:
            xmin = Ri.min()
        if xmax is None:
            xmax = Ri.max()

        R = np.linspace(xmin, xmax, 1000)
        S_mean = self.gp_S.mean(R)
        S_var = 1.96 * np.sqrt(np.diag(self.gp_S.cov(R)))
        lower = S_mean - S_var
        upper = S_mean + S_var

        if f_S is not None:
            S = f_S(R)
            ax.plot(R, S, 'k-', lw=2)

        ax.fill_between(R, lower, upper, color='r', alpha=0.2)
        ax.plot(R, S_mean, 'r-', lw=2)
        ax.plot(Ri, Si, 'ro', markersize=5)
        ax.plot(self.Rc, self.gp_S.mean(self.Rc), 'bs', markersize=4)

        ax.set_xlabel("R")
        ax.set_ylabel("S")
        ax.set_title("GP over exp(log(S))")
        ax.set_xlim(xmin, xmax)

    def plot_S(self, ax, f_S=None, xmin=None, xmax=None):
        Ri = self.R
        Si = self.S

        if xmin is None:
            xmin = Ri.min()
        if xmax is None:
            xmax = Ri.max()

        R = np.linspace(xmin, xmax, 1000)
        S_mean = self.S_mean(R)
        S_var = 1.96 * np.sqrt(self.S_var(R))
        lower = S_mean - S_var
        upper = S_mean + S_var

        if f_S is not None:
            S = f_S(R)
            ax.plot(R, S, 'k-', lw=2, label="truth")

        ax.fill_between(R, lower, upper, color='r', alpha=0.2)
        ax.plot(
            R, S_mean,
            'r-', lw=2, label="approx.")
        ax.plot(
            Ri, Si,
            'ro', markersize=5, label="observations")
        ax.plot(
            self.Rc, self.S_mean(self.Rc),
            'bs', markersize=4, label="candidates")

        ax.set_xlabel("R")
        ax.set_ylabel("S")
        ax.set_title("Final Approximation")
        ax.set_xlim(xmin, xmax)

        ax.legend(loc=0, fontsize=10)

    def plot(self, f_S=None, xmin=None, xmax=None):
        fig, axes = plt.subplots(1, 3)

        self.plot_gp_log_S(axes[0], f_S=f_S, xmin=xmin, xmax=xmax)
        self.plot_gp_S(axes[1], f_S=f_S, xmin=xmin, xmax=xmax)
        self.plot_S(axes[2], f_S=f_S, xmin=xmin, xmax=xmax)

        ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes[1:]])
        ymin = min(ymins)
        ymax = max(ymaxs)
        for ax in axes[1:]:
            ax.set_ylim(ymin, ymax)

        fig.set_figwidth(10)
        fig.set_figheight(3)
        plt.tight_layout()

        return fig, axes
