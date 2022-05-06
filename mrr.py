"""

Micro-ring resonator sensor class

Exposed methods:
    - calc_sensitivity()
    - analyze()

"""


# Standard library packages
from math import e
import numpy as np
from scipy import optimize
from scipy.special import lambertw
from typing import Callable

# Package modules
from .models import Models


class Mrr:
    """
    Micro-ring resonator class

    All lengths are in units of um

    See "Silicon micro-ring resonators" [Bogaerts, 2012] for formulas for Q (20)
    and finesse (21), with Q = (neff * L / lambda) * F. Q is also the total number
    of field oscillations in the ring, over the number of cycles around the ring (F).

    """

    def __init__(self, models: Models, logger: Callable = print):

        # Load class instance input parameters
        self.models: Models = models
        self.logger: Callable = logger

        # Define class instance internal variables
        self.previous_solution: float = -1

        # Define class instance result variables and arrays
        self.α_bend_a: np.ndarray = np.ndarray([])
        self.α_bend_b: np.ndarray = np.ndarray([])
        self.a2_wg: np.ndarray = np.ndarray([])
        self.er: np.ndarray = np.ndarray([])
        self.contrast: np.ndarray = np.ndarray([])
        self.finesse: np.ndarray = np.ndarray([])
        self.fsr: np.ndarray = np.ndarray([])
        self.fwhm: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.gamma_resampled: np.ndarray = np.ndarray([])
        self.max_s: float = 0
        self.max_s_radius: float = 0
        self.n_eff: np.ndarray = np.ndarray([])
        self.q: np.ndarray = np.ndarray([])
        self.results: list = []
        self.s: np.ndarray = np.ndarray([])
        self.r_e: np.ndarray = np.ndarray([])
        self.r_w: np.ndarray = np.ndarray([])
        self.s_e: np.ndarray = np.ndarray([])
        self.s_nr: np.ndarray = np.ndarray([])
        self.t_max: np.ndarray = np.ndarray([])
        self.t_min: np.ndarray = np.ndarray([])
        self.tau: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.u_resampled: np.ndarray = np.ndarray([])

    def _objfun_r_w(self, r: float, u: float, a: float, b: float) -> float:
        """
        Calculate the residual squared with the current solution for Rw,
        using equation (15) in the paper.
        """

        α_bend: float = a * np.exp(-b * r)
        residual: float = 1 - r * (2 * np.pi) * (
                self.α_prop(u=u) + (1 - b * r) * α_bend
        )

        return residual**2

    def _calc_r_e_and_r_w(self, gamma: float) -> tuple[float, float, float, float]:
        """
        Calculate Re(gamma) and Rw(gamma)
        """

        # u corresponding to gamma
        u: float = self.models.u_of_gamma(gamma=gamma)

        # alpha_bend(R) = A*exp(-BR) model parameters @gamma
        α_bend_a, α_bend_b = self.models.calc_α_bend_a_and_b(gamma=gamma)

        # Re
        w: float = lambertw(-e * self.α_prop(u=u) / α_bend_a, k=-1).real
        r_e: float = (1 / α_bend_b) * (1 - w)

        # Rw
        optimization_result = optimize.minimize(
            fun=self._objfun_r_w,
            x0=np.asarray(r_e),
            args=(u, α_bend_a, α_bend_b),
            method="SLSQP",
        )
        r_w: float = optimization_result["x"][0]

        return r_e, r_w, α_bend_a, α_bend_b

    def α_prop(self, u: float) -> float:
        """
        α_prop = α_wg + gamma_fluid*α_fluid
        """

        return self.models.α_wg_of_u(u=u) + (
            self.models.gamma_of_u(u) * self.models.α_fluid
        )

    def calc_α_prop_l(self, r: float, u: float) -> float:
        """
        Propagation loss component of total round-trip losses : α_prop*L
        """

        return self.α_prop(u=u) * (2 * np.pi * r)

    def calc_α_bend_l(self, r: float, u: float) -> float:
        """
        Bending loss component of total round-trip losses: α_bend*L
        """
        return self.models.α_bend(r=r, u=u) * (2 * np.pi * r)

    def calc_α_l(self, r: float, u: float) -> float:
        """
        Total ring round-trip loss factor: αL = (α_prop + α_bend)*L
        """

        return (self.α_prop(u=u) + self.models.α_bend(r=r, u=u)) * (
            2 * np.pi * r
        )

    def calc_a2_wg(self, r: float, u: float) -> float:
        """
        Ring round trio losses: a2 = e**(-α*L)
        """

        return np.e ** -self.calc_α_l(r=r, u=u)

    def calc_s_nr(self, r: float, u: float) -> float:
        """
        Calculate Snr (see paper)
        """
        return (
            (4 * np.pi / self.models.lambda_res)
            * (2 * np.pi * r)
            * self.models.gamma_of_u(u)
            * self.calc_a2_wg(r=r, u=u)
        )

    def calc_s_e(self, r: float, u: float) -> float:
        """
        Calculate Se (see paper)
        """

        return (
            2
            / (3 * np.sqrt(3))
            / (np.sqrt(self.calc_a2_wg(r=r, u=u)) * (1 - self.calc_a2_wg(r=r, u=u)))
        )

    def calc_sensitivity(self, r: float, u: float) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core dimension u
        """

        # Calculate sensitivity
        a2_wg: float = self.calc_a2_wg(r=r, u=u)
        s_nr: float = self.calc_s_nr(r=r, u=u)
        s_e: float = self.calc_s_e(r=r, u=u)
        s: float = s_nr * s_e
        assert s >= 0, "S should not be negative!"

        return s, s_nr, s_e, a2_wg

    def _obj_fun(self, u: float, r: float) -> float:
        """
        Objective function for the non-linear minimization in find_max_sensitivity()
        """

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.u_domain_max)
        u = max(u, self.models.u_domain_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self.calc_sensitivity(r=r, u=u)[0]

        return -s / 1000

    def _find_max_sensitivity(
        self, r: float
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        """
        Calculate maximum sensitivity at radius "r" over all u
        """

        # Determine u search domain extrema
        u_min, u_max = self.models.u_search_domain(r)

        # If this is the first optimization, set the initial guess for u at the
        # maximum value in the domain (at small radii, bending losses are high,
        # the optimal solution will be at high u), else use previous solution.
        u0 = u_max if self.previous_solution == -1 else self.previous_solution

        # Find u that maximizes S at radius r.
        optimization_result = optimize.minimize(
            fun=self._obj_fun,
            x0=np.asarray([u0]),
            bounds=((u_min, u_max),),
            args=(r,),
            method="Powell",
            options={"ftol": 1e-9},
        )
        u_max_s: float = optimization_result["x"][0]

        # Update previous solution
        self.previous_solution = u_max_s

        # Calculate sensitivity and other parameters at the solution
        s, s_nr, s_e, a2_wg = self.calc_sensitivity(r=r, u=u_max_s)

        # Calculate other useful MRR parameters at the solution
        a: float = np.sqrt(a2_wg)
        gamma: float = self.models.gamma_of_u(u_max_s) * 100
        n_eff: float = self.models.n_eff_of_u(u_max_s)
        tau: float = (np.sqrt(3) * a2_wg - np.sqrt(3) - 2 * a) / (a2_wg - 3)
        finesse: float = np.pi * (np.sqrt(tau * a)) / (1 - tau * a)
        q: float = (n_eff * (2 * np.pi * r) / self.models.lambda_res) * finesse
        fwhm: float = self.models.lambda_res / q
        fsr: float = finesse * fwhm
        t_max: float = ((tau + a) / (1 + tau * a)) ** 2
        t_min: float = ((tau - a) / (1 - tau * a)) ** 2
        contrast: float = t_max - t_min
        er: float = 10 * np.log10(t_max / t_min)

        # Return results to calling program
        return (
            s,
            u_max_s,
            gamma,
            s_nr,
            s_e,
            a2_wg,
            tau,
            t_max,
            t_min,
            er,
            contrast,
            n_eff,
            q,
            finesse,
            fwhm,
            fsr,
        )

    def analyze(self):
        """
        Analyse the MRR sensor performance for all radii in the R domain

        :return: None
        """
        # Analyse the sensor performance for all radii in the R domain
        self.results = [self._find_max_sensitivity(r=r) for r in self.models.r]

        # Unpack the analysis results as a function of radius into separate lists, the
        # order must be the same as in the find_max_sensitivity() return statement above
        [
            self.s,
            self.u,
            self.gamma,
            self.s_nr,
            self.s_e,
            self.a2_wg,
            self.tau,
            self.t_max,
            self.t_min,
            self.er,
            self.contrast,
            self.n_eff,
            self.q,
            self.finesse,
            self.fwhm,
            self.fsr,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_s = np.amax(self.s)
        self.max_s_radius = self.models.r[np.argmax(self.s)]

        # Calculate Re(gamma) and Rw(gamma)
        gamma_min: float = list(self.models.modes_data.values())[-1]["gamma"]
        gamma_max: float = list(self.models.modes_data.values())[0]["gamma"]
        self.gamma_resampled = np.linspace(gamma_min, gamma_max, 500)
        self.u_resampled = [self.models.u_of_gamma(g) for g in self.gamma_resampled]
        self.r_e, self.r_w, self.α_bend_a, self.α_bend_b = zip(
            *[self._calc_r_e_and_r_w(gamma=gamma) for gamma in self.gamma_resampled]
        )

        # Console message
        self.logger("MRR sensor analysis done.")
