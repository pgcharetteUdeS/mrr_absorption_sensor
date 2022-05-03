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
        self.A: np.ndarray = np.ndarray([])
        self.a2: np.ndarray = np.ndarray([])
        self.B: np.ndarray = np.ndarray([])
        self.T_max: np.ndarray = np.ndarray([])
        self.T_min: np.ndarray = np.ndarray([])
        self.ER: np.ndarray = np.ndarray([])
        self.contrast: np.ndarray = np.ndarray([])
        self.Finesse: np.ndarray = np.ndarray([])
        self.FSR: np.ndarray = np.ndarray([])
        self.FWHM: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.gamma_resampled: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.u_resampled: np.ndarray = np.ndarray([])
        self.max_S: float = 0
        self.max_S_radius: float = 0
        self.neff: np.ndarray = np.ndarray([])
        self.Q: np.ndarray = np.ndarray([])
        self.results: list = []
        self.S: np.ndarray = np.ndarray([])
        self.Re: np.ndarray = np.ndarray([])
        self.Rw: np.ndarray = np.ndarray([])
        self.Se: np.ndarray = np.ndarray([])
        self.Snr: np.ndarray = np.ndarray([])
        self.tau: np.ndarray = np.ndarray([])

    def _objfun_Rw(self, r: float, u: float, A: float, B: float) -> float:
        """
        Calculate the residual squared with the current solution for Rw,
        using equation (15) in the paper.
        """

        alpha_bend: float = A * np.exp(-B * r)
        residual: float = 1 - r * (2 * np.pi) * (
            self.alpha_prop(u=u) + (1 - B * r) * alpha_bend
        )

        return residual**2

    def _calc_Re_Rw(self, gamma: float) -> tuple[float, float, float, float]:
        """
        Calculate Re(gamma) and Rw(gamma)
        """

        # u corresponding to gamma
        u: float = self.models.u_of_gamma(gamma=gamma)

        # alpha_bend(R) = A*exp(-BR) model parameters @gamma
        A, B = self.models.calc_A_and_B(gamma=gamma)

        # Re
        W: float = lambertw(-e * self.alpha_prop(u=u) / A, k=-1).real
        Re: float = (1 / B) * (1 - W)

        # Rw
        optimization_result = optimize.minimize(
            fun=self._objfun_Rw,
            x0=np.asarray(Re),
            args=(u, A, B),
            method="SLSQP",
        )
        Rw: float = optimization_result["x"][0]

        return Re, Rw, A, B

    def alpha_prop(self, u: float) -> float:
        """
        α_prop = α_wg + gamma_fluid*α_fluid
        """

        return self.models.alpha_wg_of_u(u=u) + (
            self.models.gamma_of_u(u) * self.models.alpha_fluid
        )

    def calc_alpha_prop_L(self, r: float, u: float) -> float:
        """
        Propagation loss component of total round-trip losses : α_prop*L
        """

        return self.alpha_prop(u=u) * (2 * np.pi * r)

    def calc_alpha_bend_L(self, r: float, u: float) -> float:
        """
        Bending loss component of total round-trip losses: α_bend*L
        """
        return self.models.alpha_bend(r=r, u=u) * (2 * np.pi * r)

    def calc_alpha_L(self, r: float, u: float) -> float:
        """
        Total ring round-trip loss factor: αL = (α_prop + α_bend)*L
        """

        return (self.alpha_prop(u=u) + self.models.alpha_bend(r=r, u=u)) * (
            2 * np.pi * r
        )

    def calc_a2(self, r: float, u: float) -> float:
        """
        Ring round trio losses: a2 = e**(-α*L)
        """

        return np.e ** -self.calc_alpha_L(r=r, u=u)

    def calc_Snr(self, r: float, u: float) -> float:
        """
        Calculate Snr (see paper)
        """
        return (
            (4 * np.pi / self.models.lambda_res)
            * (2 * np.pi * r)
            * self.models.gamma_of_u(u)
            * self.calc_a2(r=r, u=u)
        )

    def calc_Se(self, r: float, u: float) -> float:
        """
        Calculate Se (see paper)
        """

        return (
            2
            / (3 * np.sqrt(3))
            / (np.sqrt(self.calc_a2(r=r, u=u)) * (1 - self.calc_a2(r=r, u=u)))
        )

    def calc_sensitivity(self, r: float, u: float) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core dimension u
        """

        # Calculate sensitivity
        a2: float = self.calc_a2(r=r, u=u)
        Snr: float = self.calc_Snr(r=r, u=u)
        Se: float = self.calc_Se(r=r, u=u)
        S: float = Snr * Se
        assert S >= 0, "S should not be negative!"

        return S, Snr, Se, a2

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
        u_max_S: float = optimization_result["x"][0]

        # Update previous solution
        self.previous_solution = u_max_S

        # Calculate sensitivity and other parameters at the solution
        S, Snr, Se, a2 = self.calc_sensitivity(r=r, u=u_max_S)

        # Calculate other useful MRR parameters at the solution
        a: float = np.sqrt(a2)
        gamma: float = self.models.gamma_of_u(u_max_S) * 100
        neff: float = self.models.neff_of_u(u_max_S)
        tau: float = (np.sqrt(3) * a2 - np.sqrt(3) - 2 * a) / (a2 - 3)
        finesse: float = np.pi * (np.sqrt(tau * a)) / (1 - tau * a)
        Q: float = (neff * (2 * np.pi * r) / self.models.lambda_res) * finesse
        FWHM: float = self.models.lambda_res / Q
        FSR: float = finesse * FWHM
        T_max: float = ((tau + a) / (1 + tau * a)) ** 2
        T_min: float = ((tau - a) / (1 - tau * a)) ** 2
        contrast: float = T_max - T_min
        ER: float = 10 * np.log10(T_max / T_min)

        # Return results to calling program
        return (
            S,
            u_max_S,
            gamma,
            Snr,
            Se,
            a2,
            tau,
            T_max,
            T_min,
            ER,
            contrast,
            neff,
            Q,
            finesse,
            FWHM,
            FSR,
        )

    def analyze(self):
        """
        Analyse the MRR sensor performance for all radii in the R domain

        :return: None
        """
        # Analyse the sensor performance for all radii in the R domain
        self.results = [self._find_max_sensitivity(r=r) for r in self.models.R]

        # Unpack the analysis results as a function of radius into separate lists, the
        # order must be the same as in the find_max_sensitivity() return statement above
        [
            self.S,
            self.u,
            self.gamma,
            self.Snr,
            self.Se,
            self.a2,
            self.tau,
            self.T_max,
            self.T_min,
            self.ER,
            self.contrast,
            self.neff,
            self.Q,
            self.Finesse,
            self.FWHM,
            self.FSR,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_S = np.amax(self.S)
        self.max_S_radius = self.models.R[np.argmax(self.S)]

        # Calculate Re(gamma) and Rw(gamma)
        gamma_min: float = list(self.models.modes_data.values())[-1]["gamma"]
        gamma_max: float = list(self.models.modes_data.values())[0]["gamma"]
        self.gamma_resampled = np.linspace(gamma_min, gamma_max, 500)
        self.u_resampled = [self.models.u_of_gamma(g) for g in self.gamma_resampled]
        self.Re, self.Rw, self.A, self.B = zip(
            *[self._calc_Re_Rw(gamma=gamma) for gamma in self.gamma_resampled]
        )

        # Console message
        self.logger("MRR sensor analysis done.")
