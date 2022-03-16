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
        self.contrast: np.ndarray = np.ndarray([])
        self.Finesse: np.ndarray = np.ndarray([])
        self.FSR: np.ndarray = np.ndarray([])
        self.FWHM: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.gamma_resampled: np.ndarray = np.ndarray([])
        self.h: np.ndarray = np.ndarray([])
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

    def _objfun_Rw(self, R: float, h: float, A: float, B: float) -> float:
        """
        Calculate the residual squared with the current solution for Rw,
        using equation (15) in the paper.
        """

        alpha_bend: float = A * np.exp(-B * R)
        residual: float = 1 - R * (2 * np.pi) * (
            self.alpha_prop(h=h) + (1 - B * R) * alpha_bend
        )

        return residual**2

    def _calc_Re_Rw(self, gamma: float) -> tuple[float, float, float, float]:
        """
        Calculate Re(gamma) and Rw(gamma)
        """

        # h corresponding to gamma
        h: float = self.models.h_of_gamma(gamma=gamma)

        # alpha_bend(R) = A*exp(-BR) model parameters @gamma
        A, B = self.models.calc_A_and_B(gamma=gamma)

        # Re
        W: float = lambertw(-e * self.alpha_prop(h=h) / A, k=-1).real
        Re: float = (1 / B) * (1 - W)

        # Rw
        optimization_result = optimize.minimize(
            fun=self._objfun_Rw,
            x0=np.asarray(Re),
            args=(h, A, B),
            method="SLSQP",
        )
        Rw: float = optimization_result["x"][0]

        return Re, Rw, A, B

    def alpha_prop(self, h: float) -> float:
        """
        Calculate alpha_prop
        """

        return self.models.alpha_wg + (
            self.models.gamma_of_h(h) * self.models.alpha_fluid
        )

    def calc_alpha_prop_L(self, r: float, h: float) -> float:
        """
        Calculate alpha_prop * L component of ring round-trip losses
        """

        return self.alpha_prop(h=h) * (2 * np.pi * r)

    def calc_alpha_bend_L(self, r: float, h: float) -> float:
        """
        Calculate alpha_bend * L component of ring round-trip losses
        """
        return self.models.alpha_bend(r=r, h=h) * (2 * np.pi * r)

    def calc_alpha_L(self, r: float, h: float) -> float:
        """
        Calculate alpha * L total ring round-trip losses
        """

        return (self.alpha_prop(h=h) + self.models.alpha_bend(r=r, h=h)) * (
            2 * np.pi * r
        )

    def calc_a2(self, r: float, h: float) -> float:
        """
        Calculate a2 = e**(-alpha * L)
        """

        return np.e ** -self.calc_alpha_L(r=r, h=h)

    def calc_Snr(self, r: float, h: float) -> float:
        """
        Calculate Snr
        """
        return (
            (4 * np.pi / self.models.lambda_res)
            * (2 * np.pi * r)
            * self.models.gamma_of_h(h)
            * self.calc_a2(r=r, h=h)
        )

    def calc_Se(self, r: float, h: float) -> float:
        """
        Calculate Se
        """

        return (
            2
            / (3 * np.sqrt(3))
            / (np.sqrt(self.calc_a2(r=r, h=h)) * (1 - self.calc_a2(r=r, h=h)))
        )

    def calc_sensitivity(self, r: float, h: float) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core height
        """

        # Calculate sensitivity
        a2: float = self.calc_a2(r=r, h=h)
        Snr: float = self.calc_Snr(r=r, h=h)
        Se: float = self.calc_Se(r=r, h=h)
        S: float = Snr * Se
        assert S >= 0, "S should not be negative!"

        return S, Snr, Se, a2

    def _obj_fun(self, h: float, r: float) -> float:
        """
        Objective function for the non-linear minimization in find_max_sensitivity()
        """

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        h = min(h, self.models.h_domain_max)
        h = max(h, self.models.h_domain_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self.calc_sensitivity(r=r, h=h)[0]

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
    ]:
        """
        Calculate maximum sensitivity at radius "r" over all h
        """

        # Determine h search domain extrema
        h_min, h_max = self.models.h_search_domain(r)

        # If this is the first optimization, set the initial guess for h at the
        # maximum value in the domain (at small radii, bending losses are high,
        # the optimal solution will be at high h), else use previous solution.
        h0 = h_max if self.previous_solution == -1 else self.previous_solution

        # Find h that maximizes S at radius r.
        optimization_result = optimize.minimize(
            fun=self._obj_fun,
            x0=np.asarray([h0]),
            bounds=((h_min, h_max),),
            args=(r,),
            method="Powell",
            options={"ftol": 1e-9},
        )
        h_max_S: float = optimization_result["x"][0]

        # Update previous solution
        self.previous_solution = h_max_S

        # Calculate sensitivity and other parameters at the solution
        S, Snr, Se, a2 = self.calc_sensitivity(r=r, h=h_max_S)

        # Calculate other useful MRR parameters at the solution
        a: float = np.sqrt(a2)
        gamma: float = self.models.gamma_of_h(h_max_S) * 100
        neff: float = self.models.neff_of_h(h_max_S)
        tau: float = (np.sqrt(3) * a2 - np.sqrt(3) - 2 * a) / (a2 - 3)
        finesse: float = np.pi * (np.sqrt(tau * a)) / (1 - tau * a)
        Q: float = (neff * (2 * np.pi * r) / self.models.lambda_res) * finesse
        FWHM: float = self.models.lambda_res / Q
        FSR: float = finesse * FWHM
        contrast: float = ((tau + a) / (1 + tau * a)) ** 2 - (
            (tau - a) / (1 - tau * a)
        ) ** 2

        # Return results to calling program
        return (
            S,
            h_max_S,
            gamma,
            Snr,
            Se,
            a2,
            tau,
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
            self.h,
            self.gamma,
            self.Snr,
            self.Se,
            self.a2,
            self.tau,
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
        self.Re, self.Rw, self.A, self.B = zip(
            *[self._calc_Re_Rw(gamma=gamma) for gamma in self.gamma_resampled]
        )

        # Console message
        self.logger("MRR sensor analysis done.")
