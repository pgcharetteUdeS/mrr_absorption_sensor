# Standard library packages
import numpy as np
from scipy import optimize
from typing import Callable

# Package modules
from .modeling import Models

#######################################################################################
#
# Micro-ring resonator class
#
#######################################################################################


class Mrr:
    """
    Micro-ring resonator class

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
        self.S: np.ndarray = np.ndarray([])
        self.h: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.Snr: np.ndarray = np.ndarray([])
        self.Se: np.ndarray = np.ndarray([])
        self.a2: np.ndarray = np.ndarray([])
        self.tau: np.ndarray = np.ndarray([])
        self.neff: np.ndarray = np.ndarray([])
        self.Q: np.ndarray = np.ndarray([])
        self.Finesse: np.ndarray = np.ndarray([])
        self.FWHM: np.ndarray = np.ndarray([])
        self.FSR: np.ndarray = np.ndarray([])
        self.max_S: float = 0
        self.max_S_radius: float = 0
        self.results: list = []

    def calc_sensitivity(self, r: float, h: float) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core height
        """

        # Calculate interpolated value of gamma
        gamma: float = self.models.gamma(h)

        # Calculate ring round-trip losses
        alpha_prop: float = self.models.alpha_wg + (gamma * self.models.alpha_fluid)
        L: float = 2 * np.pi * r
        round_trip_losses: float = (alpha_prop + self.models.alpha_bend(r=r, h=h)) * L
        a2: float = np.e ** -round_trip_losses

        # Calculate sensitivity
        Snr: float = (4 * np.pi / self.models.lambda_res) * L * gamma * a2
        Se: float = 2 / (3 * np.sqrt(3)) / (np.sqrt(a2) * (1 - a2))
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

    def find_max_sensitivity(
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
        gamma: float = self.models.gamma(h_max_S) * 100
        neff: float = self.models.neff(h_max_S)
        tau: float = (np.sqrt(3) * a2 - np.sqrt(3) - 2 * np.sqrt(a2)) / (a2 - 3)
        finesse: float = np.pi * (np.sqrt(tau * np.sqrt(a2))) / (1 - tau * np.sqrt(a2))
        Q: float = (neff * (2 * np.pi * r) / self.models.lambda_res) * finesse
        FWHM: float = self.models.lambda_res / Q
        FSR: float = finesse * FWHM

        # Return results to calling program
        return S, h_max_S, gamma, Snr, Se, a2, tau, neff, Q, finesse, FWHM, FSR

    def analyze(self):
        # Analyse the sensor performance for all radii in the R domain
        self.results = [self.find_max_sensitivity(r=r) for r in self.models.R]

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
            self.neff,
            self.Q,
            self.Finesse,
            self.FWHM,
            self.FSR,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_S = np.amax(self.S)
        self.max_S_radius = self.models.R[np.argmax(self.S)]

        # Console message
        self.logger("MRR sensor analysis done.")
