"""

Linear waveguide sensor class

Exposed methods:
    - analyze()

"""


# Standard library packages
import numpy as np
from scipy import optimize
from typing import Callable

# Package modules
from .models import Models


class Linear:
    """
    Linear waveguide class (straight waveguide of length equal to the ring diameter, 2r)

    All lengths are in units of um
    """

    def __init__(self, models: Models, logger: Callable = print):

        # Load class instance parameter values
        self.models: Models = models
        self.logger: Callable = logger

        # Initialize class instance internal variables
        self.previous_solution: float = -1

        # Declare class instance result variables and arrays
        self.S: np.ndarray = np.ndarray([])
        self.h: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.a2: np.ndarray = np.ndarray([])
        self.results: list = []

    def calc_a2(self, r: float, h: float) -> float:
        """
        Calculate a2
        """

        gamma: float = self.models.gamma(h)
        alpha_prop: float = self.models.alpha_wg + (gamma * self.models.alpha_fluid)
        L: float = 2 * r

        return np.e ** -(alpha_prop * L)

    def _calc_sensitivity(self, r: float, h: float) -> float:
        """
        Calculate sensitivity at radius r (length 2r) for a given core height
        """

        # Calculate sensitivity
        Snr: float = (
            (4 * np.pi / self.models.lambda_res)
            * (2 * r)
            * self.models.gamma(h)
            * self.calc_a2(r=r, h=h)
        )
        assert Snr >= 0, "Snr should not be negative'"

        return Snr

    def _obj_fun(self, h: float, r: float) -> float:
        """
        Objective function used for minimization in find_max_sensitivity()
        """

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        h = min(h, self.models.h_domain_max)
        h = max(h, self.models.h_domain_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self._calc_sensitivity(r=r, h=h)
        return -s / 1000

    def _find_max_sensitivity(self, r: float) -> tuple[float, float, float, float]:
        """
        Calculate maximum sensitivity at r over all h
        """

        # Fetch h domain extrema
        h_min: float = self.models.h_domain_min
        h_max: float = self.models.h_domain_max

        # If this is the first optimization, set the initial guess for h at the
        # maximum value in the domain (at small radii, bending losses are high,
        # the optimal solution will be at high h), else use previous solution.
        h0 = h_max if self.previous_solution == -1 else self.previous_solution

        # Find h that maximizes S at radius R
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

        # Calculate sensitivity at the solution
        max_S = self._calc_sensitivity(r=r, h=h_max_S)

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma(h_max_S) * 100
        a2: float = self.calc_a2(r=r, h=h_max_S)

        # Return results to calling program
        return max_S, h_max_S, gamma, a2

    def analyze(self):
        """
        Analyse the linear waveguide sensor performance for all radii in the R domain

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
            self.a2,
        ] = list(np.asarray(self.results).T)

        # Console message
        self.logger("Linear sensor analysis done.")
