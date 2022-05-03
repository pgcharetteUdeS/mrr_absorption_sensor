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
        self.u: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.a2: np.ndarray = np.ndarray([])
        self.results: list = []

    def calc_a2(self, r: float, u: float) -> float:
        """
        Calculate a2
        """

        gamma: float = self.models.gamma_of_u(u)
        alpha_prop: float = self.models.alpha_wg_of_u(u=u) + (
            gamma * self.models.alpha_fluid
        )
        L: float = 2 * r

        return np.e ** -(alpha_prop * L)

    def _calc_sensitivity(self, r: float, u: float) -> float:
        """
        Calculate sensitivity at radius r (length 2r) for a given core dimension u
        """

        # Calculate sensitivity
        Snr: float = (
            (4 * np.pi / self.models.lambda_res)
            * (2 * r)
            * self.models.gamma_of_u(u)
            * self.calc_a2(r=r, u=u)
        )
        assert Snr >= 0, "Snr should not be negative'"

        return Snr

    def _obj_fun(self, u: float, r: float) -> float:
        """
        Objective function used for minimization in find_max_sensitivity(u) @ r
        """

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.u_domain_max)
        u = max(u, self.models.u_domain_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self._calc_sensitivity(r=r, u=u)
        return -s / 1000

    def _find_max_sensitivity(self, r: float) -> tuple[float, float, float, float]:
        """
        Calculate maximum sensitivity at r over all u
        """

        # Fetch u domain extrema
        u_min: float = self.models.u_domain_min
        u_max: float = self.models.u_domain_max

        # If this is the first optimization, set the initial guess for u at the
        # maximum value in the domain (at small radii, bending losses are high,
        # the optimal solution will be at high u), else use previous solution.
        u0 = u_max if self.previous_solution == -1 else self.previous_solution

        # Find u that maximizes S at radius R
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

        # Calculate sensitivity at the solution
        max_S = self._calc_sensitivity(r=r, u=u_max_S)

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(u_max_S) * 100
        a2: float = self.calc_a2(r=r, u=u_max_S)

        # Return results to calling program
        return max_S, u_max_S, gamma, a2

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
            self.u,
            self.gamma,
            self.a2,
        ] = list(np.asarray(self.results).T)

        # Console message
        self.logger("Linear sensor analysis done.")
