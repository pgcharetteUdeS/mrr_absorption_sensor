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
        self.s: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.a2_wg: np.ndarray = np.ndarray([])
        self.results: list = []

    def calc_a2_wg(self, r: float, u: float) -> float:
        """
        Calculate a2
        """

        return np.e ** -(self.models.α_prop(u=u) * (2 * r))

    def _calc_sensitivity(self, r: float, u: float) -> float:
        """
        Calculate sensitivity at radius r (length 2r) for a given core dimension u
        """

        # Calculate sensitivity
        s_nr: float = (
            (4 * np.pi / self.models.lambda_res)
            * (2 * r)
            * self.models.gamma_of_u(u)
            * self.calc_a2_wg(r=r, u=u)
        )
        assert s_nr >= 0, "Snr should not be negative'"

        return s_nr

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
            method=self.models.parameters["optimization_method"],
            options={"ftol": 1e-12},
        )
        u_max_s: float = optimization_result["x"][0]

        # Update previous solution
        self.previous_solution = u_max_s

        # Calculate sensitivity at the solution
        max_s = self._calc_sensitivity(r=r, u=u_max_s)

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(u_max_s) * 100
        a2_wg: float = self.calc_a2_wg(r=r, u=u_max_s)

        # Return results to calling program
        return max_s, u_max_s, gamma, a2_wg

    def analyze(self):
        """
        Analyse the linear waveguide sensor performance for all radii in the R domain

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
            self.a2_wg,
        ] = list(np.asarray(self.results).T)

        # Console message
        self.logger("Linear sensor analysis done.")
