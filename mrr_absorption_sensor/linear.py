"""linear.py

Linear waveguide sensor class

"""
__all__ = ["Linear"]

from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from .constants import CONSTANTS
from .models import Models


class Linear:
    """
    Linear waveguide class (straight waveguide of length equal to the ring diameter, 2r)

    All lengths are in units of um

    Exposed methods:
        - analyze()
        - plot_optimization_results()

    """

    def __init__(self, models: Models, logger: Callable = print):
        """

        Args:
            models (Models): Models object instance
            logger (Callable): console logger
        """

        # Load class instance parameter values
        self.models: Models = models
        self.logger: Callable = logger

        # Initialize class instance internal variables
        self.previous_solution: float = -1

        # Declare class instance result variables and arrays
        self.s: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.wg_a2: np.ndarray = np.ndarray([])
        self.results: list = []

    #
    # Plotting
    #

    def plot_optimization_results(self) -> None:
        """
        PLot linear waveguide optimization results

        Returns: None

        """

        # Create figure
        fig, axs = plt.subplots(5)
        fig.suptitle(
            "Linear waveguide sensor\n"
            f"{self.models.parms.wg.polarization}"
            f", λ = {self.models.parms.wg.lambda_resonance:.3f} μm"
            f", {self.models.parms.wg.v_coord_name} = "
            f"{self.models.parms.wg.v_coord_value:.3f} μm"
        )

        # max{S}
        axs_index: int = 0
        axs[axs_index].loglog(self.models.r, self.s)
        axs[axs_index].set(
            ylabel=r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(100, self.models.plotting_extrema["S_plot_max"]),
            xticklabels=([]),
        )

        # u (h or w) @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.u)
        axs[axs_index].set(
            ylabel=f"{self.models.parms.wg.u_coord_name} (μm)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(
                self.models.plotting_extrema["u_plot_min"],
                self.models.plotting_extrema["u_plot_max"],
            ),
            xticklabels=([]),
        )

        # Gamma_fluid @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.gamma)
        axs[axs_index].set(
            ylabel=r"$\Gamma_{fluide}$ ($\%$)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(0, 100),
            xticklabels=([]),
        )

        # a2 @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.wg_a2)
        axs[axs_index].set(
            ylabel=r"$a^2$",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(0, 1),
            xticklabels=([]),
        )

        # alpha_wg @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(
            self.models.r,
            np.asarray([self.models.α_wg_of_u(u) for u in self.u])
            * CONSTANTS.per_um_to_db_per_cm,
        )
        axs[axs_index].set(
            ylabel=r"α$_{wg}$",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(
                np.floor(self.models.α_wg_model.min * CONSTANTS.per_um_to_db_per_cm),
                np.ceil(self.models.α_wg_model.max * CONSTANTS.per_um_to_db_per_cm),
            ),
        )

        axs[axs_index].set_xlabel("Ring radius (μm)")
        filename: Path = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_LINEAR.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # Explicit None return
        return None

    #
    # Optimization
    #

    def _α_prop(self, u: float) -> float:
        """
        alpha_prop = alpha_wg + gamma_fluid*alpha_fluid

        Args:
            u (float): waveguide core geometry free parameter

        Returns: alpha_prop(u)

        """

        return self.models.α_wg_of_u(u=u) + (
            self.models.gamma_of_u(u) * self.models.α_fluid
        )

    def _calc_wg_a2(self, r: float, u: float) -> float:
        """
        a2 = e**(-alpha_prop * length)

        Args:
            r (float): "radius" (length of waveguide = 2r)
            u (float): waveguide core geometry free parameter

        Returns:

        """

        return np.e ** -(self._α_prop(u=u) * (2 * r))

    def _calc_sensitivity(self, r: float, u: float) -> float:
        """
        Calculate sensitivity at radius r for a given core dimension u

        Args:
            r (float): "radius" (length of waveguide = 2r)
            u (float): waveguide core geometry free parameter

        Returns: sensitivity(r, u)

        """

        # Calculate sensitivity
        s_nr: float = (
            (4 * np.pi / self.models.parms.wg.lambda_resonance)
            * (2 * r)
            * self.models.gamma_of_u(u)
            * self._calc_wg_a2(r=r, u=u)
        )
        assert s_nr >= 0, "Snr should not be negative'"

        return s_nr

    def _obj_fun(self, u: float, r: float) -> float:
        """
        Objective function used for minimization in find_max_sensitivity(u) @ r

        Args:
            u (float): waveguide core geometry free parameter
            r (float): "radius" (length of waveguide = 2r)

        Returns: negative of sensitivity (for maximizing sensitivity)

        """

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.parms.limits.u_max)
        u = max(u, self.models.parms.limits.u_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self._calc_sensitivity(r=r, u=u)

        return -s / 1000

    def _find_max_sensitivity(self, r: float) -> Tuple[float, float, float, float]:
        """
        Calculate maximum sensitivity at radius r over all u

        Args:
            r (float): "radius" (length of waveguide = 2*r)

        Returns: max_s (maximum sensitivity)
                 u_max_s (u @ max_s)
                 gamma (gamma @ max_s)
                 wg_a2 (waveguide loss a2 @ max_s)

        """

        # If this is the first optimization, set the initial guess for u at the
        # maximum value in the domain (at small radii, bending losses are high,
        # the optimal solution will be at high u), else use previous solution.
        u0 = (
            self.models.parms.limits.u_max
            if self.previous_solution == -1
            else self.previous_solution
        )

        # Find u that maximizes S at radius R
        optimization_result: optimize.OptimizeResult = optimize.minimize(
            fun=self._obj_fun,
            x0=np.asarray([u0]),
            bounds=((self.models.parms.limits.u_min, self.models.parms.limits.u_max),),
            args=(r,),
            method=self.models.parms.fit.optimization_method,
            tol=1e-9,
        )
        u_max_s: float = optimization_result.x[0]

        # Update previous solution
        self.previous_solution = u_max_s

        # Calculate sensitivity at the solution
        max_s: float = self._calc_sensitivity(r=r, u=u_max_s)

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(u_max_s) * 100
        wg_a2: float = self._calc_wg_a2(r=r, u=u_max_s)

        # Return results to calling program
        return max_s, u_max_s, gamma, wg_a2

    def analyze(self) -> None:
        """
        Analyse the linear waveguide sensor performance for all radii in the r domain,
        store the results in the Mrr object instance

        Returns: None

        """

        # Analyse the sensor performance for all radii in the R domain
        self.results = [self._find_max_sensitivity(r=r) for r in self.models.r]

        # Unpack the analysis results as a function of radius into separate lists, the
        # order must be the same as in the find_max_sensitivity() return statement above
        [
            self.s,
            self.u,
            self.gamma,
            self.wg_a2,
        ] = list(np.asarray(self.results).T)

        # Console message
        self.logger("Linear sensor analysis done.")

        # Explicit None return
        return None
