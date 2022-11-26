"""models.py

Models class containing data and methods for interpolation/calculation of the
problem parameters (waveguide propagation and bending losses, mode effective index
and gamma, etc.).


"""
__all__ = ["Models"]

from pathlib import Path
from typing import Callable, cast, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection, PathCollection
from matplotlib.widgets import Button, CheckButtons
from numpy.polynomial import Polynomial
from rich import print
from scipy import interpolate, optimize
from sympy.core.function import FunctionClass
from scipy.linalg import lstsq
from sympy import functions, lambdify, symbols

from .constants import CONSTANTS, InputParameters, PolyModel1D


class Models:
    """
    Models class for polynomial interpolation: gamma(u), neffs(u), α_wg(u), α_bend(r, u)

    All lengths are in units of um

    Exposed methods:
        - α_bend(r)
        - α_wg_of_u()
        - calculate_plotting_extrema()
        - gamma_of_u()
        - n_eff_of_u()
        - u_of_gamma()
        - u_search_domain()

    Note:
    1) To accommodate either fixed core width or fixed core height optimization
       problems, the waveguide core geometry dimensions are labeled "u" and "v",
       with "u" being the variable components and "v" the fixed component.
    2) The model for alpha_bend(r, u) is hardcoded in fit_alpha_bend_model()
       but the code is structured in such a way that it is relatively easy to change,
       see the "USER-DEFINABLE MODEL-SPECIFIC SECTION" code section.

    """

    def __init__(
        self,
        parms: InputParameters,
        filename_path: Path,
        logger: Callable = print,
    ):
        """

        Args:
            parms (InputParameters): problem input parameters
            filename_path (Path): output file Path
            logger (Callable): console logger
        """

        # Load class instance variables
        self.parms: InputParameters = parms
        self.filename_path: Path = filename_path
        self.logger: Callable = logger

        # Initialize other class parameters
        self.plotting_extrema: dict = {}

        # Define the array of radii to be analyzed (R domain)
        self.r: np.ndarray = np.logspace(
            start=np.log10(self.parms.limits.r_min),
            stop=np.log10(self.parms.limits.r_max),
            num=int(
                (np.log10(self.parms.limits.r_max) - np.log10(self.parms.limits.r_min))
                * self.parms.limits.r_samples_per_decade
            ),
            base=10,
        )
        if np.size(self.r) == 0:
            raise ValueError("No radii to analyze!")

        # Define propagation losses in the fluid medium (1/um)
        self.α_fluid: float = (
            4 * np.pi / self.parms.wg.lambda_resonance
        ) * self.parms.wg.ni_op_point

        # Parse and validate the mode solver data loaded from the .toml file
        self.u_data: np.ndarray = np.asarray([])
        self.r_alpha_bend_data: np.ndarray = np.asarray([])
        self.ln_alpha_bend_data: np.ndarray = np.asarray([])
        self.r_α_bend_data_min: float = 0
        self.r_α_bend_min_interp: interpolate.interp1d = interpolate.interp1d(
            [0, 1], [0, 1]
        )
        self.r_α_bend_data_max: float = 0
        self.r_α_bend_max_interp: interpolate.interp1d = interpolate.interp1d(
            [0, 1], [0, 1]
        )
        self.α_bend_data_min: float = 0
        self._parse_bending_loss_mode_solver_data()

        # Calculate alpha_wg(u) values and load them into the parms geom dict
        if self.parms.wg.v_coord_name == "w":
            for key, value in self.parms.geom.items():
                self.parms.geom[key].alpha_wg = self._calc_alpha_db_per_cm(
                    n_eff=value.neff,
                    height=value.u,
                    width=self.parms.wg.v_coord_value,
                )
        else:
            for key, value in self.parms.geom.items():
                self.parms.geom[key].alpha_wg = self._calc_alpha_db_per_cm(
                    n_eff=value.neff,
                    height=self.parms.wg.v_coord_value,
                    width=value.u,
                )

        # Fit gamma(h), h(gamma), and neff(h) 1D poly models to the mode solver data
        self.α_wg_model: PolyModel1D
        self.gamma_model: PolyModel1D
        self.u_model: PolyModel1D
        self.n_eff_model: PolyModel1D
        self._fit_1d_models()
        self.α_wg_db_per_cm: float = self.α_wg_of_u() * CONSTANTS.per_um_to_db_per_cm

        # Check that the bending loss mode solver data covers the required h & R ranges
        α_prop_min: float = self.α_wg_of_u() + (
            self.gamma_of_u(self.parms.limits.u_max) * self.α_fluid
        )
        if self.α_bend_data_min > α_prop_min / 100:
            self.logger(
                "WARNING! Mode solver arrays must contain alpha_bend data"
                f" down to min(alpha_prop)/100 ({α_prop_min/100:.2e} um-1)!"
            )
            if not self.parms.debug.disable_R_domain_check:
                raise ValueError
        if self.r_α_bend_data_min > self.parms.limits.r_min:
            self.logger("WARNING! Mode solver arrays must contain data at R < Rmin!")
            if not self.parms.debug.disable_R_domain_check:
                raise ValueError

        # Fit alpha_bend(r, u) 2D polynomial model to the mode solver data
        self.α_bend = lambda *args, **kwargs: 0
        self._α_bend_model_fig: dict = {}
        self.α_bend_model_symbolic: functions.exp | None = None
        self._α_bend_model_fig_azim: float = 0
        self._α_bend_model_fig_elev: float = 0
        self._fit_α_bend_model()
        self._plot_α_bend_model_fitting_results()

        # For use in the optimization: constrain the search domain for u at low radii,
        # else the optimization erroneously converges to a local minima.
        self.r_min_for_u_search_lower_bound: float = 0
        self.r_max_for_u_search_lower_bound: float = 0
        self.u_lower_bound: interpolate.interp1d | None
        if not self.parms.debug.disable_u_search_lower_bound:
            self._set_u_search_lower_bound()

    #
    # alpha_wg(u), gamma(u), u(gamma), and neffs(u) modeling
    #

    # Payne and Lacey model for propagation losses from vertical sidewall roughness
    def _calc_k_parallel_projections(
        self, theta: float, n_clad: float, n_core: float, n_sub: float
    ) -> Tuple[float, float, float]:
        """

        Args:
            theta (float): projection angle between k and the film plane
            n_clad (float): cladding index
            n_core (float): core index
            n_sub (float): substrate index

        Returns: gamma_clad, kx_core, gamma_sub

        """

        # Calculate wave-numbers
        k0: float = (2 * np.pi) / (self.parms.wg.lambda_resonance * 1e-6)
        k_clad: float = k0 * n_clad
        k_core: float = k0 * n_core
        k_sub: float = k0 * n_sub
        beta: float = k_core * np.sin(theta)

        # Kludge because of bug in scipy_optimize
        gamma_clad: float = np.sqrt(beta**2 - k_clad**2) if beta > k_clad else 0
        kx_core: float = np.sqrt(k_core**2 - beta**2) if k_core > beta else 0
        gamma_sub: float = np.sqrt(beta**2 - k_sub**2) if beta > k_sub else 0

        # Return results!
        return gamma_clad, kx_core, gamma_sub

    def _calculate_mode_parameters_obj_fun(
        self,
        theta: float,
        h: float,
        n_clad: float,
        n_core: float,
        n_sub: float,
        polarization: str,
    ) -> float:
        """

        Args:
            theta (float): projection angle between k and the film plane
            h (float): core thickness
            n_clad (float): cladding index
            n_core (float): core index
            n_sub (float): substrate index
            polarization (str): light polarization

        Returns: squared loss exponent

        """

        # Calculate squared residual
        a: float = h / 2
        [gamma_clad, kx_core, gamma_sub] = self._calc_k_parallel_projections(
            theta=theta, n_clad=n_clad, n_core=n_core, n_sub=n_sub
        )
        if polarization == "TE":
            e = (
                2 * a * kx_core
                - np.arctan2(gamma_sub, kx_core)
                - np.arctan2(gamma_clad, kx_core)
            )
        else:
            e = (
                2 * a * kx_core
                - np.arctan2(n_core**2 * gamma_sub, n_sub**2 * kx_core)
                - np.arctan2(n_core**2 * gamma_clad, n_clad**2 * kx_core)
            )
        return e**2

    def _calc_mode_effective_index(
        self,
        h: float,
        n_clad: float,
        n_core: float,
        n_sub: float,
        polarization: str,
    ) -> Tuple[float, float]:
        """

        Args:
            h (float): core thickness
            n_clad (float): cladding index
            n_core (float): core index
            n_sub (float): substrate index
            polarization (str):

        Returns: n_eff, residual from the fit

        """

        theta_max: float = np.pi / 2
        theta_min: float = (
            np.arcsin(n_sub / n_core) if n_sub > n_clad else np.arcsin(n_clad / n_core)
        )
        optimization_result: optimize.OptimizeResult = optimize.minimize(
            fun=self._calculate_mode_parameters_obj_fun,
            x0=np.asarray([(theta_max + theta_min) / 2]),
            bounds=((theta_min, theta_max),),
            method="Powell",
            args=(h, n_clad, n_core, n_sub, polarization),
        )
        theta: float = optimization_result.x[0]
        residual: float = optimization_result.fun
        n_eff: float = n_core * np.sin(theta)

        # Return results
        return n_eff, residual

    def _calc_alpha_db_per_cm(self, n_eff: float, height: float, width: float) -> float:
        """

        Args:
            n_eff (float): mode effective index
            height (float): core height
            width (float): cor width

        Returns: alpha (db/cm)

        """

        # Physical variables and mode parameters
        k0: float = 2 * np.pi / (self.parms.wg.lambda_resonance * 1e-6)
        d: float = width * 1e-6 / 2

        # Calculate effective index for mode in 1d vertical stack
        n_eff_1d, residual_1d = self._calc_mode_effective_index(
            h=height * 1e-6,
            n_clad=self.parms.wg.n_clad,
            n_core=self.parms.wg.n_core,
            n_sub=self.parms.wg.n_sub,
            polarization=self.parms.wg.polarization,
        )

        # Payne and Lacey model
        u: float = k0 * d * np.sqrt(n_eff_1d**2 - n_eff**2)
        v: float = k0 * d * np.sqrt(n_eff_1d**2 - self.parms.wg.n_clad**2)
        w: float = k0 * d * np.sqrt(n_eff**2 - self.parms.wg.n_clad**2)
        g: float = (u * v) ** 2 / (1 + w)
        x: float = w * self.parms.wg.roughness_lc / d
        delta: float = (n_eff_1d**2 - self.parms.wg.n_clad**2) / (2 * n_eff_1d**2)
        γ: float = (self.parms.wg.n_clad * v) / (n_eff_1d * w * np.sqrt(delta))
        big_term: float = np.sqrt((1 + x**2) ** 2 + 2 * x**2 * γ**2)
        f: float = x * np.sqrt(1 - x**2 + big_term) / big_term
        alpha_db_per_m: float = (
            4.34
            * self.parms.wg.roughness_sigma**2
            / (np.sqrt(2) * k0 * d**4 * n_eff_1d)
            * g
            * f
        )

        # Return alpha_wg in dB/cm
        return alpha_db_per_m / 100

    # alpha_wg(u) model function
    def α_wg_of_u(self, u: float | None = None) -> float:
        """

        Args:
            u (float): core geometry free parameter

        Returns: alpha_wg

        """
        # If no height or width specified, return minimum alpha_wg value
        if u is None:
            return self.α_wg_model.min

        # Normal function return: polynomial model for alpha_wg(u)
        if not self.parms.fit.alpha_wg_exponential_model:
            return self.α_wg_model.model(u)

        # Debugging: hard-coded exponential model for alpha_wg(u)
        α_wg_min: float = [value.alpha_wg for value in self.parms.geom.values()][-1]
        α_wg_max: float = α_wg_min * 2
        return (
            α_wg_min
            + (α_wg_max - α_wg_min)
            * np.exp(
                -(u - self.parms.limits.u_min)
                / (self.parms.limits.u_max - self.parms.limits.u_min)
                * 5
            )
        ) / CONSTANTS.per_um_to_db_per_cm

    # gamma(u), u(gamma), neff(u) wrappers for model-specific calls to _interpolate()
    def gamma_of_u(self, u: float) -> float:
        """

        Args:
            u (float): : core geometry free parameter

        Returns: gamma(u)

        """
        return self._interpolate(model=self.gamma_model, x=u)

    def u_of_gamma(self, gamma: float) -> float:
        """

        Args:
            gamma (float): gamma

        Returns: u(gamma)

        """
        return self._interpolate(model=self.u_model, x=gamma)

    def n_eff_of_u(self, u: float) -> float:
        """

        Args:
            u (float): : core geometry free parameter():

        Returns: neff(u)

        """
        return self._interpolate(model=self.n_eff_model, x=u)

    @staticmethod
    def _interpolate(model: PolyModel1D, x: float) -> float:
        """

        Args:
            model (dict): interpolation model
            x (float): model free parameter

        Returns: interpolated value and clip to min/max boundaries

        """

        value: float = model.model(x)
        value = max(model.min, value)

        return min(model.max, value)

    # FIt alpha_wg(u), gamma(u), u(gamma), neff(u) 1D models to the mode solver data
    def _fit_1d_models(self) -> None:
        """
        1) Fit polynomial models to alpha_wg(u), gamma(u), u(gamma), and neffs(u),
           load the info (model parameters, bounds) into dictionaries for each.

        2) Fit interpolation models for r(u) @ max(alpha_bend) and r(u)
           @ min(alpha_bend), i.e. to R[0](u) and R[-1](u).

        Returns: None

        """

        # Polynomial model for alpha_wg(u) in the input mode solver data
        u_data: np.ndarray = np.asarray([value.u for value in self.parms.geom.values()])
        α_wg_data: np.ndarray = (
            np.asarray([value.alpha_wg for value in self.parms.geom.values()])
            / CONSTANTS.per_um_to_db_per_cm
        )
        self.α_wg_model = PolyModel1D(
            "alpha_wg",
            cast(
                Polynomial,
                Polynomial.fit(
                    x=u_data, y=α_wg_data, deg=self.parms.fit.alpha_wg_order
                ),
            ),
            min(α_wg_data),
            max(α_wg_data),
        )

        # Polynomial models for gamma(u) and u(gamma) in the input mode solver data
        gamma_data: np.ndarray = np.asarray(
            [value.gamma for value in self.parms.geom.values()]
        )
        self.gamma_model = PolyModel1D(
            "gamma",
            cast(
                Polynomial,
                Polynomial.fit(x=u_data, y=gamma_data, deg=self.parms.fit.gamma_order),
            ),
            0,
            1,
        )
        self.u_model = PolyModel1D(
            "u",
            cast(
                Polynomial,
                Polynomial.fit(x=gamma_data, y=u_data, deg=self.parms.fit.gamma_order),
            ),
            u_data[0],
            u_data[-1],
        )

        # Polynomial model for neff(u) in the input mode solver data
        n_eff_data: np.ndarray = np.asarray(
            [value.neff for value in self.parms.geom.values()]
        )
        self.n_eff_model = PolyModel1D(
            "neff",
            cast(
                Polynomial,
                Polynomial.fit(x=u_data, y=n_eff_data, deg=self.parms.fit.neff_order),
            ),
            min(n_eff_data),
            max(n_eff_data),
        )

        # Interpolation models for r(u) @ max(alpha_bend) and r(u) @ min(alpha_bend)
        # in the mode solver data, i.e. r(u)[0] and r(u)[-1].
        self.r_α_bend_max_interp = interpolate.interp1d(
            x=u_data, y=[value.r[0] for value in self.parms.geom.values()]
        )
        self.r_α_bend_min_interp = interpolate.interp1d(
            x=u_data,
            y=[value.r[-1] for value in self.parms.geom.values()],
        )

        # Plot modeled and original mode solver data values
        u_interp: np.ndarray = np.linspace(u_data[0], u_data[-1], 100)
        gamma_interp: np.ndarray = np.linspace(gamma_data[0], gamma_data[-1], 100)
        fig, axs = plt.subplots(4)
        fig.suptitle(
            "1D model fits\n"
            f"{self.parms.wg.polarization}, λ = {self.parms.wg.lambda_resonance:.3f} μm"
            f", {self.parms.wg.v_coord_name} = {self.parms.wg.v_coord_value:.3f} μm"
        )
        axs_index: int = 0

        # Plot of gamma(h)
        gamma_modeled: list = [100 * self.gamma_of_u(u) for u in u_interp]
        axs[axs_index].plot(u_data, gamma_data * 100, ".")
        axs[axs_index].plot(u_interp, gamma_modeled)
        axs[axs_index].set(
            title=rf"$\Gamma_{{fluide}}$({self.parms.wg.u_coord_name})"
            f", polynomial model order: {self.parms.fit.gamma_order}",
            xlabel=f"{self.parms.wg.u_coord_name} (μm)",
            ylabel=r"$\Gamma_{fluide}$ (%)",
        )
        axs_index += 1

        # plot of u(gamma)
        u_modeled: list = [self.u_of_gamma(gamma) for gamma in gamma_interp]
        axs[axs_index].plot(gamma_data * 100, u_data, ".")
        axs[axs_index].plot(gamma_interp * 100, u_modeled)
        axs[axs_index].set(
            title=rf"{self.parms.wg.u_coord_name}$(\Gamma_{{fluide}}$)"
            f", polynomial model order: {self.parms.fit.gamma_order}",
            xlabel=r"$\Gamma_{fluide}$",
            ylabel=f"{self.parms.wg.u_coord_name} (μm)",
        )
        axs_index += 1

        # plot of neff(u)
        neff_modeled: list = [self.n_eff_of_u(h) for h in u_interp]
        axs[axs_index].plot(u_data, n_eff_data, ".")
        axs[axs_index].plot(u_interp, neff_modeled)
        axs[axs_index].set(
            title=rf"n$_{{eff}}$({self.parms.wg.u_coord_name})"
            f", polynomial model order: {self.parms.fit.neff_order}",
            xlabel=f"{self.parms.wg.u_coord_name} (μm)",
            ylabel=r"n$_{eff}$ (RIU)",
        )
        axs_index += 1

        # Plot of alpha_wg(u)
        alpha_wg_modeled: np.ndarray = (
            np.asarray([self.α_wg_of_u(u) for u in u_interp])
            * CONSTANTS.per_um_to_db_per_cm
        )
        axs[axs_index].plot(u_interp, alpha_wg_modeled)
        if self.parms.fit.alpha_wg_exponential_model:
            axs[axs_index].set_title(
                rf"$\alpha_{{wg}}$ ({self.parms.wg.u_coord_name}), exponential model"
            )
        else:
            axs[axs_index].plot(u_data, α_wg_data * CONSTANTS.per_um_to_db_per_cm, ".")
            axs[axs_index].set_title(
                rf"$\alpha_{{wg}}$({self.parms.wg.u_coord_name})"
                f", polynomial model order: {self.parms.fit.alpha_wg_order}"
            )
        axs[axs_index].set(
            ylabel=r"$\alpha_{wg}$ (dB/cm)",
            xlabel=f"{self.parms.wg.u_coord_name} (μm)",
        )
        axs[axs_index].set_ylim(bottom=0)

        # Complete plot formatting
        fig.tight_layout()

        # Save graph to file
        out_filename: str = str(
            (
                self.filename_path.parent
                / f"{self.filename_path.stem}_FITTING_1D_POLY_MODELS.png"
            )
        )
        fig.savefig(out_filename)
        self.logger(f"Wrote '{out_filename}'.")

        # Explicit None return
        return None

    #
    # alpha_bend(r, h) modeling
    #

    def _parse_bending_loss_mode_solver_data(self) -> None:
        """
        Parse the alpha_bend(R, u) mode solver data, determine the extrema,
        build the "u / R / log(alpha_bend)" arrays for fitting.

        Returns: None

        """

        # For each u entry in the input dataclass: determine the radius value
        # "r_alpha_bend_threshold" that corresponds to the value of
        # "alpha_bend_threshold" specified in the input file, by fitting
        # a first order polynomial (parameters A&B) to "ln(alpha_bend) = ln(A) - R*B".
        for key, value in self.parms.geom.items():
            ln_a, neg_b = (
                Polynomial.fit(x=value.r, y=np.log(value.alpha_bend), deg=1)
                .convert()
                .coef
            )
            self.parms.geom[key].r_alpha_bend_threshold = (
                ln_a - np.log(self.parms.limits.alpha_bend_threshold)
            ) / -neg_b

        # Loop to build "u / R / log(alpha_bend)" data arrays for fitting
        for value in self.parms.geom.values():
            self.u_data = np.append(self.u_data, value.u * np.ones_like(value.r))
            self.r_alpha_bend_data = np.append(
                self.r_alpha_bend_data, np.asarray(value.r)
            )
            self.ln_alpha_bend_data = np.append(
                self.ln_alpha_bend_data, np.log(value.alpha_bend)
            )

        # Determine dynamic range extrema of the bending loss data
        self.α_bend_data_min = np.exp(min(self.ln_alpha_bend_data))
        self.r_α_bend_data_min = min(self.r_alpha_bend_data)
        self.r_α_bend_data_max = max(self.r_alpha_bend_data)

        # Explicit None return
        return None

    def _α_bend_model_fig_check_button_callback(self, label: str) -> None:
        """
        alpha_bend(r, h) 3D figure check button callback

        Args:
            label (str): check button label

        Returns: None

        """
        index = self._α_bend_model_fig["labels"].index(label)
        self._α_bend_model_fig["lines"][index].set_visible(
            not self._α_bend_model_fig["lines"][index].get_visible()
        )
        plt.draw()

        # Explicit None return
        return None

    def _α_bend_model_fig_slider_callback(self, *_, **__) -> None:
        """
        alpha_bend(r, h) 3D figure slider callback

        Args:
            *_ ():
            **__ ():

        Returns: None

        """

        self._α_bend_model_fig["surface"].set_alpha(
            self._α_bend_model_fig["slider"].val
        )
        plt.draw()

        # Explicit None return
        return None

    def _α_bend_model_fig_save(self, *_, **__) -> None:
        """
        alpha_bend(r, h) 3D figure save button callback

        Args:
            *_ ():
            **__ ():

        Returns: None

        """

        self._α_bend_model_fig["fig"].savefig(self._α_bend_model_fig["out_filename"])
        self.logger(f"Wrote '{self._α_bend_model_fig['out_filename']}'.")

        # Explicit None return
        return None

    def _α_bend_model_fig_top_view(self, *_, **__) -> None:
        """
        alpha_bend(r, h) 3D figure "top view" button callback

        Args:
            *_ ():
            **__ ():

        Returns: None

        """

        self._α_bend_model_fig["ax"].view_init(azim=0, elev=90)
        plt.draw()

        # Explicit None return
        return None

    def _α_bend_model_fig_reset_view(self, *_, **__) -> None:
        """
        alpha_bend(r, h) 3D figure "reset view"" button callback

        Args:
            *_ ():
            **__ ():

        Returns: None

        """

        self._α_bend_model_fig["ax"].view_init(
            azim=self._α_bend_model_fig_azim, elev=self._α_bend_model_fig_elev
        )
        plt.draw()

        # Explicit None return
        return None

    def _plot_α_bend_model_fitting_results(self) -> None:
        """
        plot alpha_bend(r, h) in the 3D figure

        Returns: None

        """

        # Calculate rms error over the data points
        α_bend_fitted: np.ndarray = np.asarray(
            self.α_bend(r=self.r_alpha_bend_data, u=self.u_data)
        )
        rms_error: float = float(
            np.std(self.ln_alpha_bend_data - np.log(α_bend_fitted))
        )

        # Plot model solver data, fitted points, and 3D wireframes & surface
        # to verify the goodness of fit of the alpha_bend(r, u) model.
        self._α_bend_model_fig["fig"] = plt.figure()
        ax: Axes = self._α_bend_model_fig["fig"].add_subplot(projection="3d")
        self._α_bend_model_fig["ax"] = ax
        ax.set_title(
            rf"α$_{{bend}}$(r, u) = {str(self.α_bend_model_symbolic)}"
            + "".join(
                [
                    "\nWireframes: ",
                    r"fitted $\alpha_{bend}$",
                    f"(r, {self.parms.wg.u_coord_name}) model, ",
                    rf"rms error = {rms_error:.1e} $\mu$m$^{{-1}}$ (logarithmic)",
                ]
            )
            + "".join(
                [
                    "\n",
                    rf"Surface : $\alpha_{{prop}}$({self.parms.wg.u_coord_name})",
                    rf" = α$_{{wg}}$({self.parms.wg.u_coord_name})",
                    rf"+ $\Gamma_{{fluide}}({self.parms.wg.u_coord_name})",
                    r"\times\alpha_{fluid}$, where $\alpha_{fluid}$",
                    rf" = {self.α_fluid:.2e} $\mu$m$^{{-1}}$",
                ]
            )
            + "".join(
                [
                    "\nGreen : ",
                    r"$\alpha_{bend} < \alpha_{prop}$ $< \alpha_{bend}\times 10$",
                    r", blue : $\alpha \approx \alpha_{prop}$",
                ]
            )
            + "".join(
                [
                    "\n",
                    rf"$\alpha$(r, {self.parms.wg.u_coord_name}) = $\alpha_{{bend}}$",
                    rf"(r, {self.parms.wg.u_coord_name}) + $\alpha_{{prop}}$",
                    f"({self.parms.wg.u_coord_name})",
                ]
            ),
            y=1.02,
        )
        ax.set(
            xlabel=f"{self.parms.wg.u_coord_name} (μm)",
            ylabel="log($R$) (μm)",
            zlabel=r"log$_{10}$($\alpha_{BEND}$) ($\mu$m$^{-1}$)",
        )
        raw_points: PathCollection = ax.scatter(
            self.u_data,
            np.log10(self.r_alpha_bend_data),
            self.ln_alpha_bend_data * np.log10(np.e),
            color="red",
            label="Mode solver points (raw)",
            s=1,
        )
        fitted_points: PathCollection = ax.scatter(
            self.u_data,
            np.log10(self.r_alpha_bend_data),
            np.log10(α_bend_fitted),
            color="blue",
            label="Mode solver points (fitted)",
            s=1,
        )

        # alpha_bend(r, u) wireframe, for r in [Rmin, Rmax] analysis domain
        u_domain, r_domain = np.meshgrid(
            np.linspace(self.parms.limits.u_min, self.parms.limits.u_max, 75),
            np.logspace(
                np.log10(self.parms.limits.r_min),
                np.log10(self.parms.limits.r_max),
                100,
            ),
        )
        α_bend: np.ndarray = np.asarray(self.α_bend(r=r_domain, u=u_domain))
        α_bend[α_bend < self.α_bend_data_min / 2] = self.α_bend_data_min / 2
        model_mesh: LineCollection = ax.plot_wireframe(
            X=u_domain,
            Y=np.log10(r_domain),
            Z=np.log10(α_bend),
            rstride=5,
            cstride=5,
            alpha=0.30,
            label=r"$\alpha_{bend}$"
            f"(r, {self.parms.wg.u_coord_name})" + r", $r \in [R_{min}, R_{max}$]",
        )

        # alpha_bend(r, u) wireframe, for all r in mode solver data
        u_data, r_data = np.meshgrid(
            np.linspace(self.parms.limits.u_min, self.parms.limits.u_max, 15),
            np.logspace(
                np.log10(self.r_α_bend_data_min),
                np.log10(self.r_α_bend_data_max),
                20,
            ),
        )
        α_bend_data: np.ndarray = np.asarray(self.α_bend(r=r_data, u=u_data))
        α_bend_data[α_bend_data < self.α_bend_data_min / 2] = self.α_bend_data_min / 2
        data_mesh: LineCollection = ax.plot_wireframe(
            X=u_data,
            Y=np.log10(r_data),
            Z=np.log10(α_bend_data),
            alpha=0.30,
            color="red",
            label=r"$\alpha_{bend}$"
            f"(r, {self.parms.wg.u_coord_name})" + r", $r \in$ [mode solver data]",
        )
        ax.legend(bbox_to_anchor=(0.1, 0.1))

        # alpha_prop(u) surface:
        #   red:    alpha_prop < alpha_bend
        #   green:  alpha_bend < alpha_prop < alpha_bend x 10
        #   blue:   alpha_prop > alpha_bend x 10, alpha_prop dominates
        α_prop: np.ndarray = np.ones_like(u_domain)
        for αp, u in np.nditer(
            [α_prop, u_domain], op_flags=[["readwrite"], ["readonly"]]
        ):
            αp[...] = self.α_wg_of_u(u[...]) + (self.gamma_of_u(u[...]) * self.α_fluid)
        face_colors: np.ndarray = np.copy(
            np.broadcast_to(colors.to_rgba(c="red", alpha=0.8), u_domain.shape + (4,))
        )
        for index in np.ndindex(u_domain.shape):
            if α_prop[index] > α_bend[index]:
                if α_prop[index] > α_bend[index] * 10:
                    face_colors[index] = colors.to_rgba(c="blue", alpha=0.5)
                else:
                    face_colors[index] = colors.to_rgba(c="green", alpha=1)
        α_prop_surface: PolyCollection = ax.plot_surface(
            X=u_domain,
            Y=np.log10(r_domain),
            Z=np.log10(α_prop),
            facecolors=face_colors,
            label=r"$\alpha_{prop}$" f"({self.parms.wg.u_coord_name}) surface",
            linewidth=0,
            rstride=1,
            cstride=1,
        )
        self._α_bend_model_fig["surface"] = α_prop_surface

        # Add "check button" to figure to turn visibility of individual plots on/off
        self._α_bend_model_fig["lines"] = [
            raw_points,
            fitted_points,
            model_mesh,
            data_mesh,
            α_prop_surface,
        ]
        self._α_bend_model_fig["lines"][3].set_visible(False)
        self._α_bend_model_fig["labels"] = [
            str(line.get_label()) for line in self._α_bend_model_fig["lines"]
        ]
        self._α_bend_model_fig["check_button"] = CheckButtons(
            ax=plt.axes([0.01, 0.71, 0.30, 0.15]),
            labels=self._α_bend_model_fig["labels"],
            actives=([True, True, True, False, True]),
        )
        self._α_bend_model_fig["check_button"].on_clicked(
            self._α_bend_model_fig_check_button_callback
        )

        # Add "save" button, to save the image of the 3D plot to file
        self._α_bend_model_fig["button_save"] = Button(
            ax=plt.axes([0.01, 0.4, 0.10, 0.05]), label="Save", color="white"
        )
        self._α_bend_model_fig["button_save"].on_clicked(self._α_bend_model_fig_save)

        # Add "Reset" and "Top view" buttons, to either reset the 3D view
        # to the initial perspective, or to a pure top-down view.
        self._α_bend_model_fig["button_reset_view"] = Button(
            ax=plt.axes([0.01, 0.5, 0.10, 0.05]), label="Reset view", color="white"
        )
        self._α_bend_model_fig["button_reset_view"].on_clicked(
            self._α_bend_model_fig_reset_view
        )
        self._α_bend_model_fig["button_top_view"] = Button(
            ax=plt.axes([0.01, 0.56, 0.10, 0.05]), label="Top view", color="white"
        )
        self._α_bend_model_fig["button_top_view"].on_clicked(
            self._α_bend_model_fig_top_view
        )

        # Rotate plot and write to file
        self._α_bend_model_fig_azim = 40
        self._α_bend_model_fig_elev = 30
        ax.view_init(azim=self._α_bend_model_fig_azim, elev=self._α_bend_model_fig_elev)
        self._α_bend_model_fig["out_filename"] = str(
            (
                self.filename_path.parent
                / f"{self.filename_path.stem}_FITTING_3D_POLY_MODEL.png"
            )
        )

        self._α_bend_model_fig_save()

        # Explicit None return
        return None

    def _fit_α_bend_model(self) -> None:
        """
        Polynomial model least squares fit to ln(alpha_bend(r, u))

        Current model:

        ln(alpha_bend) = c0 + c1*u + c2*r + c3*u*r + c4*u**2*r + c5*u**3*r + c6*u**4

        NB: the use of a model including a logarithm, ln(alpha_bend(r, u)),
            rather than alpha_bend(r, u) directly, was inspired by the known model
            alpha_bend(R) = A*exp(-B*R), which is fitted to mode solver data
            using linear least squares with ln(alpha_bend(R)) = ln(A) - B*R.

        To change the model, modify the matching polynomial model and M matrix
        definitions sections below ("USER-DEFINABLE MODEL-SPECIFIC SECTION").

        PS: I tried the symfit package, didn't get anything better than the result
            below with lstsq(), and it failed to converge for most polynomial forms.
        """

        #
        # USER-DEFINABLE MODEL-SPECIFIC SECTION
        #

        # Define the symbolic polynomial model. The monomials (terms) in the model
        # must match the column definitions in the "M" matrix below exactly.
        r, u = symbols("r, u")
        c: tuple = symbols("c:7")
        self.α_bend_model_symbolic: functions.exp = functions.exp(
            c[0]
            + c[1] * u
            + c[2] * r
            + c[3] * (u * r)
            + c[4] * (u**2 * r)
            + c[5] * (u**3 * r)
            + c[6] * (u**4)
        )

        # Assemble the "M" coefficient matrix, where each column holds the values of
        # the monomial terms in the model for each r & u value pair in the input data.
        # NB: the model is actually alpha_bend(u,r) = exp(model(u,r)) because
        # the model coefficients are fitted to the ln(alpha_bend(u,r)) data.
        m: np.nedarray = np.asarray(
            [
                np.ones_like(self.u_data),
                self.u_data,
                self.r_alpha_bend_data,
                self.u_data * self.r_alpha_bend_data,
                self.u_data**2 * self.r_alpha_bend_data,
                self.u_data**3 * self.r_alpha_bend_data,
                self.u_data**4,
            ]
        ).T

        #
        # END OF USER-DEFINABLE MODEL-SPECIFIC SECTION
        #

        # Check that the model definition and the coefficient matrix are consistent
        assert len(c) == m.shape[1], (
            f"Model ({len(c)} coefficients) and "
            f"coefficient matrix ({m.shape[1]} columns) are inconsistent!"
        )

        # Linear least-squares fit to "ln(alpha_bend(r, u)" to determine the values of
        # the model coefficients. Although the model contains an exponential, the model
        # without the exponential is fitted to ln(alpha_bend(r, u)), to use
        # LINEAR least squares.
        c_fitted, residual, rank, s = lstsq(m, self.ln_alpha_bend_data)
        assert rank == len(
            c_fitted
        ), f"Matrix rank ({rank}) is not equal to model order+1 ({len(c_fitted)})!"

        # Insert the fitted coefficient values into the model, then convert the symbolic
        # model to a lambda function for faster evaluation.
        α_bend_model_fitted = self.α_bend_model_symbolic.subs(list(zip(c, c_fitted)))
        self.α_bend: FunctionClass = lambdify([r, u], α_bend_model_fitted, "numpy")

        # Calculate rms fitting error over the data set
        rms_error: float = float(
            np.std(
                self.ln_alpha_bend_data
                - np.log(self.α_bend(r=self.r_alpha_bend_data, u=self.u_data))
            )
        )
        self.logger(f"Fitted ln(alpha_bend) model, rms error = {rms_error:.1e}.")

        # Explicit None return
        return None

    #
    # Utilities for use in optimization in the find_max_sensitivity() sensor methods
    #

    def _set_u_search_lower_bound(self) -> None:
        # At small ring radii, the interpolation model for alpha_bend(u, r) is
        # unreliable at low u. As a result, the search for optimal u in the optimization
        # sometimes converges towards a solution at low u whereas the solution at small
        # radii lies necessarily at high u to minimize bending losses. To mitigate this
        # problem, the search domain lower bound for u is constrained at small radii.
        #
        # The point at which the ring radius is considered "small", i.e. where the
        # alpha_bend interpolation model fails, is u-dependant. This boundary
        # is determined by calculating the radius at each u for which
        # alpha_bend exceeds a user-specified threshold ("alpha_bend_threshold"),
        # values are stored in the array "r_alpha_bend_threshold",
        # see _parse_bending_loss_mode_solver_data(). A spline interpolation
        # is used to model the u search domain lower bound as a function of radius.
        #
        # For radii greater than the spline r domain, u is allowed to take on any value
        # in the full u domain during optimization.

        # Fetch list of core geometry u values
        u: np.ndarray = np.asarray([value.u for value in self.parms.geom.values()])

        # Fit the piecewise spline to model h_lower_bound(r)
        r_α_bend_threshold: np.ndarray = np.asarray(
            [value.r_alpha_bend_threshold for value in self.parms.geom.values()]
        )
        max_indx: int = int(np.argmax(r_α_bend_threshold))
        r_α_bend_threshold: np.ndarray = r_α_bend_threshold[max_indx:]
        if len(r_α_bend_threshold) <= 1:
            raise ValueError(
                "ERROR: 'alpha_bend_threshold' value "
                f"({self.parms.limits.alpha_bend_threshold:.3f})"
                " is too high, decrease value in .toml file."
            )
        u_α_bend_threshold: np.ndarray = u[max_indx:]
        self.r_max_for_u_search_lower_bound = r_α_bend_threshold[0]
        self.r_min_for_u_search_lower_bound = r_α_bend_threshold[-1]
        self.u_lower_bound = interpolate.interp1d(
            x=r_α_bend_threshold, y=u_α_bend_threshold
        )
        if self.r_max_for_u_search_lower_bound < 500:
            self.logger(
                f"{self.parms.wg.u_coord_name} search domain lower bound constrained"
                f" for R < {self.r_max_for_u_search_lower_bound:.1f} um"
            )
        else:
            self.logger(
                f"WARNING!: {self.parms.wg.u_coord_name} search domain lower"
                f"bound ({self.r_max_for_u_search_lower_bound:.1f} um) is unusually "
                "high, raise value of 'alpha_bend_threshold' in .toml file."
            )

        # Plot u_lower_bound(r) data and spline interpolation
        fig, axs = plt.subplots()
        fig.suptitle(
            "".join(
                [
                    f"Search domain lower bound for {self.parms.wg.u_coord_name}",
                    " at low radii in the optimization\n",
                    f"{self.parms.wg.polarization}",
                ]
            )
            + "".join(
                [r", $\lambda$", f" = {self.parms.wg.lambda_resonance:.3f} ", r"$\mu$m"]
            )
            + "".join(
                [
                    f", {self.parms.wg.v_coord_name}",
                    rf" = {self.parms.wg.v_coord_value:.3f} $\mu$m",
                ]
            )
            + (
                "".join(
                    [
                        f"\nWARNING!: {self.parms.wg.u_coord_name} search domain",
                        f" lower bound ({self.r_max_for_u_search_lower_bound:.1f} um)"
                        " is VERY HIGH!",
                    ]
                )
                if self.r_max_for_u_search_lower_bound > 500
                else ""
            ),
            color="black" if self.r_max_for_u_search_lower_bound < 500 else "red",
        )
        r_α_bend_threshold_i: np.ndarray = np.linspace(
            self.r_min_for_u_search_lower_bound,
            self.r_max_for_u_search_lower_bound,
            100,
        )
        axs.plot(r_α_bend_threshold, u_α_bend_threshold, "o")
        axs.plot(r_α_bend_threshold_i, self.u_lower_bound(r_α_bend_threshold_i))
        axs.set(
            xlabel=r"R ($\mu$m)",
            ylabel=r"min$\{$" f"{self.parms.wg.u_coord_name}" + r"$\}$ ($\mu$m)",
        )
        out_filename: str = str(
            (
                self.filename_path.parent
                / "".join(
                    [
                        f"{self.filename_path.stem}_FITTING_"
                        f"{'H' if self.parms.wg.v_coord_name == 'w' else 'W'}"
                        "_SRCH_LOWR_BND.png"
                    ]
                )
            )
        )
        fig.savefig(out_filename)
        self.logger(f"Wrote '{out_filename}'.")

        # If the R values used for interpolation of the h domain lower bound are not
        # in monotonically increasing oder, exit with an error.
        if np.any(np.diff(r_α_bend_threshold) > 0):
            raise ValueError(
                "ERROR! Search domain lower bound fit:R "
                f"values are not monotonically decreasing (see {out_filename})! "
                "Decrease value of 'alpha_bend_threshold' in .toml file."
            )

        # Explicit None return
        return None

    def u_search_domain(self, r: float) -> Tuple[float, float]:
        """
        Determine u search domain extrema, see _set_u_search_lower_bound()

        Args:
            r (float): waveguide bending radius (um)

        Returns: (u_min, u_max) u search domain extrema (um)

        """

        if self.parms.debug.disable_u_search_lower_bound:
            return self.parms.limits.u_min, self.parms.limits.u_max
        if r < self.r_min_for_u_search_lower_bound:
            u_min: float = self.parms.limits.u_max
        elif r > self.r_max_for_u_search_lower_bound:
            u_min = self.parms.limits.u_min
        else:
            u_min = min(float(self.u_lower_bound(r)), self.parms.limits.u_max)
            u_min = max(u_min, self.parms.limits.u_min)
        u_max: float = self.parms.limits.u_max

        return u_min, u_max

    #
    # Calculate shared values for sensor Class instance plotting
    #

    def calculate_plotting_extrema(self, max_s: float) -> None:
        """

        Args:
            max_s (float): maximum sensitivity

        Returns: None

        """

        # R domain extrema (complete decades)
        self.plotting_extrema["r_plot_min"] = 10 ** (
            np.floor(np.log10(self.parms.limits.r_min))
        )
        self.plotting_extrema["r_plot_max"] = 10 ** (
            np.ceil(np.log10(self.parms.limits.r_max))
        )

        # u domain extrema
        u: np.ndarray = np.asarray([value.u for value in self.parms.geom.values()])
        self.plotting_extrema["u_plot_min"] = u[0] * 0.9
        self.plotting_extrema["u_plot_max"] = u[-1] * 1.1

        # Gamma domain extrema (%)
        self.plotting_extrema["gamma_plot_min"] = (
            np.floor(self.gamma_of_u(u[-1]) * 0.9 * 10) * 10
        )
        self.plotting_extrema["gamma_plot_max"] = (
            np.ceil(self.gamma_of_u(u[0]) * 1.1 * 10) * 10
        )

        # max{S} vertical marker
        self.plotting_extrema["S_plot_max"] = 10 ** np.ceil(np.log10(max_s))

        # Explicit None return
        return None
