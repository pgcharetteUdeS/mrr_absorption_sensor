"""

Models class

Exposed methods:
    - α_bend(r, u)
    - α_prop(u)
    - α_wg_of_u(u)
    - calculate_plotting_extrema()
    - gamma_of_u(u)
    - n_eff_of_u(u)
    - u_of_gamma(gamma)
    - u_search_domain(r)

    NB:
    1) To accommodate either fixed core width or fixed core height optimization
       problems, the waveguide core geometry dimensions are labeled "u" and "v",
       with "u" being the variable components and "v" the fixed component.
    2) The model for alpha_bend(r, u) is hardcoded in fit_alpha_bend_model()
       but the code is structured in such a way that it is relatively easy to change,
       see the "USER-DEFINABLE MODEL-SPECIFIC SECTION" code section.

"""
from colorama import Fore, Style
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, CheckButtons
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
from scipy import interpolate
from scipy.linalg import lstsq
from sympy import functions, lambdify, symbols
from typing import Callable

# Package modules
from .constants import PER_UM_TO_DB_PER_CM


class Models:
    """
    Models class for polynomial interpolation: gamma(u), neffs(u), α_wg(u), α_bend(r, u)

    All lengths are in units of um
    """

    def __init__(
        self,
        parameters: dict,
        modes_data: dict,
        bending_loss_data: dict,
        filename_path: Path,
        logger: Callable = print,
    ):

        # Load class instance input parameters
        self.α_bend_threshold: float = parameters["alpha_bend_threshold"]
        self.α_wg_order = parameters["alpha_wg_order"]
        self.bending_loss_data: dict = bending_loss_data
        self.core_u_name: str = parameters["core_u_name"]
        self.core_v_name: str = parameters["core_v_name"]
        self.core_v_value: float = parameters["core_v_value"]
        self.disable_u_search_lower_bound = parameters["disable_u_search_lower_bound"]
        self.filename_path: Path = filename_path
        self.gamma_order = parameters["gamma_order"]
        self.lambda_res: float = parameters["lambda_res"]
        self.logger: Callable = logger
        self.modes_data: dict = modes_data
        self.n_eff_order = parameters["neff_order"]
        self.ni_op: float = parameters["ni_op"]
        self.parameters: dict = parameters
        self.pol: str = parameters["pol"]
        self.r_min: float = parameters["Rmin"]
        self.r_max: float = parameters["Rmax"]
        self.r_samples_per_decade: int = parameters["R_samples_per_decade"]

        # Initialize other class parameters
        self.plotting_extrema: dict = {}

        # Define the array of radii to be analyzed (R domain)
        self.r: np.ndarray = np.logspace(
            start=np.log10(self.r_min),
            stop=np.log10(self.r_max),
            num=int(
                (np.log10(parameters["Rmax"]) - np.log10(parameters["Rmin"]))
                * self.r_samples_per_decade
            ),
            base=10,
        )

        # Define propagation losses in the fluid medium (1/um)
        self.α_fluid: float = (4 * np.pi / self.lambda_res) * self.ni_op

        # Parse and validate the mode solver data loaded from the .toml file
        self.u_data: np.ndarray = np.asarray([])
        self.r_alpha_bend_data: np.ndarray = np.asarray([])
        self.ln_alpha_bend_data: np.ndarray = np.asarray([])
        self.u_domain_min: float = 0
        self.u_domain_max: float = 0
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

        # Fit gamma(h), h(gamma), and neff(h) 1D poly models to the mode solver data
        self.α_wg_model: dict = {}
        self.gamma_model: dict = {}
        self.u_model: dict = {}
        self.n_eff_model: dict = {}
        self._fit_1d_models()
        self.α_wg_db_per_cm: float = self.α_wg_of_u() * PER_UM_TO_DB_PER_CM

        # Check that the bending loss mode solver data covers the required h & R ranges
        α_prop_min: float = self.α_wg_of_u() + (
            self.gamma_of_u(self.u_domain_max) * self.α_fluid
        )
        if self.α_bend_data_min > α_prop_min / 100:
            self.logger(
                f"{Fore.YELLOW}WARNING! Mode solver arrays must contain alpha_bend data"
                + f" down to min(alpha_prop)/100 ({α_prop_min/100:.2e} um-1)!"
                + f"{Style.RESET_ALL}"
            )
            if not parameters["disable_R_domain_check"]:
                raise ValueError
        if self.r_α_bend_data_min > self.r_min:
            self.logger(
                f"{Fore.YELLOW}WARNING! Mode solver arrays must contain data"
                + f" at radii < Rmin!{Style.RESET_ALL}"
            )
            if not parameters["disable_R_domain_check"]:
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
        self.u_lower_bound: interpolate.interp1d = interpolate.interp1d([0, 1], [0, 1])
        self._set_u_search_lower_bound()

    #
    # alpha_wg(u), gamma(u), u(gamma), and neffs(u) modeling
    #

    # alpha_wg(u) model function
    def α_wg_of_u(self, u: float = None) -> float:
        """

        Args:
            u ():

        Returns:

        """
        # If no height or width specified, return minimum alpha_wg value
        if u is None:
            return self.α_wg_model["min"]

        # Normal function return: polynomial model for alpha_wg(u)
        if not self.parameters["alpha_wg_exponential_model"]:
            return self.α_wg_model["model"](u)

        # Debugging: hard-coded exponential model for alpha_wg(u)
        α_wg_min: float = [value.get("alpha_wg") for value in self.modes_data.values()][
            -1
        ]
        α_wg_max: float = α_wg_min * 2
        return (
            α_wg_min
            + (α_wg_max - α_wg_min)
            * np.exp(
                -(u - self.u_domain_min) / (self.u_domain_max - self.u_domain_min) * 5
            )
        ) / PER_UM_TO_DB_PER_CM

    # gamma(u), u(gamma), neff(u) wrappers for model-specific calls to _interpolate()
    def gamma_of_u(self, u: float) -> float:
        """

        :param u:  waveguide core u dimension (um)
        :return: gamma(u) polynomial model estimate
        """
        return self._interpolate(model=self.gamma_model, x=u)

    def u_of_gamma(self, gamma: float) -> float:
        """

        :param gamma:  gamma
        :return: u(gamma) polynomial model estimate
        """
        return self._interpolate(model=self.u_model, x=gamma)

    def n_eff_of_u(self, u: float) -> float:
        """

        :param u: waveguide core dimension (um)
        :return:  neff(u) polynomial model estimate
        """
        return self._interpolate(model=self.n_eff_model, x=u)

    @staticmethod
    def _interpolate(model: dict, x: float) -> float:
        """
        Interpolate gamma(u), u(gamma), and neffs(u) with models fitted
        by _fit_1D_models(), limit x to the allowed bounds.
        """

        value: float = model["model"](x)
        value = max(model["min"], value)
        value = min(model["max"], value)

        return value

    # FIt alpha_wg(u), gamma(u), u(gamma), neff(u) 1D models to the mode solver data
    def _fit_1d_models(self):
        """
        1) Fit polynomial models to alpha_wg(u), gamma(u), u(gamma), and neffs(u),
           load the info (model parameters, bounds) into dictionaries for each.

        2) Fit interpolation models for R(u) @ max(alpha_bend) and R(u)
           @ min(alpha_bend), i.e. to R[0](u) and R[-1](u).
        """

        # Polynomial model for alpha_wg(u) in the input mode solver data
        u_data = np.asarray([value.get("u") for value in self.modes_data.values()])
        α_wg_data = (
            np.asarray([value.get("alpha_wg") for value in self.modes_data.values()])
            / PER_UM_TO_DB_PER_CM
        )
        self.α_wg_model = {
            "name": "alpha_wg",
            "model": Polynomial.fit(x=u_data, y=α_wg_data, deg=self.α_wg_order),
            "min": min(α_wg_data),
            "max": max(α_wg_data),
        }

        # Polynomial models for gamma(u) and u(gamma) in the input mode solver data
        gamma_data = np.asarray(
            [value.get("gamma") for value in self.modes_data.values()]
        )
        self.gamma_model = {
            "name": "gamma",
            "model": Polynomial.fit(x=u_data, y=gamma_data, deg=self.gamma_order),
            "min": 0,
            "max": 1,
        }
        self.u_model = {
            "name": "u",
            "model": Polynomial.fit(x=gamma_data, y=u_data, deg=self.gamma_order),
            "min": u_data[0],
            "max": u_data[-1],
        }

        # Polynomial model for neff(u) in the input mode solver data
        n_eff_data = np.asarray(
            [value.get("neff") for value in self.modes_data.values()]
        )
        self.n_eff_model = {
            "name": "neff",
            "model": Polynomial.fit(x=u_data, y=n_eff_data, deg=self.n_eff_order),
            "min": min(n_eff_data),
            "max": max(n_eff_data),
        }

        # Interpolation models for R(u) @ max(alpha_bend) and R(u) @ min(alpha_bend)
        # in the input mode solver data, i.e. R[0](u) and R[-1](u).
        self.r_α_bend_max_interp = interpolate.interp1d(
            x=u_data, y=[value.get("R")[0] for value in self.bending_loss_data.values()]
        )
        self.r_α_bend_min_interp = interpolate.interp1d(
            x=u_data,
            y=[value.get("R")[-1] for value in self.bending_loss_data.values()],
        )

        # Plot modeled and original mode solver data values
        u_interp: np.ndarray = np.linspace(u_data[0], u_data[-1], 100)
        gamma_interp: np.ndarray = np.linspace(gamma_data[0], gamma_data[-1], 100)
        fig, axs = plt.subplots(4)
        fig.suptitle(
            "1D model fits\n"
            + f"{self.pol}"
            + f", λ = {self.lambda_res:.3f} μm"
            + f", {self.core_v_name} = {self.core_v_value:.3f} μm"
        )
        axs_index: int = 0

        # Plot of gamma(h)
        gamma_modeled = [100 * self.gamma_of_u(u) for u in u_interp]
        axs[axs_index].plot(u_data, gamma_data * 100, ".")
        axs[axs_index].plot(u_interp, gamma_modeled)
        axs[axs_index].set_title(
            r"$\Gamma_{fluide}$"
            + f"({self.core_u_name})"
            + f", polynomial model order: {self.gamma_order}"
        )
        axs[axs_index].set_xlabel("".join([f"{self.core_u_name}", "(μm)"]))
        axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ (%)")
        axs_index += 1

        # plot of u(gamma)
        u_modeled = [self.u_of_gamma(gamma) for gamma in gamma_interp]
        axs[axs_index].plot(gamma_data * 100, u_data, ".")
        axs[axs_index].plot(gamma_interp * 100, u_modeled)
        axs[axs_index].set_title(
            f"{self.core_u_name}"
            + r"$(\Gamma_{fluide}$)"
            + f", polynomial model order: {self.gamma_order}"
        )
        axs[axs_index].set_xlabel(r"$\Gamma_{fluide}$")
        axs[axs_index].set_ylabel("".join([f"{self.core_u_name}", "(μm)"]))
        axs_index += 1

        # plot of neff(u)
        neff_modeled = [self.n_eff_of_u(h) for h in u_interp]
        axs[axs_index].plot(u_data, n_eff_data, ".")
        axs[axs_index].plot(u_interp, neff_modeled)
        axs[axs_index].set_title(
            r"n$_{eff}$"
            + f"({self.core_u_name})"
            + f", polynomial model order: {self.n_eff_order}"
        )
        axs[axs_index].set_ylabel(r"n$_{eff}$ (RIU)")
        axs[axs_index].set_xlabel("".join([f"{self.core_u_name} (μm)"]))
        axs_index += 1

        # Plot of alpha_wg(u)
        alpha_wg_modeled = np.asarray([self.α_wg_of_u(u) for u in u_interp]) * (
            PER_UM_TO_DB_PER_CM
        )
        axs[axs_index].plot(u_interp, alpha_wg_modeled)
        if self.parameters["alpha_wg_exponential_model"]:
            axs[axs_index].set_title(
                rf"$\alpha_{{wg}}$ ({self.core_u_name}), exponential model"
            )
        else:
            axs[axs_index].plot(u_data, α_wg_data * PER_UM_TO_DB_PER_CM, ".")
            axs[axs_index].set_title(
                r"$\alpha_{wg}$"
                + f"({self.core_u_name})"
                + f", polynomial model order: {self.α_wg_order}"
            )
        axs[axs_index].set_ylabel(r"$\alpha_{wg}$ (dB/cm)")
        axs[axs_index].set_ylim(bottom=0)
        axs[axs_index].set_xlabel("".join([f"{self.core_u_name} (μm)"]))

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

    #
    # alpha_bend(r, h) modeling
    #

    def _parse_bending_loss_mode_solver_data(self):
        """
        Parse the alpha_bend(R, u) mode solver data, determine the extrema,
        build the "u / R / log(alpha_bend)" arrays for fitting.
        """

        # For each u entry in the input dictionary: determine the radius value
        # "r_alpha_bend_threshold" that corresponds to the value of
        # "alpha_bend_threshold" specified in the input file, by fitting
        # a first order polynomial (parameters A&B) to "ln(alpha_bend) = ln(A) - R*B".
        for key, value in self.bending_loss_data.items():
            ln_a, neg_b = (
                Polynomial.fit(x=value["R"], y=np.log(value["alpha_bend"]), deg=1)
                .convert()
                .coef
            )
            self.bending_loss_data[key]["r_alpha_bend_threshold"] = (
                ln_a - np.log(self.α_bend_threshold)
            ) / -neg_b

        # Loop to build "u / R / log(alpha_bend)" data arrays for fitting
        for u_key_um, value in self.bending_loss_data.items():
            self.u_data = np.append(self.u_data, u_key_um * np.ones_like(value["R"]))
            self.r_alpha_bend_data = np.append(
                self.r_alpha_bend_data, np.asarray(value["R"])
            )
            self.ln_alpha_bend_data = np.append(
                self.ln_alpha_bend_data, np.log(value["alpha_bend"])
            )

        # Determine dynamic range extrema of the bending loss data
        self.u_domain_min: float = float(list(self.bending_loss_data.keys())[0])
        self.u_domain_max: float = float(list(self.bending_loss_data.keys())[-1])
        self.α_bend_data_min = np.exp(min(self.ln_alpha_bend_data))
        self.r_α_bend_data_min = min(self.r_alpha_bend_data)
        self.r_α_bend_data_max = max(self.r_alpha_bend_data)

    def _α_bend_model_fig_check_button_callback(self, label):
        index = self._α_bend_model_fig["labels"].index(label)
        self._α_bend_model_fig["lines"][index].set_visible(
            not self._α_bend_model_fig["lines"][index].get_visible()
        )
        plt.draw()

    def _α_bend_model_fig_slider_callback(self, *_, **__):
        self._α_bend_model_fig["surface"].set_alpha(
            self._α_bend_model_fig["slider"].val
        )
        plt.draw()

    def _α_bend_model_fig_save(self, *_, **__):
        self._α_bend_model_fig["fig"].savefig(self._α_bend_model_fig["out_filename"])
        self.logger(f"Wrote '{self._α_bend_model_fig['out_filename']}'.")

    def _α_bend_model_fig_top_view(self, *_, **__):
        self._α_bend_model_fig["ax"].view_init(azim=0, elev=90)
        plt.draw()

    def _α_bend_model_fig_reset_view(self, *_, **__):
        self._α_bend_model_fig["ax"].view_init(
            azim=self._α_bend_model_fig_azim, elev=self._α_bend_model_fig_elev
        )
        plt.draw()

    def _plot_α_bend_model_fitting_results(self):
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
        ax = self._α_bend_model_fig["fig"].add_subplot(projection="3d")
        self._α_bend_model_fig["ax"] = ax
        ax.set_title(
            rf"α$_{{bend}}$(r, u) = {str(self.α_bend_model_symbolic)}"
            + "".join(
                [
                    "\nWireframes: ",
                    r"fitted $\alpha_{bend}$",
                    f"(r, {self.core_u_name}) model, ",
                    f"rms error = {rms_error:.1e}",
                    r" $\mu$m$^{-1}$",
                    " (logarithmic)",
                ]
            )
            + "".join(
                [
                    "\n",
                    r"Surface : $\alpha_{prop}$",
                    f"({self.core_u_name})",
                    rf" = α$_{{wg}}$({self.core_u_name})",
                    r"+ $\Gamma_{fluide}",
                    f"({self.core_u_name})",
                    r"\times\alpha_{fluid}$, where ",
                    r"$\alpha_{fluid}$",
                    f" = {self.α_fluid:.2e} ",
                    r"$\mu$m$^{-1}$",
                ]
            )
            + "".join(
                [
                    "\nGreen : ",
                    r"$\alpha_{bend} < \alpha_{prop}$ ",
                    r"$< \alpha_{bend}\times 10$",
                    r", blue : $\alpha \approx \alpha_{prop}$",
                ]
            )
            + "".join(
                [
                    "\n",
                    r"$\alpha$",
                    f"(r, {self.core_u_name})",
                    r" = $\alpha_{bend}$",
                    f"(r, {self.core_u_name})",
                    r" + $\alpha_{prop}$",
                    f"({self.core_u_name})",
                ]
            ),
            y=1.02,
        )
        ax.set_xlabel("".join([f"{self.core_u_name} (μm)"]))
        ax.set_ylabel("log($R$) (μm)")
        ax.set_zlabel(r"log$_{10}$($\alpha_{BEND}$) ($\mu$m$^{-1}$)")
        raw_points = ax.scatter(
            self.u_data,
            np.log10(self.r_alpha_bend_data),
            self.ln_alpha_bend_data * np.log10(np.e),
            color="red",
            label="Mode solver points (raw)",
            s=1,
        )
        fitted_points = ax.scatter(
            self.u_data,
            np.log10(self.r_alpha_bend_data),
            np.log10(α_bend_fitted),
            color="blue",
            label="Mode solver points (fitted)",
            s=1,
        )

        # alpha_bend(r, u) wireframe, for r in [Rmin, Rmax] analysis domain
        u_domain, r_domain = np.meshgrid(
            np.linspace(self.u_domain_min, self.u_domain_max, 75),
            np.logspace(np.log10(self.r_min), np.log10(self.r_max), 100),
        )
        α_bend: np.ndarray = np.asarray(self.α_bend(r=r_domain, u=u_domain))
        α_bend[α_bend < self.α_bend_data_min / 2] = self.α_bend_data_min / 2
        model_mesh = ax.plot_wireframe(
            X=u_domain,
            Y=np.log10(r_domain),
            Z=np.log10(α_bend),
            rstride=5,
            cstride=5,
            alpha=0.30,
            label=r"$\alpha_{bend}$"
            + f"(r, {self.core_u_name})"
            + r", $r \in [R_{min}, R_{max}$]",
        )

        # alpha_bend(r, u) wireframe, for all r in mode solver data
        u_data, r_data = np.meshgrid(
            np.linspace(self.u_domain_min, self.u_domain_max, 15),
            np.logspace(
                np.log10(self.r_α_bend_data_min),
                np.log10(self.r_α_bend_data_max),
                20,
            ),
        )
        α_bend_data: np.ndarray = np.asarray(self.α_bend(r=r_data, u=u_data))
        α_bend_data[α_bend_data < self.α_bend_data_min / 2] = self.α_bend_data_min / 2
        data_mesh = ax.plot_wireframe(
            X=u_data,
            Y=np.log10(r_data),
            Z=np.log10(α_bend_data),
            alpha=0.30,
            color="red",
            label=r"$\alpha_{bend}$"
            + f"(r, {self.core_u_name})"
            + r", $r \in$ [mode solver data]",
        )
        ax.legend(bbox_to_anchor=(0.1, 0.1))

        # alpha_prop(u) surface:
        #   red:    alpha_prop < alpha_bend
        #   green:  alpha_bend < alpha_prop < alpha_bend x 10
        #   blue:   alpha_prop > alpha_bend x 10, alpha_prop dominates
        """
        gamma: np.ndarray = np.ones_like(u_domain)
        for g, u in np.nditer([gamma, u_domain], op_flags=["readwrite"]):
            g[...] = self.gamma_of_u(u[...])
        alpha_prop: np.ndarray = self.alpha_wg_of_u() + (gamma * self.alpha_fluid)
        """
        α_prop: np.ndarray = np.ones_like(u_domain)
        for αp, u in np.nditer([α_prop, u_domain], op_flags=["readwrite"]):
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
            label=r"$\alpha_{prop}$" + f"({self.core_u_name}) surface",
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

    def _fit_α_bend_model(self):
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
        c = symbols("c:7")
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
            + f"coefficient matrix ({m.shape[1]} columns) are inconsistent!"
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
        self.α_bend = lambdify([r, u], α_bend_model_fitted, "numpy")

        # Calculate rms fitting error over the data set
        rms_error: float = float(
            np.std(
                self.ln_alpha_bend_data
                - np.log(self.α_bend(r=self.r_alpha_bend_data, u=self.u_data))
            )
        )
        self.logger(f"Fitted ln(alpha_bend) model, rms error = {rms_error:.1e}.")

    #
    # Utilities for use in optimization in the find_max_sensitivity() sensor methods
    #

    def _set_u_search_lower_bound(self):
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

        # Fetch list of core geometry values, u, in the input data dictionary
        u = list(self.bending_loss_data.keys())

        # Fit the piecewise spline to model h_lower_bound(r)
        r_α_bend_threshold: np.ndarray = np.asarray(
            [
                value["r_alpha_bend_threshold"]
                for value in self.bending_loss_data.values()
            ]
        )
        max_indx: int = int(np.argmax(r_α_bend_threshold))
        r_α_bend_threshold: np.ndarray = r_α_bend_threshold[max_indx:]
        if len(r_α_bend_threshold) <= 1:
            raise ValueError(
                f"{Fore.YELLOW}'ERROR: alpha_bend_threshold' value us too high!"
                + f"{Style.RESET_ALL}"
            )
        u_α_bend_threshold: np.ndarray = np.asarray(u[max_indx:])
        self.r_min_for_u_search_lower_bound = r_α_bend_threshold[-1]
        self.r_max_for_u_search_lower_bound = r_α_bend_threshold[0]
        self.u_lower_bound = interpolate.interp1d(
            x=r_α_bend_threshold, y=u_α_bend_threshold
        )
        if self.r_max_for_u_search_lower_bound < 500:
            self.logger(
                f"{self.core_u_name} search domain lower bound constrained for R < "
                + f"{self.r_max_for_u_search_lower_bound:.1f} um"
            )
        else:
            self.logger(
                f"{Fore.YELLOW}WARNING!: {self.core_u_name} search domain lower"
                + f"bound ({self.r_max_for_u_search_lower_bound:.1f} um) is unusually "
                + "high, raise value of 'alpha_bend_threshold' in .toml file."
                + f"{Style.RESET_ALL}"
            )

        # Plot u_lower_bound(r) data and spline interpolation
        fig, axs = plt.subplots()
        fig.suptitle(
            "".join(
                [
                    f"Search domain lower bound for {self.core_u_name}",
                    " at low radii in the optimization\n",
                    f"{self.pol}",
                ]
            )
            + "".join([r", $\lambda$", f" = {self.lambda_res:.3f} ", r"$\mu$m"])
            + "".join([f", {self.core_v_name} = {self.core_v_value:.3f} ", r"$\mu$m"])
            + (
                "".join(
                    [
                        f"\nWARNING!: {self.core_u_name} search domain lower bound ",
                        f"({self.r_max_for_u_search_lower_bound:.1f} um) is VERY HIGH!",
                    ]
                )
                if self.r_max_for_u_search_lower_bound > 500
                else ""
            ),
            color="black" if self.r_max_for_u_search_lower_bound < 500 else "red",
        )
        r_α_bend_threshold_i: np.ndarra = np.linspace(
            self.r_min_for_u_search_lower_bound,
            self.r_max_for_u_search_lower_bound,
            100,
        )
        axs.plot(r_α_bend_threshold, u_α_bend_threshold, "o")
        axs.plot(r_α_bend_threshold_i, self.u_lower_bound(r_α_bend_threshold_i))
        axs.set_xlabel(r"R ($\mu$m)")
        axs.set_ylabel(r"min$\{$" + f"{self.core_u_name}" + r"$\}$ ($\mu$m)")
        out_filename: str = str(
            (
                self.filename_path.parent
                / "".join(
                    [
                        f"{self.filename_path.stem}_FITTING_"
                        + f"{'H' if self.core_v_name == 'w' else 'W'}_SRCH_LOWR_BND.png"
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
                f"{Fore.YELLOW}ERROR! Search domain lower bound fit:"
                + "R values are not monotonically decreasing "
                + f"(see {out_filename})! "
                + "Decrease value of 'alpha_bend_threshold' in .toml file."
                + f"{Style.RESET_ALL}"
            )

    def u_search_domain(self, r: float) -> tuple[float, float]:
        """
        Determine u search domain extrema, see _set_u_search_lower_bound()

        :param r: waveguide bending radius (um)
        :return: (u_min, u_max) u search domain extrema (um)
        """

        if self.disable_u_search_lower_bound:
            return self.u_domain_min, self.u_domain_max
        if r < self.r_min_for_u_search_lower_bound:
            u_min = self.u_domain_max
        elif r > self.r_max_for_u_search_lower_bound:
            u_min = self.u_domain_min
        else:
            u_min = min(float(self.u_lower_bound(r)), self.u_domain_max)
            u_min = max(u_min, self.u_domain_min)
        u_max = self.u_domain_max
        return u_min, u_max

    #
    # Calculate shared values for sensor Class instance plotting
    #

    def calculate_plotting_extrema(self, max_s: float):
        """

        Returns:

        """

        # R domain extrema (complete decades)
        self.plotting_extrema["r_plot_min"] = 10 ** (np.floor(np.log10(self.r_min)))
        self.plotting_extrema["r_plot_max"] = 10 ** (np.ceil(np.log10(self.r_max)))

        # u domain extrema
        u: np.ndarray = np.asarray(list(self.bending_loss_data.keys()))
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
