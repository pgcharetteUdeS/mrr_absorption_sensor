"""
Models class

Exposed methods:
    - gamma()
    - neff()
    - alpha_bend(r, h)
    - h_search_domain()

    NB: The model for alpha_bend(r, h) is hardcoded in fit_alpha_bend_model()
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
import sys
from typing import Callable


class Models:
    """
    Models class for polynomial interpolation of gamma(h), neffs(h), alpha_bend(r, h)

    All lengths are in units of um
    """

    def __init__(
        self,
        modes_data: dict,
        bending_loss_data: dict,
        Rmin: float,
        Rmax: float,
        R_samples_per_decade: int,
        lambda_res: float,
        pol: str,
        core_width: float,
        ni_op: float,
        alpha_wg_dB_per_cm: float,
        filename_path: Path,
        alpha_bend_threshold: float,
        gamma_order,
        neff_order,
        disable_R_domain_check: bool = False,
        logger: Callable = print,
    ):

        # Load class instance input parameters
        self.modes_data: dict = modes_data
        self.bending_loss_data: dict = bending_loss_data
        self.Rmin: float = Rmin
        self.Rmax: float = Rmax
        self.R_samples_per_decade: int = R_samples_per_decade
        self.lambda_res: float = lambda_res
        self.pol: str = pol
        self.core_width: float = core_width
        self.ni_op: float = ni_op
        self.alpha_wg_dB_per_cm: float = alpha_wg_dB_per_cm
        self.filename_path: Path = filename_path
        self.alpha_bend_threshold: float = alpha_bend_threshold
        self.gamma_order = gamma_order
        self.neff_order = neff_order
        self.logger: Callable = logger

        # Define the array of radii to be analyzed (R domain)
        self.R: np.ndarray = np.logspace(
            start=np.log10(self.Rmin),
            stop=np.log10(self.Rmax),
            num=int((np.log10(Rmax) - np.log10(Rmin)) * self.R_samples_per_decade),
            base=10,
        )

        # Define propagation losses in the waveguide core and in the fluid medium (1/um)
        self.alpha_wg: float = self.alpha_wg_dB_per_cm / 4.34 / 10000
        self.alpha_fluid: float = (4 * np.pi / self.lambda_res) * self.ni_op

        # Parse and validate the mode solver data loaded from the .toml file
        self.h_data: np.ndarray = np.asarray([])
        self.R_data: np.ndarray = np.asarray([])
        self.ln_alpha_bend_data: np.ndarray = np.asarray([])
        self.h_domain_min: float = 0
        self.h_domain_max: float = 0
        self.R_data_min: float = 0
        self.R_data_max: float = 0
        self.alpha_bend_data_min: float = 0
        self.alpha_bend_data_max: float = 0
        self._parse_bending_loss_mode_solver_data()

        # Fit gamma(h) and neff(h) 1D polynomial models to the mode solver data
        self.gamma_model: dict = {}
        self.neff_model: dict = {}
        self._fit_gamma_and_neff_models()

        # Check that the bending loss mode solver data covers the required h & R ranges
        alpha_prop_min: float = self.alpha_wg + (
            self.gamma(self.h_domain_max) * self.alpha_fluid
        )
        if self.alpha_bend_data_min > alpha_prop_min / 100:
            self.logger(
                f"{Fore.YELLOW}WARNING! Mode solver arrays must contain alpha_bend data"
                + f" down to min(alpha_prop)/100 ({alpha_prop_min/100:.2e} um-1)!"
                + f"{Style.RESET_ALL}"
            )
            if not disable_R_domain_check:
                sys.exit()
        if self.R_data_min > self.Rmin:
            self.logger(
                f"{Fore.YELLOW}WARNING! Mode solver arrays must contain data"
                + f" at radii < Rmin!{Style.RESET_ALL}"
            )
            if not disable_R_domain_check:
                sys.exit()

        # Fit alpha_bend(r, h) 2D polynomial model to the mode solver data
        self.alpha_bend_model = lambda *args, **kwargs: 0
        self._alpha_bend_model_fig: dict = {}
        self.alpha_bend_model_symbolic: functions.exp | None = None
        self._alpha_bend_model_fig_azim: float = 0
        self._alpha_bend_model_fig_elev: float = 0
        self._fit_alpha_bend_model()
        self._plot_alpha_bend_model_fitting_results()

        # For use in the optimization: constrain the search domain for h at low radii,
        # else the optimization erroneously converges to a local minima.
        self.r_min_for_h_search_lower_bound: float = 0
        self.r_max_for_h_search_lower_bound: float = 0
        self.h_lower_bound: interpolate.interp1d = interpolate.interp1d([0, 1], [0, 1])
        self._set_h_search_lower_bound()

        # Display plots
        plt.show()

    #
    # gamma(h) and neffs(h) modeling
    #

    # Wrappers for models-specific calls to _interpolate()
    def gamma(self, h: float) -> float:
        """

        :param h:  waveguide core height (um)
        :return: gamma(h) polynomial model estimate
        """
        return self._interpolate(model=self.gamma_model, h=h)

    def neff(self, h: float) -> float:
        """

        :param h: waveguide core height (um)
        :return:  neff(h) polynomial model estimate
        """
        return self._interpolate(model=self.neff_model, h=h)

    @staticmethod
    def _interpolate(model: dict, h: float) -> float:
        """
        Interpolate gamma(h) and neffs(h) at core height h with models fitted
        by _fit_gamma_and_neff_models(), limit h to the allowed bounds.
        """

        value: float = model["model"](h)
        value = max(model["min"], value)
        value = min(model["max"], value)

        return value

    def _fit_gamma_and_neff_models(self):
        """
        Fit polynomial models to gamma(h) and neffs(h), load the info
        (model parameters, bounds) into dictionaries for each.
        """

        # gamma
        h_data_modes = np.asarray(
            [value.get("h") for value in self.modes_data.values()]
        )
        gamma_data = np.asarray(
            [value.get("gamma") for value in self.modes_data.values()]
        )
        self.gamma_model = {
            "name": "gamma",
            "model": Polynomial.fit(x=h_data_modes, y=gamma_data, deg=self.gamma_order),
            "min": 0,
            "max": 1,
        }

        # neff
        neff_data = np.asarray(
            [value.get("neff") for value in self.modes_data.values()]
        )
        self.neff_model = {
            "name": "neff",
            "model": Polynomial.fit(x=h_data_modes, y=neff_data, deg=self.neff_order),
            "min": np.amin(neff_data),
            "max": np.amax(neff_data),
        }

        # Plot interpolated and dataset values
        fig, axs = plt.subplots(2)
        fig.suptitle(
            "1D model fits over the h domain\n"
            + f"{self.pol}"
            + "".join([r", $\lambda$", f" = {self.lambda_res:.3f} ", r"$\mu$m"])
            + "".join([r", $\alpha_{wg}$", f" = {self.alpha_wg_dB_per_cm:.1f} dB/cm"])
            + "".join([f", w = {self.core_width:.3f} ", r"$\mu$m"])
        )
        axs_index: int = 0
        h_interp: np.ndarray = np.linspace(h_data_modes[0], h_data_modes[-1], 100)
        gamma_interp = [100 * self.gamma(h) for h in h_interp]
        axs[axs_index].plot(h_data_modes, gamma_data * 100, ".")
        axs[axs_index].plot(h_interp, gamma_interp)
        axs[axs_index].set_title(
            f"Gamma(h), polynomial model order: {self.gamma_order}"
        )
        axs[axs_index].set_ylabel("Gamma (%)")
        axs_index += 1
        neff_interp = [self.neff(h) for h in h_interp]
        axs[axs_index].plot(h_data_modes, neff_data, ".")
        axs[axs_index].plot(h_interp, neff_interp)
        axs[axs_index].set_title(
            r"n$_{eff}$" + f"(h), polynomial model order: {self.neff_order}"
        )
        axs[axs_index].set_ylabel(r"n$_{eff}$ (RIU)")
        axs[axs_index].set_xlabel(r"h ($\mu$m)")
        fig.tight_layout()
        out_filename: str = str(
            (
                self.filename_path.parent
                / f"{self.filename_path.stem}_POLY_1D_MODELS.png"
            )
        )

        fig.savefig(out_filename)
        self.logger(f"Wrote '{out_filename}'.")

    #
    # alpha_bend(r, h) modeling
    #

    def _parse_bending_loss_mode_solver_data(self):
        """
        Parse the alpha_bend(R, h) mode solver data, determine the extrema,
        build the "h / R / log(alpha_bend)" arrays for fitting.
        """

        # For each h entry in the input dictionary: determine the radius value
        # "r_alpha_bend_threshold" that corresponds to the value of
        # "alpha_bend_threshold" specified in the input file, by fitting
        # a first order polynomial (parameters A&B) to "ln(alpha_bend) = ln(A) - R*B".
        for key, value in self.bending_loss_data.items():
            lnA, negB = (
                Polynomial.fit(x=value["R"], y=np.log(value["alpha_bend"]), deg=1)
                .convert()
                .coef
            )
            self.bending_loss_data[key]["r_alpha_bend_threshold"] = (
                lnA - np.log(self.alpha_bend_threshold)
            ) / -negB

        # Loop to build "h / R / log(alpha_bend) / w"  data arrays for fitting
        for h_key_um, value in self.bending_loss_data.items():
            self.h_data = np.append(self.h_data, h_key_um * np.ones_like(value["R"]))
            self.R_data = np.append(self.R_data, np.asarray(value["R"]))
            self.ln_alpha_bend_data = np.append(
                self.ln_alpha_bend_data, np.log(value["alpha_bend"])
            )

        # Determine dynamic range extrema of the bending loss data
        self.h_domain_min: float = list(self.bending_loss_data.keys())[0]
        self.h_domain_max: float = list(self.bending_loss_data.keys())[-1]
        self.alpha_bend_data_min = np.exp(min(self.ln_alpha_bend_data))
        self.alpha_bend_data_max = np.exp(max(self.ln_alpha_bend_data))
        self.R_data_min = min(self.R_data)
        self.R_data_max = max(self.R_data)

    def alpha_bend(self, r: float, h: float) -> float:
        """
        Interpolate bending loss coefficient at radius r and core height h

        :param r: waveguide bending radius (h)
        :param h: waveguide core height (um)
        :return: alpha_bend(r, h) polynomial model estimate (um-1)
        """

        # Calculate the alpha_bend(r, h) model estimate
        alpha_bend_val: float = float(self.alpha_bend_model(r=r, h=h))

        # Impose a lower bound on bending losses. Because of the concentric rings
        # and S-bend in the spiral, the bending losses will be much higher than
        # in the mode solver data, so no upper bound is imposed.
        alpha_bend_val = max(self.alpha_bend_data_min, alpha_bend_val)

        return alpha_bend_val

    def _alpha_bend_model_fig_check_button_callback(self, label):
        index = self._alpha_bend_model_fig["labels"].index(label)
        self._alpha_bend_model_fig["lines"][index].set_visible(
            not self._alpha_bend_model_fig["lines"][index].get_visible()
        )
        plt.draw()

    def _alpha_bend_model_fig_slider_callback(self, *_, **__):
        self._alpha_bend_model_fig["surface"].set_alpha(
            self._alpha_bend_model_fig["slider"].val
        )
        plt.draw()

    def _alpha_bend_model_fig_save(self, *_, **__):
        self._alpha_bend_model_fig["fig"].savefig(
            self._alpha_bend_model_fig["out_filename"]
        )
        self.logger(f"Wrote '{self._alpha_bend_model_fig['out_filename']}'.")

    def _alpha_bend_model_fig_top_view(self, *_, **__):
        self._alpha_bend_model_fig["ax"].view_init(azim=0, elev=90)
        plt.draw()

    def _alpha_bend_model_fig_reset_view(self, *_, **__):
        self._alpha_bend_model_fig["ax"].view_init(
            azim=self._alpha_bend_model_fig_azim, elev=self._alpha_bend_model_fig_elev
        )
        plt.draw()

    def _plot_alpha_bend_model_fitting_results(self):
        # Calculate rms error over the data points
        alpha_bend_fitted: np.ndarray = np.asarray(
            self.alpha_bend_model(r=self.R_data, h=self.h_data)
        )
        rms_error: float = float(
            np.std(self.ln_alpha_bend_data - np.log(alpha_bend_fitted))
        )

        # Plot model solver data, fitted points, and 3D wireframes & surface
        # to verify the goodness of fit of the alpha_bend(r, h) model.
        self._alpha_bend_model_fig["fig"] = plt.figure()
        ax = self._alpha_bend_model_fig["fig"].add_subplot(projection="3d")
        self._alpha_bend_model_fig["ax"] = ax
        ax.set_title(
            r"$\alpha_{bend}(r, h)$ = "
            + str(self.alpha_bend_model_symbolic)
            + "\nWireframes: "
            + r"fitted $\alpha_{bend}(r, h)$ model, "
            + "".join(
                [f"rms error = {rms_error:.1e}", r" $\mu$m$^{-1}$", " (logarithmic)\n"]
            )
            + r"Surface : $\alpha_{prop}(h) = \alpha_{wg} "
            + r"+ \Gamma_{fluide}(h)\times\alpha_{fluid}$, where "
            + "".join(
                [r"$\alpha_{wg}$", f" = {self.alpha_wg:.2e} ", r"$\mu$m$^{-1}$ and "]
            )
            + "".join(
                [r"$\alpha_{fluid}$", f" = {self.alpha_fluid:.2e} ", r"$\mu$m$^{-1}$"]
            )
            + "\n("
            + "".join(
                [
                    "Green : ",
                    r"$\alpha_{bend} < \alpha_{prop}$ ",
                    r"$< \alpha_{bend}\times 10$",
                ]
            )
            + "".join(
                [
                    r", blue : $\alpha \approx \alpha_{prop}$",
                ]
            )
            + ")\n"
            + r"$\alpha(r, h) = \alpha_{bend}(r, h) + \alpha_{prop}(h)$",
            y=1.02,
        )
        ax.set_xlabel(r"$h$ ($\mu$m)")
        ax.set_ylabel(r"log($R$) ($\mu$m)")
        ax.set_zlabel(r"log$_{10}$($\alpha_{BEND}$) ($\mu$m$^{-1}$)")
        raw_points = ax.scatter(
            self.h_data,
            np.log10(self.R_data),
            self.ln_alpha_bend_data * np.log10(np.e),
            color="red",
            label="Mode solver points (raw)",
            s=1,
        )
        fitted_points = ax.scatter(
            self.h_data,
            np.log10(self.R_data),
            np.log10(alpha_bend_fitted),
            color="blue",
            label="Mode solver points (fitted)",
            s=1,
        )

        # alpha_bend(r, h) wireframe, for r in [Rmin, Rmax] analysis domain
        h_domain, r_domain = np.meshgrid(
            np.linspace(self.h_domain_min, self.h_domain_max, 75),
            np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), 100),
        )
        alpha_bend: np.ndarray = np.asarray(
            self.alpha_bend_model(r=r_domain, h=h_domain)
        )
        alpha_bend[alpha_bend < self.alpha_bend_data_min / 2] = (
            self.alpha_bend_data_min / 2
        )
        model_mesh = ax.plot_wireframe(
            X=h_domain,
            Y=np.log10(r_domain),
            Z=np.log10(alpha_bend),
            rstride=5,
            cstride=5,
            alpha=0.30,
            label=r"$\alpha_{bend}(r, h)$, $r \in$ [R$_{min}$, R$_{max}$]",
        )

        # alpha_bend(r, h) wireframe, for all r in mode solver data
        h_data, r_data = np.meshgrid(
            np.linspace(self.h_domain_min, self.h_domain_max, 15),
            np.logspace(np.log10(self.R_data_min), np.log10(self.R_data_max), 20),
        )
        alpha_bend_data: np.ndarray = np.asarray(
            self.alpha_bend_model(r=r_data, h=h_data)
        )
        alpha_bend_data[alpha_bend_data < self.alpha_bend_data_min / 2] = (
            self.alpha_bend_data_min / 2
        )
        data_mesh = ax.plot_wireframe(
            X=h_data,
            Y=np.log10(r_data),
            Z=np.log10(alpha_bend_data),
            alpha=0.30,
            color="red",
            label=r"$\alpha_{bend}(r, h)$, $r \in$ [mode solver data]",
        )
        ax.legend(bbox_to_anchor=(0.1, 0.1))

        # alpha_prop(h) surface:
        #   red:    alpha_prop < alpha_bend
        #   green:  alpha_bend < alpha_prop < alpha_bend x 10
        #   blue:   alpha_prop > alpha_bend x 10, alpha_prop dominates
        gamma: np.ndarray = np.ones_like(h_domain)
        for g, h in np.nditer([gamma, h_domain], op_flags=["readwrite"]):
            g[...] = self.gamma(h[...])
        alpha_prop: np.ndarray = self.alpha_wg + (gamma * self.alpha_fluid)
        face_colors: np.ndarray = np.copy(
            np.broadcast_to(colors.to_rgba(c="red", alpha=0.8), h_domain.shape + (4,))
        )
        for index in np.ndindex(h_domain.shape):
            if alpha_prop[index] > alpha_bend[index]:
                if alpha_prop[index] > alpha_bend[index] * 10:
                    face_colors[index] = colors.to_rgba(c="blue", alpha=0.5)
                else:
                    face_colors[index] = colors.to_rgba(c="green", alpha=1)
        alpha_prop_surface: PolyCollection = ax.plot_surface(
            X=h_domain,
            Y=np.log10(r_domain),
            Z=np.log10(alpha_prop),
            facecolors=face_colors,
            label=r"$\alpha_{prop}(h)$ surface",
            linewidth=0,
            rstride=1,
            cstride=1,
        )
        self._alpha_bend_model_fig["surface"] = alpha_prop_surface

        # Add "check button" to figure to turn visibility of individual plots on/off
        self._alpha_bend_model_fig["lines"] = [
            raw_points,
            fitted_points,
            model_mesh,
            data_mesh,
            alpha_prop_surface,
        ]
        self._alpha_bend_model_fig["lines"][3].set_visible(False)
        self._alpha_bend_model_fig["labels"] = [
            str(line.get_label()) for line in self._alpha_bend_model_fig["lines"]
        ]
        self._alpha_bend_model_fig["check_button"] = CheckButtons(
            ax=plt.axes([0.01, 0.71, 0.30, 0.15]),
            labels=self._alpha_bend_model_fig["labels"],
            actives=([True, True, True, False, True]),
        )
        self._alpha_bend_model_fig["check_button"].on_clicked(
            self._alpha_bend_model_fig_check_button_callback
        )

        # Add "save" button, to save the image of the 3D plot to file
        self._alpha_bend_model_fig["button_save"] = Button(
            ax=plt.axes([0.01, 0.4, 0.10, 0.05]), label="Save", color="white"
        )
        self._alpha_bend_model_fig["button_save"].on_clicked(
            self._alpha_bend_model_fig_save
        )

        # Add "Reset" and "Top view" buttons, to either reset the 3D view
        # to the initial perspective, or to a pure top-down view.
        self._alpha_bend_model_fig["button_reset_view"] = Button(
            ax=plt.axes([0.01, 0.5, 0.10, 0.05]), label="Reset view", color="white"
        )
        self._alpha_bend_model_fig["button_reset_view"].on_clicked(
            self._alpha_bend_model_fig_reset_view
        )
        self._alpha_bend_model_fig["button_top_view"] = Button(
            ax=plt.axes([0.01, 0.56, 0.10, 0.05]), label="Top view", color="white"
        )
        self._alpha_bend_model_fig["button_top_view"].on_clicked(
            self._alpha_bend_model_fig_top_view
        )

        # Rotate plot and write to file
        self._alpha_bend_model_fig_azim = 40
        self._alpha_bend_model_fig_elev = 30
        ax.view_init(
            azim=self._alpha_bend_model_fig_azim, elev=self._alpha_bend_model_fig_elev
        )
        self._alpha_bend_model_fig["out_filename"] = str(
            (
                self.filename_path.parent
                / f"{self.filename_path.stem}_POLY_3D_MODEL_FIT.png"
            )
        )

        self._alpha_bend_model_fig_save()

    def _fit_alpha_bend_model(self):
        """
        Polynomial model least squares fit to ln(alpha_bend(r, h))

        Current model:

        ln(alpha_bend) = c0 + c1*h + c2*r + c3*h*r + c4*h**2*r + c5*h**3*r + c6*h**4

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
        r, h = symbols("r, h")
        c = symbols("c:7")
        self.alpha_bend_model_symbolic: functions.exp = functions.exp(
            c[0]
            + c[1] * h
            + c[2] * r
            + c[3] * (h * r)
            + c[4] * (h**2 * r)
            + c[5] * (h**3 * r)
            + c[6] * (h**4)
        )

        # Assemble the "M" coefficient matrix, where each column holds the values of
        # the monomial terms in the model for each r & h value pair in the input data.
        M: np.nedarray = np.asarray(
            [
                np.ones_like(self.h_data),
                self.h_data,
                self.R_data,
                self.h_data * self.R_data,
                self.h_data**2 * self.R_data,
                self.h_data**3 * self.R_data,
                self.h_data**4,
            ]
        ).T

        #
        # END OF USER-DEFINABLE MODEL-SPECIFIC SECTION
        #

        # Check that the model definition and the coefficient matrix are consistent
        assert len(c) == M.shape[1], (
            f"Model ({len(c)} coefficients) and "
            + f"coefficient matrix ({M.shape[1]} columns) are inconsistent!"
        )

        # Linear least-squares fit to "ln(alpha_bend(r, h)" to determine the values of
        # the model coefficients. Although the model contains an exponential, the model
        # without the exponential is fitted to ln(alpha_bend(r, h)), to use
        # LINEAR least squares.
        c_fitted, residual, rank, s = lstsq(M, self.ln_alpha_bend_data)
        assert rank == len(
            c_fitted
        ), f"Matrix rank ({rank}) is not equal to model order+1 ({len(c_fitted)})!"

        # Insert the fitted coefficient values into the model, then convert the symbolic
        # model to a lambda function for faster evaluation.
        alpha_bend_model_fitted = self.alpha_bend_model_symbolic.subs(
            list(zip(c, c_fitted))
        )
        self.alpha_bend_model = lambdify([r, h], alpha_bend_model_fitted, "numpy")

        # Calculate rms fitting error over the data set
        rms_error: float = float(
            np.std(
                self.ln_alpha_bend_data
                - np.log(self.alpha_bend_model(r=self.R_data, h=self.h_data))
            )
        )
        self.logger(f"Fitted ln(alpha_bend) model, rms error = {rms_error:.1e}.")

    #
    # Utilities for use in optimization in the find_max_sensitivity() sensor methods
    #

    def _set_h_search_lower_bound(self):
        # At small ring radii, the interpolation model for alpha_bend(h, r) is
        # unreliable at low h. As a result, the search for optimal h in the optimization
        # sometimes converges towards a solution at low h whereas the solution at small
        # radii lies necessarily at high h to minimize bending losses. To mitigate this
        # problem, the search domain lower bound for h is constrained at small radii.
        #
        # The point at which the ring radius is considered "small", i.e. where the
        # alpha_bend interpolation model fails, is h-dependant. This boundary
        # is determined by calculating the radius at each h for which
        # alpha_bend exceeds a user-specified threshold ("alpha_bend_threshold"),
        # values are stored in the array "r_alpha_bend_threshold",
        # see fit_gamma_and_neff_models(). A spline interpolation is used to model the
        # h search domain lower bound as a function of radius.
        #
        # For radii greater than the spline r domain, h is allowed to take on any value
        # in the full h domain during optimization.

        # Fetch list of heights, h, in the input data dictionary
        h = list(self.bending_loss_data.keys())

        # Fit the piecewise spline to model h_lower_bound(r)
        R_alpha_bend_threshold: np.ndarray = np.asarray(
            [
                value["r_alpha_bend_threshold"]
                for value in self.bending_loss_data.values()
            ]
        )
        max_indx: int = int(np.argmax(R_alpha_bend_threshold))
        R_alpha_bend_threshold: np.ndarray = R_alpha_bend_threshold[max_indx:]
        h_alpha_bend_threshold: np.ndarray = np.asarray(h[max_indx:])
        self.r_min_for_h_search_lower_bound = R_alpha_bend_threshold[-1]
        self.r_max_for_h_search_lower_bound = R_alpha_bend_threshold[0]
        self.h_lower_bound = interpolate.interp1d(
            x=R_alpha_bend_threshold, y=h_alpha_bend_threshold
        )
        self.logger(
            "h search domain lower bound constrained for R < "
            + f"{self.r_max_for_h_search_lower_bound:.1f} um"
        )

        # Plot h_lower_bound(r) data and spline interpolation
        fig, axs = plt.subplots()
        fig.suptitle(
            "Search domain lower bound for h at low radii in the optimization\n"
            + f"{self.pol}"
            + "".join([r", $\lambda$", f" = {self.lambda_res:.3f} ", r"$\mu$m"])
            + "".join([r", $\alpha_{wg}$", f" = {self.alpha_wg_dB_per_cm:.1f} dB/cm"])
            + "".join([f", w = {self.core_width:.3f} ", r"$\mu$m"])
        )
        R_alpha_bend_threshold_i: np.ndarra = np.linspace(
            self.r_min_for_h_search_lower_bound,
            self.r_max_for_h_search_lower_bound,
            100,
        )
        axs.plot(R_alpha_bend_threshold, h_alpha_bend_threshold, "o")
        axs.plot(R_alpha_bend_threshold_i, self.h_lower_bound(R_alpha_bend_threshold_i))
        axs.set_xlabel(r"R ($\mu$m)")
        axs.set_ylabel(r"min$\{h\}$ ($\mu$m)")
        out_filename: str = str(
            (
                self.filename_path.parent
                / f"{self.filename_path.stem}_H_SEARCH_LOWER_BOUND.png"
            )
        )

        fig.savefig(out_filename)
        self.logger(f"Wrote '{out_filename}'.")

    def h_search_domain(self, r: float) -> tuple[float, float]:
        """
        Determine h search domain extrema, see _set_h_search_lower_bound()

        :param r: waveguide bending radius (um)
        :return: (h_min, h_max) h search domain extrema (um)
        """

        if r < self.r_min_for_h_search_lower_bound:
            h_min = self.h_domain_max
        elif r > self.r_max_for_h_search_lower_bound:
            h_min = self.h_domain_min
        else:
            h_min = min(self.h_lower_bound(r), self.h_domain_max)
            h_min = max(h_min, self.h_domain_min)
        h_max = self.h_domain_max

        return h_min, h_max
