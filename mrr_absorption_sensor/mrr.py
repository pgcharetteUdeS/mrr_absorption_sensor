"""

Micro-ring resonator sensor class

Exposed methods:
    - analyze()
    - plot_optimization_results()
    - plot_combined_linear_mrr_spiral_optimization_results()

"""


from math import e
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from openpyxl.workbook import Workbook
from scipy import optimize
from scipy.special import lambertw

from .constants import constants, LINE_STYLES
from .linear import Linear
from .models import Models
from .spiral import Spiral


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
        self.lc: float = self.models.parameters["lc"]
        self.previous_solution: float = -1

        # Define class instance result variables and arrays
        self.α_bend_a: np.ndarray = np.ndarray([])
        self.α_bend_b: np.ndarray = np.ndarray([])
        self.α_bend: np.ndarray = np.ndarray([])
        self.α_wg: np.ndarray = np.ndarray([])
        self.α_prop: np.ndarray = np.ndarray([])
        self.α: np.ndarray = np.ndarray([])
        self.αl: np.ndarray = np.ndarray([])
        self.wg_a2: np.ndarray = np.ndarray([])
        self.er: np.ndarray = np.ndarray([])
        self.contrast: np.ndarray = np.ndarray([])
        self.finesse: np.ndarray = np.ndarray([])
        self.fsr: np.ndarray = np.ndarray([])
        self.fwhm: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.gamma_resampled: np.ndarray = np.ndarray([])
        self.max_s: float = 0
        self.max_s_radius: float = 0
        self.n_eff: np.ndarray = np.ndarray([])
        self.plotting_extrema: dict = {}
        self.q: np.ndarray = np.ndarray([])
        self.results: list = []
        self.s: np.ndarray = np.ndarray([])
        self.r_e: np.ndarray = np.ndarray([])
        self.r_w: np.ndarray = np.ndarray([])
        self.s_e: np.ndarray = np.ndarray([])
        self.s_nr: np.ndarray = np.ndarray([])
        self.t_max: np.ndarray = np.ndarray([])
        self.t_min: np.ndarray = np.ndarray([])
        self.tau: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.u_resampled: np.ndarray = np.ndarray([])

    #
    # Plotting
    #

    def _calculate_plotting_extrema(self):
        # Other extrema for Mrr plots
        self.plotting_extrema["Se_plot_max"] = (
            np.ceil(np.amax(self.s_e * np.sqrt(self.wg_a2)) * 1.1 / 10) * 10
        )
        self.plotting_extrema["Finesse_plot_max"] = (
            np.ceil(np.amax(self.finesse / (2 * np.pi)) * 1.1 / 10) * 10
        )

    @staticmethod
    def _write_image_data_to_excel(
        filename: str,
        x_array: np.ndarray,
        x_label: str,
        y_array: np.ndarray,
        y_label: str,
        z_array: list,
        z_labels: list,
    ):
        """
        Write image data to Excel file
        """

        wb = Workbook()

        # X sheet
        x_sheet = wb["Sheet"]
        x_sheet.title = x_label
        x_sheet.append(x_array.tolist())

        # Y sheet
        y_sheet = wb.create_sheet(y_label)
        for y in y_array:
            y_sheet.append([y])

        # Z sheets
        for i, Z in enumerate(z_array):
            z_sheet = wb.create_sheet(z_labels[i])
            for z in Z:
                z_sheet.append(z.tolist())

        # Save file
        wb.save(filename=filename)

    def _plot_mrr_result_2d_maps(self):
        """

        Returns:

        """

        # Generate 2D map row/column index R,u arrays (x/y)
        r_2d_map_indices = np.linspace(
            np.log10(self.models.r[0]),
            np.log10(self.models.r[-1]),
            self.models.parameters["map2D_n_grid_points"],
        )
        u_2d_map_indices = np.linspace(
            list(self.models.bending_loss_data)[0],
            list(self.models.bending_loss_data)[-1],
            self.models.parameters["map2D_n_grid_points"],
        )

        # Indices for dashed lines at radii for max(Smrr)
        r_max_s_mrr_index: int = int(
            (np.abs(self.models.r - self.max_s_radius)).argmin()
        )
        r_max_s_mrr_u: float = self.u[r_max_s_mrr_index]

        #
        # 2D maps as a function of R/u
        #

        # 2D map of S(u, R)
        s_2d_map = np.asarray(
            [
                [
                    self._calc_sensitivity(r=10**log10_R, u=u)
                    for log10_R in r_2d_map_indices
                ]
                for u in u_2d_map_indices
            ]
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            u_2d_map_indices,
            s_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.invert_yaxis()
        ax.set_title(
            f"MRR sensitivity as a function of {self.models.core_u_name} and R\n"
            + f"{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(f"{self.models.core_u_name} (μm)")
        fig.colorbar(cm, label=r"S (RIU $^{-1}$)")
        max_s_u_min_index: int = np.where(self.s > self.max_s / 25)[0][0]
        ax.plot(
            np.log10(self.models.r[max_s_u_min_index:]),
            self.u[max_s_u_min_index:],
            color=self.models.parameters["map2D_overlay_color_light"],
            label=r"max$\{S(h, R)\}$",
        )
        max_s_u_line_search: np.ndarray = u_2d_map_indices[np.argmax(s_2d_map, axis=0)]
        max_s_u_line_search_min_index: int = np.where(
            np.amax(s_2d_map, axis=0) > self.max_s / 25
        )[0][0]
        ax.plot(
            r_2d_map_indices[max_s_u_line_search_min_index:],
            max_s_u_line_search[max_s_u_line_search_min_index:],
            "k--",
            label=r"max$\{S(h, R)\}$ - line search",
        )
        """
        ax.plot(
            [np.log10(self.max_S_radius), np.log10(self.max_S_radius)],
            [u_2D_map[-1], R_max_Smrr_u],
            color=self.models.parameters["map2D_overlay_color_dark"],
        )
        """
        ax.plot(
            [r_2d_map_indices[0], np.log10(self.max_s_radius)],
            [r_max_s_mrr_u, r_max_s_mrr_u],
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle=LINE_STYLES["loosely dashdotted"],
            label=rf"max{{max{{$S_{{MRR}}$}}}} = {self.max_s:.0f} RIU $^{{-1}}$"
            + f" @ R = {self.max_s_radius:.0f} μm"
            + f", {self.models.core_u_name} = {r_max_s_mrr_u:.3f} μm",
        )
        ax.plot(
            np.log10(self.r_e),
            self.u_resampled,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="--",
            label=r"Re$(\Gamma_{fluid})$",
        )
        ax.plot(
            np.log10(self.r_w),
            self.u_resampled,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="-.",
            label=r"Rw$(\Gamma_{fluid})$",
        )
        ax.legend(loc="lower right")
        name = (
            f"{self.models.filename_path.stem}_MRR_2DMAP_S_VS_"
            + f"{self.models.core_u_name}_and_R.png"
        )
        filename = self.models.filename_path.parent / name
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")
        if self.models.parameters["write_excel_files"]:
            fname = (
                f"{self.models.filename_path.stem}_MRR_2DMAPS_VS_"
                + f"{self.models.core_u_name}_and_R.xlsx"
            )
            self._write_image_data_to_excel(
                filename=str(self.models.filename_path.parent / fname),
                x_array=10**r_2d_map_indices,
                x_label="R (um)",
                y_array=u_2d_map_indices,
                y_label=f"{self.models.core_u_name} (um)",
                z_array=[s_2d_map],
                z_labels=["S (RIU-1)"],
            )
            self.logger(f"Wrote '{filename.with_suffix('.xlsx')}'.")

        #
        # 2D maps as a function of R/gamma
        #

        # Generate gamma(u) array matching u array. If the values are not monotonically
        # decreasing due to positive curvature of the modeled values at the beginning of
        # the array, flag as warning and replace values, else pcolormesh() complains.
        gamma_2d_map = np.asarray(
            [self.models.gamma_of_u(u) * 100 for u in u_2d_map_indices]
        )
        if np.any(np.diff(gamma_2d_map) > 0):
            gamma_2d_map[: int(np.argmax(gamma_2d_map))] = gamma_2d_map[
                int(np.argmax(gamma_2d_map))
            ]
            gamma_2d_map[int(np.argmin(gamma_2d_map)) :] = gamma_2d_map[
                int(np.argmin(gamma_2d_map))
            ]
            self.logger(
                f"{Fore.YELLOW}WARNING! Gamma({self.models.core_u_name}) is not "
                + "monotonically decreasing, first/last values replaced"
                + f" with gamma max/min!{Style.RESET_ALL}"
            )

        # Indices for dashed lines at radii for max(Smrr)
        r_max_s_mrr_gamma: float = self.gamma[r_max_s_mrr_index]

        # 2D map of Smrr(gamma, R)
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            s_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.set_title(
            r"MRR sensitivity, $S_{MRR}$, as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$S_{MRR}$ (RIU $^{-1}$)")
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_light"],
            label=r"max$\{S_{MRR}(\Gamma_{fluid}, R)\}$",
        )
        """
        ax.plot(
            [np.log10(self.max_S_radius), np.log10(self.max_S_radius)],
            [gamma_2D_map[-1], R_max_Smrr_gamma],
            color=self.models.parameters["map2D_overlay_color_light"],
        )
        """
        ax.plot(
            [r_2d_map_indices[0], np.log10(self.max_s_radius)],
            [r_max_s_mrr_gamma, r_max_s_mrr_gamma],
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle=LINE_STYLES["loosely dashdotted"],
            label=rf"max{{max{{$S_{{MRR}}$}}}} = {self.max_s:.0f} RIU$^{{-1}}$"
            + f" @ R = {self.max_s_radius:.0f} μm"
            + rf", $\Gamma$ = {r_max_s_mrr_gamma:.0f}$\%$",
        )
        ax.plot(
            np.log10(self.r_e),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="--",
            label=r"Re$(\Gamma_{fluid})$",
        )
        ax.plot(
            np.log10(self.r_w),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="-.",
            label=r"Rw$(\Gamma_{fluid})$",
        )
        """
        for line in self.models.parameters["map_line_profiles"] or []:
            ax.plot(
                [
                    np.log10(self.models.plotting_extrema["r_plot_min"]),
                    np.log10(self.models.plotting_extrema["r_plot_max"]),
                ],
                [line, line],
                color=self.models.parameters["map2D_overlay_color_light"],
                linestyle=LINE_STYLES["loosely dotted"],
            )
        """
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_S_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of Snr(gamma, R)
        s_nr_2d_map = np.asarray(
            [
                [self._calc_s_nr(r=10**log10_R, u=u) for log10_R in r_2d_map_indices]
                for u in u_2d_map_indices
            ]
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            s_nr_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_dark"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.set_title(
            r"MRR $S_{NR}$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$S_{NR}$ (RIU$^{-1}$)")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_Snr_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of Se(gamma, R)
        s_e_2d_map = np.asarray(
            [
                [self._calc_s_e(r=10**log10_R, u=u) for log10_R in r_2d_map_indices]
                for u in u_2d_map_indices
            ]
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            s_e_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_dark"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.set_title(
            r"MRR $S_e$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$S_e$")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_Se_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of Se*a(gamma, R)
        s_e_times_a_2d_map = np.asarray(
            [
                [
                    self._calc_s_e(r=10**log10_R, u=u)
                    * np.sqrt(self._calc_wg_a2(r=10**log10_R, u=u))
                    for log10_R in r_2d_map_indices
                ]
                for u in u_2d_map_indices
            ]
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            s_e_times_a_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_dark"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.set_title(
            r"MRR $S_e \times a$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f}μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$S_e \times a$")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_Se_x_a_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of a2(gamma, R)
        wg_a2_map = np.asarray(
            [
                [self._calc_wg_a2(r=10**log10_R, u=u) for log10_R in r_2d_map_indices]
                for u in u_2d_map_indices
            ]
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            wg_a2_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_light"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.plot(
            np.log10(self.r_e),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="--",
            label=r"Re$(\Gamma_{fluid})$",
        )
        ax.plot(
            np.log10(self.r_w),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_light"],
            linestyle="-.",
            label=r"Rw$(\Gamma_{fluid})$",
        )
        ax.set_title(
            r"MRR $a^2$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$a^2$")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_a2_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of alpha*L(gamma, R)
        db_per_cm_to_per_cm: float = 1.0 / 4.34
        αl_2d_map = (
            np.asarray(
                [
                    [
                        self._calc_α_l(r=10**log10_R, u=u)
                        for log10_R in r_2d_map_indices
                    ]
                    for u in u_2d_map_indices
                ]
            )
            / db_per_cm_to_per_cm
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            αl_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_dark"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.plot(
            np.log10(self.r_e),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_dark"],
            linestyle="--",
            label=r"Re$(\Gamma_{fluid})$",
        )
        ax.plot(
            np.log10(self.r_w),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_dark"],
            linestyle="-.",
            label=r"Rw$(\Gamma_{fluid})$",
        )
        """
        for line in self.models.parameters["map_line_profiles"] or []:
            ax.plot(
                [
                    np.log10(self.models.plotting_extrema["r_plot_min"]),
                    np.log10(self.models.plotting_extrema["r_plot_max"]),
                ],
                [line, line],
                color=self.models.parameters["map2D_overlay_color_dark"],
                linestyle=LINE_STYLES["loosely dotted"],
            )
        """
        ax.set_title(
            r"MRR $\alpha L$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"$\alpha L$ (dB)")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_alphaL_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # 2D map of 1/alpha(gamma, R)
        db_per_cm_to_per_cm: float = 1.0 / 4.34
        α_inv_2d_map = (
            np.asarray(
                [
                    [
                        self._α_prop(u=u) + self.models.α_bend(r=10**log10_R, u=u)
                        for log10_R in r_2d_map_indices
                    ]
                    for u in u_2d_map_indices
                ]
            )
        )
        fig, ax = plt.subplots()
        cm = ax.pcolormesh(
            r_2d_map_indices,
            gamma_2d_map,
            α_inv_2d_map,
            cmap=self.models.parameters["map2D_colormap"],
        )
        ax.plot(
            np.log10(self.models.r),
            self.gamma,
            color=self.models.parameters["map2D_overlay_color_dark"],
            label=r"max$\{S_{MRR}\}$",
        )
        ax.plot(
            np.log10(self.r_e),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_dark"],
            linestyle="--",
            label=r"Re$(\Gamma_{fluid})$",
        )
        ax.plot(
            np.log10(self.r_w),
            self.gamma_resampled * 100,
            color=self.models.parameters["map2D_overlay_color_dark"],
            linestyle="-.",
            label=r"Rw$(\Gamma_{fluid})$",
        )
        """
        for line in self.models.parameters["map_line_profiles"] or []:
            ax.plot(
                [
                    np.log10(self.models.plotting_extrema["r_plot_min"]),
                    np.log10(self.models.plotting_extrema["r_plot_max"]),
                ],
                [line, line],
                color=self.models.parameters["map2D_overlay_color_dark"],
                linestyle=LINE_STYLES["loosely dotted"],
            )
        """
        ax.set_title(
            r"MRR 1/$\alpha$ as a function of $\Gamma_{fluid}$ and $R$"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )
        ax.set_xlabel("log(R) (μm)")
        ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
        fig.colorbar(cm, label=r"1/$\alpha$ ($\mu$m)")
        ax.set_xlim(
            left=np.log10(self.models.plotting_extrema["r_plot_min"]),
            right=np.log10(self.models.plotting_extrema["r_plot_max"]),
        )
        ax.set_ylim(bottom=gamma_2d_map[-1], top=gamma_2d_map[0])
        ax.legend(loc="lower right")
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_2DMAP_alpha_inv_VS_GAMMA_and_R.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # Save 2D map data as a function of gamma and R to output Excel file
        if self.models.parameters["write_excel_files"]:
            # In addition to alpha*L, calculate 2D maps of alpha_prop*L and alpha_bend*L
            α_prop_l_2d_map = (
                np.asarray(
                    [
                        [
                            self._calc_α_prop_l(r=10**log10_R, u=u)
                            for log10_R in r_2d_map_indices
                        ]
                        for u in u_2d_map_indices
                    ]
                )
                / db_per_cm_to_per_cm
            )
            α_bend_l_2d_map = (
                np.asarray(
                    [
                        [
                            self._calc_α_bend_l(r=10**log10_R, u=u)
                            for log10_R in r_2d_map_indices
                        ]
                        for u in u_2d_map_indices
                    ]
                )
                / db_per_cm_to_per_cm
            )

            # Write all 2D maps to single Excel file
            self._write_image_data_to_excel(
                filename=str(
                    self.models.filename_path.parent
                    / f"{self.models.filename_path.stem}_MRR_2DMAPS_VS_GAMMA_and_R.xlsx"
                ),
                x_array=10**r_2d_map_indices,
                x_label="R (um)",
                y_array=gamma_2d_map,
                y_label="gamma (%)",
                z_array=[
                    s_2d_map,
                    s_nr_2d_map,
                    s_e_2d_map,
                    αl_2d_map,
                    α_inv_2d_map,
                    α_bend_l_2d_map,
                    α_prop_l_2d_map,
                ],
                z_labels=[
                    "S (RIU-1)",
                    "Snr (RIU-1)",
                    "Se",
                    "alpha x L (dB)",
                    r"alpha-1 (um)",
                    "alpha_bend x L (dB)",
                    "alpha_prop x L (dB)",
                ],
            )
            self.logger(f"Wrote '{filename.with_suffix('.xlsx')}'.")

    def _plot_mrr_sensing_parameters_at_optimum(self):
        """

        Returns:

        """

        # max{S}, S_NR, Se, a, u, gamma, Finesse
        fig, axs = plt.subplots(7)
        fig.suptitle(
            "MRR - Sensing parameters\n"
            + f"{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
            + f", Lc = {self.lc:.1f} μm\n"
            + rf"max{{max{{$S$}}}} = {self.max_s:.0f} (RIU$^{{-1}}$)"
            + rf" @ $R$ = {self.max_s_radius:.0f} μm"
        )

        # max{S}
        axs_index: int = 0
        axs[axs_index].set_ylabel(r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)")
        axs[axs_index].loglog(self.models.r, self.s)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [100, self.models.plotting_extrema["S_plot_max"]],
            "r--",
        )
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(100, self.models.plotting_extrema["S_plot_max"])
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # S_NR @ max{S}
        axs_index += 1
        axs[axs_index].loglog(self.models.r, self.s_nr)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [10, self.models.plotting_extrema["S_plot_max"]],
            "r--",
        )
        axs[axs_index].set_ylabel(r"S$_{NR}$" + "\n" + r"(RIU $^{-1}$)")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(10, self.models.plotting_extrema["S_plot_max"])
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # Se @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.s_e * np.sqrt(self.wg_a2))
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [0, self.plotting_extrema["Se_plot_max"]],
            "r--",
        )
        axs[axs_index].set_ylabel(r"S$_e \times a$")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(0, self.plotting_extrema["Se_plot_max"])
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # u (h or w) @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.u)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [
                self.models.plotting_extrema["u_plot_min"],
                self.models.plotting_extrema["u_plot_max"],
            ],
            "r--",
        )
        axs[axs_index].set_ylabel(f"{self.models.core_u_name} (μm)")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(
            self.models.plotting_extrema["u_plot_min"],
            self.models.plotting_extrema["u_plot_max"],
        )
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # Gamma_fluid @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.gamma)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [
                self.models.plotting_extrema["gamma_plot_min"],
                self.models.plotting_extrema["gamma_plot_max"],
            ],
            "r--",
        )
        axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(
            self.models.plotting_extrema["gamma_plot_min"],
            self.models.plotting_extrema["gamma_plot_max"],
        )
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # a2 @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.wg_a2)
        axs[axs_index].plot([self.max_s_radius, self.max_s_radius], [0, 1], "r--")
        axs[axs_index].set_ylabel(r"$a^2$")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(0, 1)
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # alpha_wg @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(
            self.models.r,
            np.asarray([self.models.α_wg_of_u(u) for u in self.u])
            * constants.PER_UM_TO_DB_PER_CM,
        )
        axs[axs_index].set_ylabel(r"α$_{wg}$")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(
            np.floor(self.models.α_wg_model["min"] * constants.PER_UM_TO_DB_PER_CM),
            np.ceil(self.models.α_wg_model["max"] * constants.PER_UM_TO_DB_PER_CM),
        )

        axs[axs_index].set_xlabel("Ring radius (μm)")
        filename: Path = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_sens_parms.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

    def _plot_mrr_ring_parameters_at_optimum(self):
        """

        Returns:

        """

        # max{S}, Q, Finesse, FWHM, FSR, contrast
        fig, axs = plt.subplots(6)
        fig.suptitle(
            "MRR - Ring parameters"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm\n"
            + rf"max{{max{{$S$}}}} = {self.max_s:.0f} (RIU$^{{-1}}$)"
            + rf" @ $R$ = {self.max_s_radius:.0f} μm"
        )
        # max{S}
        axs_index = 0
        axs[axs_index].set_ylabel(r"max$\{S\}$")
        axs[axs_index].loglog(self.models.r, self.s)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [100, self.models.plotting_extrema["S_plot_max"]],
            "r--",
        )
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylim(100, self.models.plotting_extrema["S_plot_max"])
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # Contrast, tau & a @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.tau, color="blue", label="τ")
        axs[axs_index].semilogx(
            self.models.r, np.sqrt(self.wg_a2), color="green", label="a"
        )
        axs[axs_index].semilogx(
            self.models.r, self.contrast, color="red", label="contrast"
        )
        axs[axs_index].plot([self.max_s_radius, self.max_s_radius], [0, 1], "r--")
        axs[axs_index].set_ylim(0, 1)
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylabel(r"Contrast, $a$, $\tau$")
        axs[axs_index].axes.get_xaxis().set_ticklabels([])
        axs[axs_index].legend(loc="upper right")

        # ER @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.er, label="Q")
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [0, np.amax(self.er)],
            "r--",
        )
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylabel("Extinction\nratio\n(dB)")
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # Q @ max{S}
        axs_index += 1
        axs[axs_index].loglog(self.models.r, self.q, label="Q")
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [0, np.amax(self.q)],
            "r--",
        )
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylabel("Q")
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # (Finesse/2pi) / Se*a @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(
            self.models.r,
            self.finesse / (2 * np.pi) / (self.s_e * np.sqrt(self.wg_a2)),
        )
        axs[axs_index].set_ylim(0, 2.5)
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylabel(r"$\frac{Finesse/2\pi}{S_e\times a}$")
        axs[axs_index].axes.get_xaxis().set_ticklabels([])

        # FWHM, FSR, Finesse/2pi @ max{S}
        axs_index += 1
        axs[axs_index].loglog(self.models.r, self.fwhm * 1e6, "b", label="FWHM")
        axs[axs_index].loglog(self.models.r, self.fsr * 1e6, "g", label="FSR")
        axs[axs_index].set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        axs[axs_index].set_ylabel("FWHM and FSR\n(pm)")
        axs[axs_index].set_xlabel("Ring radius (μm)")
        ax_right = axs[axs_index].twinx()
        ax_right.semilogx(
            self.models.r, self.finesse / (2 * np.pi), "k--", label="Finesse/2π"
        )
        ax_right.set_ylabel("Finesse/2π")
        ax_right.grid(visible=False)
        ax_lines = (
            axs[axs_index].get_legend_handles_labels()[0]
            + ax_right.get_legend_handles_labels()[0]
        )
        ax_labels = (
            axs[axs_index].get_legend_handles_labels()[1]
            + ax_right.get_legend_handles_labels()[1]
        )
        axs[axs_index].legend(ax_lines, ax_labels, loc="upper right")
        axs[axs_index].patch.set_visible(False)
        ax_right.patch.set_visible(True)
        axs[axs_index].set_zorder(ax_right.get_zorder() + 1)

        # Write figure to file
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_MRR_ring_parms.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

    def plot_optimization_results(self):
        """

        Returns:

        """
        self._plot_mrr_sensing_parameters_at_optimum()
        self._plot_mrr_ring_parameters_at_optimum()
        if self.models.parameters["write_2D_maps"]:
            self._plot_mrr_result_2d_maps()

    def plot_combined_sensor_optimization_results(self, linear: Linear, spiral: Spiral):
        """

        Args:
            linear ():
            spiral ():

        Returns:

        """

        # Calculate minimum sensitivity required to detect the minimum resolvable
        # change in ni for a given transmission measurement SNR
        s_min: float = (
            10 ** (-self.models.parameters["T_SNR"] / 10)
            / self.models.parameters["min_delta_ni"]
        )

        # Create plot figure
        fig, ax = plt.subplots()
        ax.set_title(
            "Maximum sensitivity for MRR and linear sensors"
            if self.models.parameters["no_spiral"]
            else "Maximum sensitivity for MRR, linear, and spiral sensors"
            + f"\n{self.models.pol}"
            + f", λ = {self.models.lambda_res:.3f} μm"
            + f", {self.models.core_v_name} = {self.models.core_v_value:.3f} μm"
        )

        # MRR
        ax.set_xlabel("Ring radius (μm)")
        ax.set_ylabel(r"Maximum sensitivity (RIU$^{-1}$)")
        ax.loglog(self.models.r, self.s, color="b", label="MRR")

        # Linear waveguide
        ax.loglog(
            self.models.r,
            linear.s,
            color="g",
            label=r"Linear waveguide ($L = 2R$)",
        )
        ax.loglog(
            [
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ],
            [s_min, s_min],
            "r--",
            label="".join(
                [
                    r"min$\{S\}$ to resolve $\Delta n_{i}$",
                    f" = {self.models.parameters['min_delta_ni']:.0E} "
                    + f"@ SNR = {self.models.parameters['T_SNR']:.0f} dB",
                ]
            ),
        )
        ax.set_xlim(
            self.models.plotting_extrema["r_plot_min"],
            self.models.plotting_extrema["r_plot_max"],
        )
        ax.set_ylim(100, 1000000)

        # Spiral and MRR/spiral sensitivity ratio, if required
        if not self.models.parameters["no_spiral"]:
            ax.loglog(
                self.models.r[spiral.s > 1],
                spiral.s[spiral.s > 1],
                color="k",
                label=f"Spiral (spacing = {spiral.spacing:.0f} μm"
                + f", min turns = {spiral.turns_min:.2f})",
            )
            ax_right = ax.twinx()
            ax_right.semilogx(
                self.models.r,
                self.s / spiral.s,
                "k--",
                label=r"max$\{S_{MRR}\}$ / max$\{S_{SPIRAL}\}$",
            )
            ax_right.set_ylabel(r"max$\{S_{MRR}\}$ / max$\{S_{SPIRAL}\}$")
            ax_right.set_ylim(0, 30)
            ax_right.grid(visible=False)
            ax_lines = (
                ax.get_legend_handles_labels()[0]
                + ax_right.get_legend_handles_labels()[0]
            )
            ax_labels = (
                ax.get_legend_handles_labels()[1]
                + ax_right.get_legend_handles_labels()[1]
            )
            ax.legend(ax_lines, ax_labels, loc="lower right")
            ax.patch.set_visible(False)
            ax_right.patch.set_visible(True)
            ax.set_zorder(ax_right.get_zorder() + 1)
        else:
            ax.legend(loc="lower right")

        # Save figure
        filename = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_ALL_RESULTS.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

    def _calc_α_bend_a_and_b(self, gamma: float) -> tuple[float, float]:
        """
        Fit A & B model parameters to alpha_bend(R) = A*exp(-B*R) @ gamma
        """

        u: float = self.models.u_of_gamma(gamma=gamma)
        r: np.ndarray = np.arange(
            self.models.r_α_bend_min_interp(u),
            self.models.r_α_bend_max_interp(u),
            (self.models.r_α_bend_max_interp(u) - self.models.r_α_bend_min_interp(u))
            / 10,
        )
        α_bend: np.ndarray = np.asarray([self.models.α_bend(r=r, u=u) for r in r])
        minus_b, ln_a = np.linalg.lstsq(
            a=np.vstack([r, np.ones(len(r))]).T, b=np.log(α_bend), rcond=None
        )[0]

        return np.exp(ln_a), -minus_b

    def _objfun_r_w(
        self, r: float, u: float, α_bend_a: float, α_bend_b: float
    ) -> float:
        """
        Calculate the residual squared with the current solution for Rw,
        using equation (15) in the paper.
        """

        α_bend: float = α_bend_a * np.exp(-α_bend_b * r)
        residual: float = 1 - r * (2 * np.pi) * (
            self._α_prop(u=u) + (1 - α_bend_b * r) * α_bend
        )

        return residual**2

    def _calc_r_e_and_r_w(self, gamma: float) -> tuple[float, float, float, float]:
        """
        Calculate Re(gamma) and Rw(gamma)
        """

        # u corresponding to gamma
        u: float = self.models.u_of_gamma(gamma=gamma)

        # alpha_bend(R) = A*exp(-BR) model parameters @gamma
        α_bend_a, α_bend_b = self._calc_α_bend_a_and_b(gamma=gamma)

        # Re
        w: float = lambertw(-e * self._α_prop(u=u) / α_bend_a, k=-1).real
        r_e: float = (1 / α_bend_b) * (1 - w)

        # Rw
        optimization_result = optimize.minimize(
            fun=self._objfun_r_w,
            x0=np.asarray(r_e),
            args=(u, α_bend_a, α_bend_b),
            method="SLSQP",
        )
        r_w: float = optimization_result["x"][0]

        return r_e, r_w, α_bend_a, α_bend_b

    #
    # Optimization
    #

    def _α_prop(self, u: float) -> float:
        """
        α_prop = α_wg + gamma_fluid*α_fluid
        """

        return self.models.α_wg_of_u(u=u) + (
            self.models.gamma_of_u(u) * self.models.α_fluid
        )

    def _calc_α_prop_l(self, r: float, u: float) -> float:
        """
        Propagation loss component of total round-trip losses : α_prop*L
        """

        return self._α_prop(u=u) * ((2 * np.pi * r) + (2 * self.lc))

    def _calc_α_bend_l(self, r: float, u: float) -> float:
        """
        Bending loss component of total round-trip losses: α_bend*L
        """
        return self.models.α_bend(r=r, u=u) * (2 * np.pi * r)

    def _calc_α_l(self, r: float, u: float) -> float:
        """
        Total ring round-trip loss factor: αL = (α_prop + α_bend)*L
        """

        return self._calc_α_prop_l(r=r, u=u) + self._calc_α_bend_l(r=r, u=u)

    def _calc_wg_a2(self, r: float, u: float) -> float:
        """
        Ring round trio losses: a2 = e**(-α*L)
        """

        return np.e ** -self._calc_α_l(r=r, u=u)

    def _calc_s_nr(self, r: float, u: float) -> float:
        """
        Calculate Snr (see paper)
        """
        return (
            (4 * np.pi / self.models.lambda_res)
            * ((2 * np.pi * r) + (2 * self.lc))
            * self.models.gamma_of_u(u)
            * self._calc_wg_a2(r=r, u=u)
        )

    def _calc_s_e(self, r: float, u: float) -> float:
        """
        Calculate Se (see paper)
        """

        return (
            2
            / (3 * np.sqrt(3))
            / (np.sqrt(self._calc_wg_a2(r=r, u=u)) * (1 - self._calc_wg_a2(r=r, u=u)))
        )

    def _calc_sensitivity(self, r: float, u: float) -> float:
        """
        Calculate sensitivity at radius r for a given core dimension u
        """

        s: float = self._calc_s_nr(r=r, u=u) * self._calc_s_e(r=r, u=u)
        assert s >= 0, "S should not be negative!"

        return s

    def _obj_fun(self, u: float, *args) -> float:
        """
        Objective function for the non-linear minimization in find_max_sensitivity()
        """

        # Fetch additional parameters
        r = args[0]

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.u_domain_max)
        u = max(u, self.models.u_domain_min)

        # Calculate sensitivity at current solution vector S(r, h)
        s: float = self._calc_sensitivity(r=r, u=u)

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
        if u_min != u_max:
            if self.models.parameters["optimization_local"]:
                optimization_result = optimize.minimize(
                    fun=self._obj_fun,
                    x0=np.asarray([u0]),
                    bounds=((u_min, u_max),),
                    args=(r,),
                    method=self.models.parameters["optimization_method"],
                    tol=1e-9,
                )
            else:
                optimization_result = optimize.shgo(
                    func=self._obj_fun,
                    bounds=[(u_min, u_max)],
                    args=(r,),
                    iters=3,
                    options={"minimize_every_iter": True},
                )
            u_max_s = optimization_result.x[0]
        else:
            u_max_s = u0

        # Update previous solution
        self.previous_solution = u_max_s

        # Calculate sensitivity and other parameters at the solution
        s = self._calc_sensitivity(r=r, u=u_max_s)

        # Calculate other useful MRR parameters at the solution
        wg_a2: float = self._calc_wg_a2(r=r, u=u_max_s)
        a: float = np.sqrt(wg_a2)
        tau: float = (np.sqrt(3) * wg_a2 - np.sqrt(3) - 2 * a) / (wg_a2 - 3)
        t_max: float = ((tau + a) / (1 + tau * a)) ** 2
        t_min: float = ((tau - a) / (1 - tau * a)) ** 2
        gamma: float = self.models.gamma_of_u(u_max_s) * 100
        n_eff: float = self.models.n_eff_of_u(u_max_s)
        finesse: float = np.pi * (np.sqrt(tau * a)) / (1 - tau * a)
        q: float = (n_eff * (2 * np.pi * r) / self.models.lambda_res) * finesse
        fwhm: float = self.models.lambda_res / q
        fsr: float = finesse * fwhm
        contrast: float = t_max - t_min
        er: float = 10 * np.log10(t_max / t_min)
        s_nr: float = self._calc_s_nr(r=r, u=u_max_s)
        s_e: float = self._calc_s_e(r=r, u=u_max_s)
        α_bend: float = self.models.α_bend(r=r, u=u_max_s)
        α_wg: float = self.models.α_wg_of_u(u=u_max_s)
        α_prop: float = self._α_prop(u=u_max_s)
        α: float = α_prop + α_bend
        αl: float = self._calc_α_l(r=r, u=u_max_s)

        # Return results to calling program
        return (
            s,
            u_max_s,
            gamma,
            s_nr,
            s_e,
            α_bend,
            α_wg,
            α_prop,
            α,
            αl,
            wg_a2,
            tau,
            t_max,
            t_min,
            er,
            contrast,
            n_eff,
            q,
            finesse,
            fwhm,
            fsr,
        )

    def analyze(self):
        """
        Analyse the MRR sensor performance for all radii in the R domain

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
            self.s_nr,
            self.s_e,
            self.α_bend,
            self.α_wg,
            self.α_prop,
            self.α,
            self.αl,
            self.wg_a2,
            self.tau,
            self.t_max,
            self.t_min,
            self.er,
            self.contrast,
            self.n_eff,
            self.q,
            self.finesse,
            self.fwhm,
            self.fsr,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_s = np.amax(self.s)
        self.max_s_radius = self.models.r[np.argmax(self.s)]

        # Calculate Re(gamma) and Rw(gamma)
        gamma_min: float = list(self.models.modes_data.values())[-1]["gamma"]
        gamma_max: float = list(self.models.modes_data.values())[0]["gamma"]
        self.gamma_resampled = np.linspace(gamma_min, gamma_max, 500)
        self.u_resampled = [self.models.u_of_gamma(g) for g in self.gamma_resampled]
        self.r_e, self.r_w, self.α_bend_a, self.α_bend_b = zip(
            *[self._calc_r_e_and_r_w(gamma=gamma) for gamma in self.gamma_resampled]
        )

        # Calculate extrema for plotting of results
        self._calculate_plotting_extrema()

        # Console message
        self.logger("MRR sensor analysis done.")
