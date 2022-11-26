"""spiral.py

Spiral sensor class

"""
__all__ = ["Spiral"]


import io
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, TiffImagePlugin
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from pathlib import Path
from rich import print
from scipy import optimize, integrate

from .constants import CONSTANTS
from .models import Models


class Spiral:
    """
    Spiral class

    Archimedes spiral: "r(theta) = a + b*theta", where:
        - "a": outer spiral starting point offset from the origin along the "x" axis
        - "2*pi*b": distance between the line pairs in the spiral.
        - theta: angle from the "x" axis
        - the outer and inner spiral waveguides are joined by an S-bend at the center

        In this case, the center-to-center spacing between the rings in the spiral,
        i.e. the "linewidth", is 2 x (waveguide core width + inter-waveguide spacing),
        thus "2*pi*b" = 2 x (waveguide core width + inter-waveguide spacing).

        - Constraints by hypothesis (MAGIC NUMBERS):
            1) min{a} = 1.5 x linewidth to allow for space for the S-bend joint at the
               center of the spiral.
            2) The spiral must have at least one complete turn, else it's just a
               "curved waveguides", therefore the minimum spiral radius = a + linewidth,
               but this parameter us user-selectable in the .toml file.

    All lengths are in units of um

    Exposed methods:
        - analyze()
        - plot_optimization_results()

    """

    def __init__(
        self,
        models: Models,
        logger: Callable = print,
    ):
        """

        Args:
            models (Models): Models object that contains the problem data
            logger (Callable): console logger
        """

        # Load class instance parameter values
        self.models: Models = models
        self.logger: Callable = logger

        # Calculate spiral parameters
        self.line_width_min: float = 2 * (
            (
                self.models.parms.wg.v_coord_value
                if self.models.parms.wg.v_coord_name == "w"
                else self.models.parms.limits.u_min
            )
            + self.models.parms.spiral.spacing
        )
        self.a_spiral_min: float = 1.5 * self.line_width_min
        self.b_spiral_min: float = self.line_width_min / (2 * np.pi)

        # Declare class instance internal variables
        self.previous_solution: np.ndarray = np.asarray([-1, -1])

        # Declare class instance result variables and arrays
        self.s: np.ndarray = np.ndarray([])
        self.u: np.ndarray = np.ndarray([])
        self.n_turns: np.ndarray = np.ndarray([])
        self.outer_spiral_r_min: np.ndarray = np.ndarray([])
        self.l: np.ndarray = np.ndarray([])
        self.gamma: np.ndarray = np.ndarray([])
        self.wg_a2: np.ndarray = np.ndarray([])
        self.max_s: float = 0
        self.max_s_radius: float = 0
        self.results: list = []

    #
    # Plotting
    #

    def _calc_spiral_parameters(self, w: float) -> tuple[float, float, float]:
        """
        Calculate spiral parameters line_width, a, b as a function of waveguide width

        Args:
            w (float): waveguide width

        Returns: spiral line_width (float), a (float), b (float)

        """

        line_width: float = 2 * (w + self.models.parms.spiral.spacing)
        b: float = line_width / (2 * np.pi)
        a: float = 1.5 * line_width

        return line_width, a, b

    @staticmethod
    def _plot_arc(
        thetas: np.ndarray,
        r: np.ndarray | float,
        x0: float = 0,
        y0: float = 0,
        theta0: float = 0,
        spiral_x: np.ndarray = np.asarray([]),
        spiral_y: np.ndarray = np.asarray([]),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate parametric arc theta(n) & r(n) centered at [x0,y0] and rotated by
        theta0, append arc x[n]/y[n] arc sample coordinates to x/y spiral arrays

        Args:
            thetas (np.ndarray): array of angles spanning the arc
            r (np.ndarray): array of radii spanning the arc
            x0 (float): spiral center x coordinate
            y0 (float): spiral center y coordinate
            theta0 (float): spiral rotation about the center
            spiral_x (np.ndarray): array of current spiral point x coordinates to update
            spiral_y (np.ndarray): array of current spiral point y coordinates to update

        Returns: updated spiral_x (np.ndarray), spiral_y (np.ndarray)

        """

        # Build array of radii by duplication if r is a float
        rs: np.ndarray = r * np.ones(len(thetas)) if isinstance(r, float) else r

        # Rotate the arc by "theta0" so that consecutive spirals are aligned
        x_rot: np.ndarray = x0 + rs * np.cos(thetas)
        y_rot: np.ndarray = y0 + rs * np.sin(thetas)
        x: np.ndarray = x_rot * np.cos(theta0) - y_rot * np.sin(theta0)
        y: np.ndarray = y_rot * np.cos(theta0) + x_rot * np.sin(theta0)

        # return length and updated x,y arrays
        return np.append(spiral_x, x), np.append(spiral_y, y)

    @staticmethod
    def _append_image_to_seq(images: list, fig: plt.Figure) -> None:
        """
        Append a matplotlib Figure to a list of PIL Image objects (from figs2tiff.py)

        Args:
            images (list): List of PIL Image objects to which to append the figure
            fig (matplotlib.pyplot.Figure): matplotlib Figure object to append to list

        Note: An empty list must be declared in the calling program prior to
              the first call to the function.

        Returns: None

        """

        with io.BytesIO() as buffer:
            # Convert Figure to PIL Image using an intermediate memory buffer
            fig.savefig(buffer, format="tif")
            img: Image = Image.open(buffer)

            # Initialize Image encoderinfo, encoderconfig, and mode (RGB)
            # properties as required for a multi-image TIFF file
            img = img.convert("RGB")
            img.encoderinfo = {"tiffinfo": TiffImagePlugin.ImageFileDirectory()}
            img.encoderconfig = ()

            # Append Image object to the List
            images.append(img)

        # Explicit None return
        return None

    def _write_spiral_sequence_to_file(self) -> None:
        """
        Write sequence of consecutive spirals to file, if n turns > self.n_turns_min

        Returns: None

        """

        # Calculate spiral sequence looping indices (min, max, index)
        biggest_spiral_index: int = int(np.argmax(self.n_turns))
        index_min: int = int(
            np.argmax(
                self.n_turns[:biggest_spiral_index] > self.models.parms.spiral.turns_min
            )
        )
        index_max: int = (
            int(np.argmax(self.n_turns[biggest_spiral_index:] < 1))
            + biggest_spiral_index
        )
        indices: range = range(index_min, index_max)

        # Check for adequate range of spirals in sequence, else exit with warning
        if len(indices) <= 2:
            self.logger(
                "WARNING! Insufficient range in number of spiral turns "
                + f"(array indices: [{indices[0]}, {indices[-1]}]), max number "
                + f"of turns = {self.n_turns[biggest_spiral_index]:.1f}, "
                + "no sequence written!"
            )
            return

        # Loop to write generate the spiral images in the sequence
        fig, _ = plt.subplots()
        images: list = []
        for index in indices:
            fig, *_ = self._draw_spiral(
                r_outer=self.models.r[index],
                h=self.u[index]
                if self.models.parms.wg.v_coord_name == "w"
                else self.models.parms.wg.v_coord_value,
                w=self.models.parms.wg.v_coord_value
                if self.models.parms.wg.v_coord_name == "w"
                else self.u[index],
                n_turns=self.n_turns[index],
                r_window=(self.models.r[index_max] // 25 + 1) * 25,
                figure=fig,
            )
            self._append_image_to_seq(images=images, fig=fig)
        plt.close(fig)

        # Save sequence to tiff multi-image file
        filename: Path = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_SPIRAL_sequence.tif"
        )
        images[0].save(
            str(filename),
            save_all=True,
            append_images=images[1:],
            duration=40,
        )
        self.logger(f"Wrote '{filename}'.")

        # Explicit None return
        return None

    def _write_spiral_waveguide_coordinates_to_excel_file(
        self, spiral_waveguide_coordinates: dict
    ) -> None:
        """
         Write the spiral inner and outer waveguide x/y coordinates to an Excel file

        Args:
            spiral_waveguide_coordinates (dict): dictionary fo spiral inner
            and outer waveguide edge point coordinates

        Returns: None

        """

        filename: Path = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_SPIRAL_SCHEMATIC.xlsx"
        )
        wb: Workbook = Workbook()
        outer_spiral_sheet: Worksheet = wb["Sheet"]
        outer_spiral_sheet.title = "Outer waveguide"
        outer_spiral_sheet.append(
            [
                "Outer edge x (um)",
                "Outer edge y (um)",
                "Inner edge x (um)",
                "Inner edge y (um)",
            ]
        )
        for row in zip(
            spiral_waveguide_coordinates["outer_spiral_x_out"],
            spiral_waveguide_coordinates["outer_spiral_y_out"],
            spiral_waveguide_coordinates["outer_spiral_x_in"],
            spiral_waveguide_coordinates["outer_spiral_y_in"],
        ):
            outer_spiral_sheet.append(row)
        inner_spiral_sheet: Worksheet = wb.create_sheet("Inner waveguide")
        inner_spiral_sheet.append(
            [
                "Outer edge x (um)",
                "Outer edge y (um)",
                "Inner edge x (um)",
                "Inner edge y (um)",
            ]
        )
        for row in zip(
            spiral_waveguide_coordinates["inner_spiral_x_out"],
            spiral_waveguide_coordinates["inner_spiral_y_out"],
            spiral_waveguide_coordinates["inner_spiral_x_in"],
            spiral_waveguide_coordinates["inner_spiral_y_in"],
        ):
            inner_spiral_sheet.append(row)
        wb.save(filename=filename)
        self.logger(f"Wrote '{filename}'.")

        # Explicit None return
        return None

    def _plot_spiral_results_at_optimum(self) -> None:
        """
        Plot the optimization results for the spiral at optimum (maximum sensitivity
        as a function of spiral outer radius)

        Returns: None

        """

        # Plot max{S}, u, gamma, n turns mas, L
        fig, axs = plt.subplots(7)
        fig.suptitle(
            "Archimedes spiral\n"
            + f"{self.models.parms.wg.polarization}"
            + f", λ = {self.models.parms.wg.lambda_resonance:.3f} μm"
            + f", {self.models.parms.wg.v_coord_name} = "
            f"{self.models.parms.wg.v_coord_value:.3f} μm"
            + f", spacing = {self.models.parms.spiral.spacing:.0f} μm"
            + f", min turns = {self.models.parms.spiral.turns_min:.2}\n"
            + rf"max{{max{{$S$}}}} = {self.max_s:.0f} (RIU$^{{-1}}$)"
            + rf" @ $R$ = {self.max_s_radius:.0f} μm"
        )
        # max{S}
        axs_index: int = 0
        axs[axs_index].loglog(self.models.r, self.s)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [100, self.models.plotting_extrema["S_plot_max"]],
            "--",
        )
        axs[axs_index].set(
            ylabel=r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(100, self.models.plotting_extrema["S_plot_max"]),
            xticklabels=([]),
        )

        # u @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.u)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [
                self.models.plotting_extrema["u_plot_min"],
                self.models.plotting_extrema["u_plot_max"],
            ],
            "--",
        )
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

        # gamma_fluid @ max{S}
        axs_index += 1
        axs[axs_index].semilogx(self.models.r, self.gamma)
        axs[axs_index].plot([self.max_s_radius, self.max_s_radius], [0, 100], "--")
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
        axs[axs_index].plot([self.max_s_radius, self.max_s_radius], [0, 1], "--")
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
            xticklabels=([]),
        )

        # n turns @ max{S}
        axs_index += 1
        n_turns_plot_max: float = np.ceil(np.amax(self.n_turns) * 1.1 / 10) * 10 * 2
        axs[axs_index].semilogx(self.models.r, self.n_turns * 2)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [0, n_turns_plot_max],
            "--",
        )
        axs[axs_index].set(
            ylabel="n turns\n(inner+outer)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(0, n_turns_plot_max),
            xticklabels=([]),
        )

        # L @ max{S}
        axs_index += 1
        axs[axs_index].loglog(self.models.r, self.l)
        axs[axs_index].plot(
            [self.max_s_radius, self.max_s_radius],
            [100, self.models.plotting_extrema["S_plot_max"]],
            "--",
        )
        axs[axs_index].set(
            xlabel="Ring radius (μm)",
            ylabel="L (μm)",
            xlim=(
                self.models.plotting_extrema["r_plot_min"],
                self.models.plotting_extrema["r_plot_max"],
            ),
            ylim=(100, self.models.plotting_extrema["S_plot_max"]),
        )
        filename: Path = (
            self.models.filename_path.parent
            / f"{self.models.filename_path.stem}_SPIRAL.png"
        )
        fig.savefig(filename)
        self.logger(f"Wrote '{filename}'.")

        # Explicit None return
        return None

    def plot_optimization_results_and_spiral_shape(self) -> None:
        """
        Plot the optimization results and shape for the spiral at optimum (maximum
        sensitivity as a function of r)

        Returns: None

        """

        # Plot spiral optimization results: u, gamma, n turns, a2, L @max(S)
        self._plot_spiral_results_at_optimum()

        # Draw the spiral with the greatest number of turns found in the optimization
        if self.models.parms.io.draw_largest_spiral:
            largest_spiral_index: int = int(np.argmax(self.n_turns))
            (fig, spiral_waveguide_coordinates,) = self._draw_spiral(
                r_outer=self.models.r[largest_spiral_index],
                h=self.u[largest_spiral_index]
                if self.models.parms.wg.v_coord_name == "w"
                else self.models.parms.wg.v_coord_value,
                w=self.models.parms.wg.v_coord_value
                if self.models.parms.wg.v_coord_name == "w"
                else self.u[largest_spiral_index],
                n_turns=self.n_turns[largest_spiral_index],
                r_window=(self.models.r[largest_spiral_index] // 25 + 1) * 25,
            )
            filename = (
                self.models.filename_path.parent
                / f"{self.models.filename_path.stem}_SPIRAL_SCHEMATIC.png"
            )
            fig.savefig(fname=filename)
            self.logger(f"Wrote '{filename}'.")

            # Write spiral inner and outer waveguide x/y coordinates to an Excel file
            if self.models.parms.io.write_excel_files:
                self._write_spiral_waveguide_coordinates_to_excel_file(
                    spiral_waveguide_coordinates=spiral_waveguide_coordinates
                )

        # Write sequence of consecutive spirals with n turns > self.n_turns_min
        if self.models.parms.io.write_spiral_sequence_to_file:
            self._write_spiral_sequence_to_file()

        # Explicit None return
        return None

    def _draw_spiral(
        self,
        r_outer: float,
        h: float,
        w: float,
        n_turns: float,
        r_window: float,
        figure: plt.Figure = None,
    ) -> tuple[plt.Figure, dict,]:
        """

        Args:
            r_outer (float): spiral outer radius
            h (float): waveguide core width
            w (float): waveguide core height
            n_turns (float): number of turns in the spiral
            r_window (float): square plotting window width/height
            figure (plt.Figure): plotting figure, if supplied

        Returns:
            fig (plt.Figure): plotting window
            spiral_waveguide_coordinates (dict): dictionary of spiral waveguide
                                                 xy point coordinate arrays

        """

        # Archimedes spiral parameters
        line_width, a_spiral, b_spiral = self._calc_spiral_parameters(w=w)

        # Define new figure if required, else use axes passed as a function parameter
        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig = figure
            ax = fig.axes[0]
            ax.clear()

        # Outer spiral
        theta_max: float = (
            r_outer - (a_spiral + self.models.parms.spiral.spacing + w)
        ) / b_spiral
        theta_min: float = max(theta_max - (n_turns * 2 * np.pi), 0)
        thetas_spiral: np.ndarray = np.linspace(theta_max, theta_min, 1000)
        theta0: float = -theta_max + np.pi / 2
        r_outer_spiral_inner: np.ndarray = (
            a_spiral + self.models.parms.spiral.spacing
        ) + (b_spiral * thetas_spiral)

        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_spiral,
            r=r_outer_spiral_inner,
            theta0=theta0,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_spiral,
            r=r_outer_spiral_inner + w,
            theta0=theta0,
        )

        # Inner spiral
        r_inner_spiral_inner: np.ndarray = a_spiral + (b_spiral * thetas_spiral)
        inner_spiral_x_in, inner_spiral_y_in = self._plot_arc(
            thetas=thetas_spiral,
            r=r_inner_spiral_inner,
            theta0=theta0,
        )
        inner_spiral_x_out, inner_spiral_y_out = self._plot_arc(
            thetas=thetas_spiral,
            r=r_inner_spiral_inner + w,
            theta0=theta0,
        )

        # Joint: half circle between outer waveguide and S-bend
        thetas_half_circle: np.ndarray = np.linspace(theta_min, theta_min - np.pi, 100)
        r_joint_inner: np.ndarray = (a_spiral + self.models.parms.spiral.spacing) + (
            b_spiral * thetas_half_circle
        )
        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_half_circle,
            r=r_joint_inner,
            theta0=theta0,
            spiral_x=outer_spiral_x_in,
            spiral_y=outer_spiral_y_in,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_half_circle,
            r=r_joint_inner + w,
            theta0=theta0,
            spiral_x=outer_spiral_x_out,
            spiral_y=outer_spiral_y_out,
        )

        # Joint: half S-bend between outer waveguide half-circle and origin
        thetas_s_bend_outer: np.ndarray = np.linspace(
            thetas_half_circle[-1], thetas_half_circle[-1] - np.pi, 100
        )
        outer_spiral_x_in, outer_spiral_y_in = self._plot_arc(
            thetas=thetas_s_bend_outer,
            r=(r_joint_inner[-1] - w / 2) / 2,
            x0=-((r_joint_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=-((r_joint_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=outer_spiral_x_in,
            spiral_y=outer_spiral_y_in,
        )
        outer_spiral_x_out, outer_spiral_y_out = self._plot_arc(
            thetas=thetas_s_bend_outer,
            r=(r_joint_inner[-1] + 3 * w / 2) / 2,
            x0=-((r_joint_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=-((r_joint_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=outer_spiral_x_out,
            spiral_y=outer_spiral_y_out,
        )

        # Joint: half S-bend between inner waveguide and origin
        thetas_s_bend_inner: np.ndarray = np.linspace(
            thetas_spiral[-1], thetas_spiral[-1] - np.pi, 100
        )
        inner_spiral_x_in, inner_spiral_y_in = self._plot_arc(
            thetas=thetas_s_bend_inner,
            r=(r_inner_spiral_inner[-1] - w / 2) / 2,
            x0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=inner_spiral_x_in,
            spiral_y=inner_spiral_y_in,
        )
        inner_spiral_x_out, inner_spiral_y_out = self._plot_arc(
            thetas=thetas_s_bend_inner,
            r=(r_inner_spiral_inner[-1] + 3 * w / 2) / 2,
            x0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.cos(thetas_half_circle[0]),
            y0=((r_inner_spiral_inner[-1] + w / 2) / 2) * np.sin(thetas_half_circle[0]),
            theta0=theta0,
            spiral_x=inner_spiral_x_out,
            spiral_y=inner_spiral_y_out,
        )

        # Total spiral length (inner + outer spirals) calculated by finite differences
        # (sum of averaged outer/inner edges for both spirals)
        outer_spiral_length: float = (
            np.sum(
                np.sqrt(
                    np.diff(outer_spiral_x_out) ** 2 + np.diff(outer_spiral_y_out) ** 2
                )
            )
            + np.sum(
                np.sqrt(
                    np.diff(outer_spiral_x_in) ** 2 + np.diff(outer_spiral_y_in) ** 2
                )
            )
        ) / 2
        inner_spiral_length: float = (
            np.sum(
                np.sqrt(
                    np.diff(inner_spiral_x_out) ** 2 + np.diff(inner_spiral_y_out) ** 2
                )
            )
            + np.sum(
                np.sqrt(
                    np.diff(inner_spiral_x_in) ** 2 + np.diff(inner_spiral_y_in) ** 2
                )
            )
        ) / 2
        spiral_length: float = outer_spiral_length + inner_spiral_length

        # Plot info & formatting
        s, _, length, _ = self._calc_sensitivity(
            r=r_outer,
            u=h if self.models.parms.wg.v_coord_name == "w" else w,
            n_turns=n_turns,
        )
        ax.set(
            title=f"Archimedes spiral : {n_turns: .2f} turns, "
            f"w = {self.models.parms.wg.v_coord_value:.3f} μm, "
            f"spacing = {self.models.parms.spiral.spacing:.1f} μm, "
            f"h = {h:.3f} μm, "
            f"S = {s:.0f} RIU$^{{-1}}$"
            f"\nR = {r_outer:.1f} μm, "
            f"R$_{{min}}$ = {r_joint_inner[-1]:.1f} μm, "
            f"S-bend radius = {(r_joint_inner[-1] - w / 2) / 2:.1f} μm, "
            f"L = {length:.1f} ({spiral_length:.2f} by finite diffs) μm",
            aspect="equal",
            xlabel=r"$\mu$m",
            ylabel=r"$\mu$m",
            xlim=(-r_window, r_window),
            ylim=(-r_window, r_window),
        )
        ax.plot(outer_spiral_x_in, outer_spiral_y_in, color="red")
        ax.plot(outer_spiral_x_out, outer_spiral_y_out, color="red")
        ax.plot(inner_spiral_x_in, inner_spiral_y_in, color="blue")
        ax.plot(inner_spiral_x_out, inner_spiral_y_out, color="blue")

        # Build dictionary of spiral waveguide vertex coordinates
        spiral_waveguide_coordinates: dict = {
            "outer_spiral_x_out": outer_spiral_x_out,
            "outer_spiral_y_out": outer_spiral_y_out,
            "outer_spiral_x_in": outer_spiral_x_in,
            "outer_spiral_y_in": outer_spiral_y_in,
            "inner_spiral_x_out": inner_spiral_x_out,
            "inner_spiral_y_out": inner_spiral_y_out,
            "inner_spiral_x_in": inner_spiral_x_in,
            "inner_spiral_y_in": inner_spiral_y_in,
        }

        # Return the figure and the dictionary of spiral point coordinates
        return (
            fig,
            spiral_waveguide_coordinates,
        )

    #
    # Optimization
    #

    def _α_prop(self, u: float) -> float:
        """
        alpha_prop(u) = alpha_wg(u) + gamma_fluid(u)*alpha_fluid

        Args:
            u (float): waveguide core geometry free parameter

        Returns: alpha_prop (float)

        """

        return self.models.α_wg_of_u(u=u) + (
            self.models.gamma_of_u(u) * self.models.α_fluid
        )

    def _line_element(self, r: float, w: float) -> float:
        """
        Line element for numerical integration in polar coordinates (r, theta)
        "dl = sqrt(r**2 + (dr/dtheta)**2)*dtheta", converted to a function of "r" only
        with the spiral equation (r = a + b*theta) to "dl(r) = sqrt((r/b)**2 + 1)*dr".

        Args:
            r (float): spiral outer radius
            w (float): waveguide core width

        Returns: dl (float)

        """

        _, _, b = self._calc_spiral_parameters(w=w)

        return np.sqrt((r / b) ** 2 + 1)

    def _line_element_bend_loss(self, r: float, u: float, w: float) -> float:
        """
        Bending losses for a line element: alpha_bend(r)*dl(r)

        Args:
            r (float): spiral outer radius
            u (float): waveguide core geometry free parameter
            w (float): waveguide core width

        Returns: alpha_bend(r)*dl(r) (float)

        """

        return self.models.α_bend(r=r, u=u) * self._line_element(r=r, w=w)

    def _calc_sensitivity(
        self, r: float, u: float, n_turns: float
    ) -> tuple[float, float, float, float]:
        """
        Calculate sensitivity at radius r for a given core height & width
        and number of turns

        Args:
            r (float): spiral outer radius
            u (float): waveguide core geometry free parameter
            n_turns (float): number of turns in the spiral

        Returns:
            s_nr (float): Snr
            outer_spiral_r_min (float): outer spiral minimum radius
            l_total (float): total length of spiral
            wg_a2 (float): a2 loss parameter for spiral waveguides

        """

        # Determine waveguide core width
        w: float = (
            self.models.parms.wg.v_coord_value
            if self.models.parms.wg.v_coord_name == "w"
            else u
        )

        # Archimedes spiral: "r = a + b*theta", where "a" is the minimum radius of the
        # outer spiral and "2*pi*b" is the spacing between the lines pairs.
        # Check that r is above the allowed minimum.
        line_width, a_spiral, b_spiral = self._calc_spiral_parameters(w=w)
        a_spiral_outer: float = a_spiral + self.models.parms.spiral.spacing + w

        # Spiral radii and angle extrema
        theta_max: float = (r - a_spiral_outer) / b_spiral
        theta_min: float = theta_max - (n_turns * 2 * np.pi)
        if theta_min < 0:
            return 1.0e-10, 0, 0, 0

        # Spiral radii extrema
        outer_spiral_r_max: float = r
        outer_spiral_r_min: float = a_spiral_outer + b_spiral * theta_min
        inner_spiral_r_max: float = outer_spiral_r_max - line_width / 2
        inner_spiral_r_min: float = outer_spiral_r_min - line_width / 2

        # Calculate the total spiral length (sum of outer and inner spirals)
        # by numerical integration w/r to the radius.
        length: float = (
            integrate.quad(
                func=self._line_element,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
                args=(w,),
            )[0]
            + integrate.quad(
                func=self._line_element,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
                args=(w,),
            )[0]
        )

        # Calculate propagation losses in the spiral
        α_prop: float = self._α_prop(u=u)
        prop_losses_spiral: float = α_prop * length

        # Calculate bending losses in the spiral by integration w/r to radius
        # (sum for inner and outer spiral losses)
        bend_losses_spiral: float = (
            integrate.quad(
                func=self._line_element_bend_loss,
                a=outer_spiral_r_min,
                b=outer_spiral_r_max,
                args=(
                    u,
                    w,
                ),
            )[0]
            + integrate.quad(
                func=self._line_element_bend_loss,
                a=inner_spiral_r_min,
                b=inner_spiral_r_max,
                args=(
                    u,
                    w,
                ),
            )[0]
        )

        # Approximate the "joint" between the two parallel waveguides in the spiral
        # at the center by:
        # - Outer waveguide: Half circle of radius equal to the outer
        #                    spiral minimum radius (length = L_HC)
        # - Inner waveguide: "S-bend" with 2 half circles of radii equal to half
        #                    the minimum radii of the inner spiral (length = L_SB1)
        #                    and the outer spiral (length = L_SB2).
        l_hc: float = np.pi * outer_spiral_r_min
        l_sb1: float = np.pi * (inner_spiral_r_min / 2)
        l_sb2: float = np.pi * (outer_spiral_r_min / 2)
        prop_losses_joint: float = α_prop * (l_hc + l_sb1 + l_sb2)
        bend_losses_joint: float = (
            self.models.α_bend(r=outer_spiral_r_min, u=u) * l_hc
            + self.models.α_bend(r=inner_spiral_r_min / 2, u=u) * l_sb1
            + self.models.α_bend(r=outer_spiral_r_min / 2, u=u) * l_sb2
        )

        # Total losses in the spiral
        total_losses: float = (
            prop_losses_spiral
            + bend_losses_spiral
            + prop_losses_joint
            + bend_losses_joint
        )
        wg_a2: float = np.e**-total_losses

        # Calculate the sensitivity of the spiral
        l_total: float = length + l_hc + l_sb1 + l_sb2
        s_nr: float = (
            (4 * np.pi / self.models.parms.wg.lambda_resonance)
            * l_total
            * self.models.gamma_of_u(u)
            * wg_a2
        )

        return s_nr, outer_spiral_r_min, l_total, wg_a2

    def _obj_fun(self, x: np.ndarray, r: float) -> float:
        """
        Objective function for the optimization in find_max_sensitivity()

        Args:
            x (float): array of free parameters in the optimization (u, n_turns)
            r (float): spiral outer radius

        Returns: negative of sensitivity, scaled by 1000 to aid convergence (float)

        """

        # Fetch the solution vector components
        u: float = x[0]
        n_turns: float = x[1]

        # Minimizer sometimes tries values of the solution vector outside the bounds...
        u = min(u, self.models.parms.limits.u_max)
        u = max(u, self.models.parms.limits.u_min)
        n_turns = max(n_turns, 0)

        # Calculate sensitivity at current solution vector S(r, u, n_turns)
        s, _, _, _ = self._calc_sensitivity(r=r, u=u, n_turns=n_turns)
        assert s >= 0, "S should not be negative!"

        return -s / 1000

    def _find_max_sensitivity(
        self, r: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """
        Calculate maximum sensitivity at r over all h and n_turns

        Args:
            r (float): spiral outer radius

        Returns:
            s (float): maximum sensitivity, max{S}
            u_max_s (float): u @ max{S}
            n_turns_max_s (float): number of turns @ max{S}
            outer_spiral_r_min (float): outer spiral radius @ max{S}
            length (float): spiral length @ max{S}
            gamma (float): waveguide gamma @ max{S}
            a2 (float): waveguide loss factor a2 @ max{S}

        """

        # Determine search domain extrema for u
        u_min, u_max = self.models.u_search_domain(r)

        # Spiral properties at minimum core width
        w_min: float = (
            self.models.parms.wg.v_coord_value
            if self.models.parms.wg.v_coord_name == "w"
            else u_min
        )
        line_width_min, a_spiral_min, b_spiral_min = self._calc_spiral_parameters(
            w=w_min
        )
        theta_min: float = 2 * np.pi * self.models.parms.spiral.turns_min
        r_min: float = (
            a_spiral_min + self.models.parms.spiral.spacing + w_min
        ) + theta_min * b_spiral_min

        # Determine search domain maximum for the numer of turns in the spiral
        n_turns_max: float = (r - r_min) / line_width_min
        n_turns_max = max(n_turns_max, self.models.parms.spiral.turns_min)
        n_turns_max = min(n_turns_max, self.models.parms.spiral.turns_max)

        # Only proceed with the minimization if the radius at which the solution is
        # sought is greater than the minimum allowed outer spiral radius.
        if r >= r_min + line_width_min * self.models.parms.spiral.turns_min:
            # If this is the first optimization, set the initial guesses for u at the
            # maximum value in the domain and the numbers of turns at the minimum
            # value (at small radii, bending losses are high, the optimal solution
            # will be at high u and low number of turns),else use previous solution.
            if np.any(self.previous_solution == -1):
                u0: float = u_max
                n_turns_0: float = self.models.parms.spiral.turns_min
            else:
                u0, n_turns_0 = self.previous_solution

            # Find h and n_turns that maximize S at radius R
            optimization_result: optimize.OptimizeResult = optimize.minimize(
                fun=self._obj_fun,
                x0=np.asarray([u0, n_turns_0]),
                bounds=(
                    (u_min, u_max),
                    (self.models.parms.spiral.turns_min, n_turns_max),
                ),
                args=(r,),
                method=self.models.parms.fit.optimization_method,
                tol=1e-9,
            )
            u_max_s: float = optimization_result.x[0]
            n_turns_max_s: float = optimization_result.x[1]

            # Calculate maximum sensitivity at the solution
            s, outer_spiral_r_min, length, a2 = self._calc_sensitivity(
                r=r, u=u_max_s, n_turns=n_turns_max_s
            )

            # Update previous solution
            self.previous_solution = np.asarray([u_max_s, n_turns_max_s])

        else:
            u_max_s = u_max
            n_turns_max_s = 0
            s = 1
            outer_spiral_r_min = 0
            length = 0
            a2 = 0

            # Update previous solution
            self.previous_solution = np.asarray(
                [u_max, self.models.parms.spiral.turns_min]
            )

        # Calculate other useful parameters at the solution
        gamma: float = self.models.gamma_of_u(u_max_s) * 100

        return s, u_max_s, n_turns_max_s, outer_spiral_r_min, length, gamma, a2

    def analyze(self) -> None:
        """
        Analyse the sensor performance for all radii in the r domain,
        store the results in the Spiral object instance

        return: None
        """

        # Analyse the sensor performance for all radii in the R domain
        self.results = [self._find_max_sensitivity(r=r) for r in self.models.r]

        # Unpack the analysis results as a function of radius into separate lists, the
        # order must be the same as in the find_max_sensitivity() return statement above
        [
            self.s,
            self.u,
            self.n_turns,
            self.outer_spiral_r_min,
            self.l,
            self.gamma,
            self.wg_a2,
        ] = list(np.asarray(self.results).T)

        # Find maximum sensitivity overall and corresponding radius
        self.max_s = np.amax(self.s)
        self.max_s_radius = self.models.r[np.argmax(self.s)]

        # If dynamic range of mode solver data exceeded for the spiral, show warning
        inner_spiral_r_min: float = (
            min(self.outer_spiral_r_min[self.outer_spiral_r_min > 0])
            - self.line_width_min
        )
        if inner_spiral_r_min < self.models.r_α_bend_data_min:
            self.logger(
                "WARNING: Minimum spiral bend radius "
                + f"({inner_spiral_r_min:.2f} um) below minimum value in mode solver "
                + f"data ({self.models.r_α_bend_data_min:.2f} um)!"
            )

        # Console message
        self.logger("Spiral sensor analysis done.")
