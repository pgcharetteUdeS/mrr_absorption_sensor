"""

Plotting utilities

Exposed methods:
   - plot_results()

"""


# Standard library packages
from colorama import Fore, Style
from openpyxl.workbook import Workbook
import io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, TiffImagePlugin

# Package modules
from .models import Models
from .mrr import Mrr
from .linear import Linear
from .spiral import Spiral
from .fileio import write_image_data_to_Excel
from .constants import PER_UM_TO_DB_PER_CM


def _calc_5X_10X_comp_data(
    R: np.ndarray, R_max_Smrr_index: int, mrr: Mrr, spiral: Spiral
) -> tuple[int, float, float, int, float, float]:
    """
    Calculate comparison info between MRR and spiral results for use in Fig.6 & Fig.6b
    """

    R_5X_index: int = int(
        (np.abs((mrr.S[:R_max_Smrr_index] / spiral.S[:R_max_Smrr_index]) - 5)).argmin()
    )
    R_5X: float = R[R_5X_index]
    R_5X_S: float = mrr.S[R_5X_index]

    R_10X_index: int = int(
        (np.abs((mrr.S[:R_max_Smrr_index] / spiral.S[:R_max_Smrr_index]) - 10)).argmin()
    )
    R_10X: float = R[R_10X_index]
    R_10X_S: float = mrr.S[R_10X_index]

    return R_5X_index, R_5X, R_5X_S, R_10X_index, R_10X, R_10X_S


def _calc_plotting_extrema(models: Models, mrr: Mrr) -> dict:

    # R domain extrema (complete decades)
    plotting_extrema: dict = {
        "r_plot_min": 10 ** (np.floor(np.log10(models.Rmin))),
        "r_plot_max": 10 ** (np.ceil(np.log10(models.Rmax))),
    }

    # u domain extrema
    u: np.ndarray = np.asarray(list(models.bending_loss_data.keys()))
    plotting_extrema["u_plot_min"] = u[0] * 0.9
    plotting_extrema["u_plot_max"] = u[-1] * 1.1

    # max{S} vertical marker
    plotting_extrema["S_plot_max"] = 10 ** np.ceil(np.log10(mrr.max_S))

    # Other extrema for Mrr plots
    plotting_extrema["Se_plot_max"] = (
        np.ceil(np.amax(mrr.Se * np.sqrt(mrr.a2)) * 1.1 / 10) * 10
    )
    plotting_extrema["Finesse_plot_max"] = (
        np.ceil(np.amax(mrr.Finesse / (2 * np.pi)) * 1.1 / 10) * 10
    )
    plotting_extrema["gamma_plot_min"] = np.floor(np.amin(mrr.gamma) * 0.9 / 10) * 10
    plotting_extrema["gamma_plot_max"] = np.ceil(np.amax(mrr.gamma) * 1.1 / 10) * 10

    return plotting_extrema


def _append_image_to_seq(images: list, fig: plt.Figure):
    """
    Append a matplotlib Figure to a list of PIL Image objects (from figs2tiff.py)

    NB: An empty list must be declared in the calling program prior to
        the first call to the function.

    Parameters
    ----------
    images :
        List of PIL Image objects to which to append the figure.
    fig : matplotlib.pyplot.Figure
        matplotlib Figure object to append to the List.

    Returns
    -------
    None.

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


def _write_spiral_sequence_to_file(
    models: Models, spiral: Spiral, filename_path: Path, logger=print
):
    """
    Write sequence of consecutive spirals with n turns > spiral.n_turns_min
    """

    # Calculate spiral sequence looping indices (min, max, index)
    biggest_spiral_index: int = int(np.argmax(spiral.n_turns))
    index_min: int = int(
        np.argmax(spiral.n_turns[:biggest_spiral_index] > spiral.turns_min)
    )
    index_max: int = (
        int(np.argmax(spiral.n_turns[biggest_spiral_index:] < 1)) + biggest_spiral_index
    )
    indices: range = range(index_min, index_max)

    # Check for adequate range of spirals in sequence, else exit with warning
    if len(indices) <= 2:
        logger(
            f"{Fore.YELLOW}WARNING! Insufficient range in number of spiral turns "
            + f"(array indices: [{indices[0]}, {indices[-1]}]), "
            + f"max number of turns = {spiral.n_turns[biggest_spiral_index]:.1f}, "
            + f"no sequence written!{Style.RESET_ALL}"
        )
        return

    # Loop to write generate the spiral images in the sequence
    fig, _ = plt.subplots()
    images: list = []
    for index in indices:
        fig, *_ = spiral.draw_spiral(
            r_outer=models.R[index],
            h=spiral.u[index] if models.core_v_name == "w" else models.core_v_value,
            w=models.core_v_value if models.core_v_name == "w" else spiral.u[index],
            n_turns=spiral.n_turns[index],
            r_window=(models.R[index_max] // 25 + 1) * 25,
            figure=fig,
        )
        _append_image_to_seq(images=images, fig=fig)
    plt.close(fig)

    # Save sequence to tiff multi-image file
    filename = filename_path.parent / f"{filename_path.stem}_SPIRAL_sequence.tif"
    images[0].save(
        str(filename),
        save_all=True,
        append_images=images[1:],
        duration=40,
    )
    logger(f"Wrote '{filename}'.")


def _plot_spiral_optimization_results(
    models: Models,
    spiral: Spiral,
    plotting_extrema: dict,
    filename_path: Path,
    logger=print,
):
    """ """

    # Plot max{S}, u, gamma, n turns mas, L
    fig, axs = plt.subplots(6)
    fig.suptitle(
        "Archimedes spiral\n"
        + f"{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
        + f", spacing = {spiral.spacing:.0f} μm"
        + f", min turns = {spiral.turns_min:.2}\n"
        + rf"max{{max{{$S$}}}} = {spiral.max_S:.0f} (RIU$^{{-1}}$)"
        + rf" @ $R$ = {spiral.max_S_radius:.0f} μm"
    )
    # max{S}
    axs_index = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)")
    axs[axs_index].loglog(models.R, spiral.S)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius],
        [100, plotting_extrema["S_plot_max"]],
        "--",
    )
    axs[axs_index].set_ylim(100, plotting_extrema["S_plot_max"])
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # u @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(f"{models.core_u_name} (μm)")
    axs[axs_index].semilogx(models.R, spiral.u)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius],
        [plotting_extrema["u_plot_min"], plotting_extrema["u_plot_max"]],
        "--",
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(
        plotting_extrema["u_plot_min"], plotting_extrema["u_plot_max"]
    )
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # gamma_fluid @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
    axs[axs_index].semilogx(models.R, spiral.gamma)
    axs[axs_index].plot([spiral.max_S_radius, spiral.max_S_radius], [0, 100], "--")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(0, 100)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # n turns @ max{S}
    axs_index += 1
    n_turns_plot_max: float = np.ceil(np.amax(spiral.n_turns) * 1.1 / 10) * 10 * 2
    axs[axs_index].set_ylabel("n turns\n(inner+outer)")
    axs[axs_index].semilogx(models.R, spiral.n_turns * 2)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [0, n_turns_plot_max], "--"
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(0, n_turns_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # a2 @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"$a^2$")
    axs[axs_index].semilogx(models.R, spiral.a2)
    axs[axs_index].plot([spiral.max_S_radius, spiral.max_S_radius], [0, 1], "--")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # L @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel("L (μm)")
    axs[axs_index].loglog(models.R, spiral.L)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius],
        [100, plotting_extrema["S_plot_max"]],
        "--",
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(100, plotting_extrema["S_plot_max"])
    axs[axs_index].set_xlabel("Ring radius (μm)")
    filename = filename_path.parent / f"{filename_path.stem}_SPIRAL.png"
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")


def _write_spiral_waveguide_coordinates_to_Excel_file(
    spiral_waveguide_coordinates: dict, filename_path: Path, logger=print
):
    """
    Write the spiral inner and outer waveguide x/y coordinates to an Excel file
    """

    filename = filename_path.parent / f"{filename_path.stem}_SPIRAL_SCHEMATIC.xlsx"
    wb = Workbook()
    outer_spiral_sheet = wb["Sheet"]
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
    inner_spiral_sheet = wb.create_sheet("Inner waveguide")
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
    logger(f"Wrote '{filename}'.")


def _plot_spiral_results(
    models: Models,
    spiral: Spiral,
    plotting_extrema: dict,
    filename_path: Path,
    logger=print,
):
    """ """

    # Plot spiral optimization results: u, gamma, n turns, a2, L @max(S)
    _plot_spiral_optimization_results(
        models,
        spiral=spiral,
        plotting_extrema=plotting_extrema,
        filename_path=filename_path,
        logger=logger,
    )

    # Draw the spiral with the greatest number of turns found in the optimization
    if models.parameters["draw_largest_spiral"]:
        largest_spiral_index: int = int(np.argmax(spiral.n_turns))
        (fig, spiral_waveguide_coordinates,) = spiral.draw_spiral(
            r_outer=models.R[largest_spiral_index],
            h=spiral.u[largest_spiral_index]
            if models.core_v_name == "w"
            else models.core_v_value,
            w=models.core_v_value
            if models.core_v_name == "w"
            else spiral.u[largest_spiral_index],
            n_turns=spiral.n_turns[largest_spiral_index],
            r_window=(models.R[largest_spiral_index] // 25 + 1) * 25,
        )
        filename = filename_path.parent / f"{filename_path.stem}_SPIRAL_SCHEMATIC.png"
        fig.savefig(fname=filename)
        logger(f"Wrote '{filename}'.")

        # Write the spiral inner and outer waveguide x/y coordinates to an Excel file
        if models.parameters["write_excel_files"]:
            _write_spiral_waveguide_coordinates_to_Excel_file(
                spiral_waveguide_coordinates=spiral_waveguide_coordinates,
                filename_path=filename_path,
                logger=logger,
            )

    # Write sequence of consecutive spirals with n turns > spiral.n_turns_min
    if models.parameters["write_spiral_sequence_to_file"]:
        _write_spiral_sequence_to_file(
            models, spiral=spiral, filename_path=filename_path, logger=logger
        )


def _plot_2D_maps(
    models: Models,
    mrr: Mrr,
    plotting_extrema: dict,
    filename_path: Path,
    logger=print,
):

    # Define extra line styles
    # See "https://matplotlib.org/3.5.1/gallery/lines_bars_and_markers/linestyles.html"
    linestyles: dict = {
        "loosely dotted": (0, (1, 10)),
        "dotted": (0, (1, 1)),
        "densely dotted": (0, (1, 1)),
        "loosely dashed": (0, (5, 10)),
        "dashed": (0, (5, 5)),
        "densely dashed": (0, (5, 1)),
        "loosely dashdotted": (0, (3, 10, 1, 10)),
        "dashdotted": (0, (3, 5, 1, 5)),
        "densely dashdotted": (0, (3, 1, 1, 1)),
        "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
        "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
        "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
    }

    # Generate 2D map data R,u arrays (x/y)
    R_2D_map = np.linspace(
        np.log10(models.R[0]),
        np.log10(models.R[-1]),
        models.parameters["map2D_n_grid_points"],
    )
    u_2D_map = np.linspace(
        list(models.bending_loss_data)[0],
        list(models.bending_loss_data)[-1],
        models.parameters["map2D_n_grid_points"],
    )

    # Indices for dashed lines at radii for max(Smrr)
    R_max_Smrr_index: int = int((np.abs(models.R - mrr.max_S_radius)).argmin())
    R_max_Smrr_u: float = mrr.u[R_max_Smrr_index]

    #
    # 2D maps as a function of R/u
    #

    # 2D map of S(u, R)
    S_2D_map = np.asarray(
        [
            [mrr.calc_sensitivity(r=10**log10_R, u=u)[0] for log10_R in R_2D_map]
            for u in u_2D_map
        ]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, u_2D_map, S_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.invert_yaxis()
    ax.set_title(
        f"MRR sensitivity as a function of {models.core_u_name} and R\n"
        + f"{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(f"{models.core_u_name} (μm)")
    fig.colorbar(cm, label=r"S (RIU $^{-1}$)")
    ax.plot(
        np.log10(models.R),
        mrr.u,
        color=models.parameters["map2D_overlay_color_light"],
        label=r"max$\{S(h, R)\}$",
    )
    """
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [u_2D_map[-1], R_max_Smrr_u],
        color=models.parameters["map2D_overlay_color_dark"],
    )
    """
    ax.plot(
        [R_2D_map[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_u, R_max_Smrr_u],
        color=models.parameters["map2D_overlay_color_light"],
        linestyle=linestyles["loosely dashdotted"],
        label=rf"max{{max{{$S_{{MRR}}$}}}} = {mrr.max_S:.0f} RIU $^{{-1}}$"
        + f" @ R = {mrr.max_S_radius:.0f} μm"
        + f", {models.core_u_name} = {R_max_Smrr_u:.3f} μm",
    )
    ax.plot(
        np.log10(mrr.Re),
        mrr.u_resampled,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw),
        mrr.u_resampled,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="-.",
        label=r"Rw$(\Gamma_{fluid})$",
    )
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent
        / f"{filename_path.stem}_MRR_2DMAP_S_VS_{models.core_u_name}_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")
    if models.parameters["write_excel_files"]:
        write_image_data_to_Excel(
            filename=str(
                filename_path.parent
                / f"{filename_path.stem}_MRR_2DMAPS_VS_{models.core_u_name}_and_R.xlsx"
            ),
            X=10**R_2D_map,
            x_label="R (um)",
            Y=u_2D_map,
            y_label=f"{models.core_u_name} (um)",
            Zs=[S_2D_map],
            z_labels=["S (RIU-1)"],
        )
        logger(f"Wrote '{filename.with_suffix('.xlsx')}'.")

    #
    # 2D maps as a function of R/gamma
    #

    # Generate gamma(u) array matching u array. If the values are not monotonically
    # decreasing due to positive curvature of the modeled values at the beginning of
    # the array, flag as warning and replace values, else pcolormesh() complains.
    gamma_2D_map = np.asarray([models.gamma_of_u(u) * 100 for u in u_2D_map])
    if np.any(np.diff(gamma_2D_map) > 0):
        gamma_2D_map[: int(np.argmax(gamma_2D_map))] = gamma_2D_map[
            int(np.argmax(gamma_2D_map))
        ]
        logger(
            f"{Fore.YELLOW}WARNING! Gamma({models.core_u_name}) is not monotonically "
            + f"decreasing, first values replaced with gamma max!{Style.RESET_ALL}"
        )

    # Indices for dashed lines at radii for max(Smrr)
    R_max_Smrr_gamma: float = mrr.gamma[R_max_Smrr_index]

    # 2D map of Smrr(gamma, R)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, gamma_2D_map, S_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.set_title(
        r"MRR sensitivity, $S_{MRR}$, as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$S_{MRR}$ (RIU $^{-1}$)")
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_light"],
        label=r"max$\{S_{MRR}(\Gamma_{fluid}, R)\}$",
    )
    """
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [gamma_2D_map[-1], R_max_Smrr_gamma],
        color=models.parameters["map2D_overlay_color_light"],
    )
    """
    ax.plot(
        [R_2D_map[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_gamma, R_max_Smrr_gamma],
        color=models.parameters["map2D_overlay_color_light"],
        linestyle=linestyles["loosely dashdotted"],
        label=rf"max{{max{{$S_{{MRR}}$}}}} = {mrr.max_S:.0f} RIU$^{{-1}}$"
        + f" @ R = {mrr.max_S_radius:.0f} μm"
        + rf", $\Gamma$ = {R_max_Smrr_gamma:.0f}$\%$",
    )
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="-.",
        label=r"Rw$(\Gamma_{fluid})$",
    )
    for line in models.parameters["map_line_profiles"] or []:
        ax.plot(
            [
                np.log10(plotting_extrema["r_plot_min"]),
                np.log10(plotting_extrema["r_plot_max"]),
            ],
            [line, line],
            color=models.parameters["map2D_overlay_color_light"],
            linestyle=linestyles["loosely dotted"],
        )
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent / f"{filename_path.stem}_MRR_2DMAP_S_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # 2D map of Snr(gamma, R)
    Snr_2D_map = np.asarray(
        [[mrr.calc_Snr(r=10**log10_R, u=u) for log10_R in R_2D_map] for u in u_2D_map]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, gamma_2D_map, Snr_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_dark"],
        label=r"max$\{S_{MRR}\}$",
    )
    ax.set_title(
        r"MRR $S_{NR}$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$S_{NR}$ (RIU$^{-1}$)")
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent / f"{filename_path.stem}_MRR_2DMAP_Snr_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # 2D map of Se(gamma, R)
    Se_2D_map = np.asarray(
        [[mrr.calc_Se(r=10**log10_R, u=u) for log10_R in R_2D_map] for u in u_2D_map]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, gamma_2D_map, Se_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_dark"],
        label=r"max$\{S_{MRR}\}$",
    )
    ax.set_title(
        r"MRR $S_e$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$S_e$")
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent / f"{filename_path.stem}_MRR_2DMAP_Se_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # 2D map of Se*a(gamma, R)
    Se_times_a_2D_map = np.asarray(
        [
            [
                mrr.calc_Se(r=10**log10_R, u=u)
                * np.sqrt(mrr.calc_a2(r=10**log10_R, u=u))
                for log10_R in R_2D_map
            ]
            for u in u_2D_map
        ]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map,
        gamma_2D_map,
        Se_times_a_2D_map,
        cmap=models.parameters["map2D_colormap"],
    )
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_dark"],
        label=r"max$\{S_{MRR}\}$",
    )
    ax.set_title(
        r"MRR $S_e \times a$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f}μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$S_e \times a$")
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent
        / f"{filename_path.stem}_MRR_2DMAP_Se_x_a_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # 2D map of a2(gamma, R)
    a2_2D_map = np.asarray(
        [[mrr.calc_a2(r=10**log10_R, u=u) for log10_R in R_2D_map] for u in u_2D_map]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, gamma_2D_map, a2_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_light"],
        label=r"max$\{S_{MRR}\}$",
    )
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_light"],
        linestyle="-.",
        label=r"Rw$(\Gamma_{fluid})$",
    )
    ax.set_title(
        r"MRR $a^2$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$a^2$")
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent / f"{filename_path.stem}_MRR_2DMAP_a2_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # 2D map of alpha*L(gamma, R)
    dB_per_cm_to_per_cm: float = 1.0 / 4.34
    alpha_L_2D_map = (
        np.asarray(
            [
                [mrr.calc_alpha_L(r=10**log10_R, u=u) for log10_R in R_2D_map]
                for u in u_2D_map
            ]
        )
        / dB_per_cm_to_per_cm
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(
        R_2D_map, gamma_2D_map, alpha_L_2D_map, cmap=models.parameters["map2D_colormap"]
    )
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color=models.parameters["map2D_overlay_color_dark"],
        label=r"max$\{S_{MRR}\}$",
    )
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_dark"],
        linestyle="--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw),
        mrr.gamma_resampled * 100,
        color=models.parameters["map2D_overlay_color_dark"],
        linestyle="-.",
        label=r"Rw$(\Gamma_{fluid})$",
    )
    for line in models.parameters["map_line_profiles"] or []:
        ax.plot(
            [
                np.log10(plotting_extrema["r_plot_min"]),
                np.log10(plotting_extrema["r_plot_max"]),
            ],
            [line, line],
            color=models.parameters["map2D_overlay_color_dark"],
            linestyle=linestyles["loosely dotted"],
        )
    ax.set_title(
        r"MRR $\alpha L$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm"
    )
    ax.set_xlabel("log(R) (μm)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$\alpha L$ (dB)")
    ax.set_xlim(
        left=np.log10(plotting_extrema["r_plot_min"]),
        right=np.log10(plotting_extrema["r_plot_max"]),
    )
    ax.set_ylim(bottom=gamma_2D_map[-1], top=gamma_2D_map[0])
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent
        / f"{filename_path.stem}_MRR_2DMAP_alphaL_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # Save 2D map data as a function of gamma and R to output Excel file, if required
    if models.parameters["write_excel_files"]:
        # In addition to alpha*L, calculate 2D maps of alpha_prop*L and alpha_bend*L
        alpha_prop_L_2D_map = (
            np.asarray(
                [
                    [
                        mrr.calc_alpha_prop_L(r=10**log10_R, u=u)
                        for log10_R in R_2D_map
                    ]
                    for u in u_2D_map
                ]
            )
            / dB_per_cm_to_per_cm
        )
        alpha_bend_L_2D_map = (
            np.asarray(
                [
                    [
                        mrr.calc_alpha_bend_L(r=10**log10_R, u=u)
                        for log10_R in R_2D_map
                    ]
                    for u in u_2D_map
                ]
            )
            / dB_per_cm_to_per_cm
        )

        # Write all 2D maps to single Excel file
        write_image_data_to_Excel(
            filename=str(
                filename_path.parent
                / f"{filename_path.stem}_MRR_2DMAPS_VS_GAMMA_and_R.xlsx"
            ),
            X=10**R_2D_map,
            x_label="R (um)",
            Y=gamma_2D_map,
            y_label="gamma (%)",
            Zs=[
                S_2D_map,
                Snr_2D_map,
                Se_2D_map,
                alpha_L_2D_map,
                alpha_bend_L_2D_map,
                alpha_prop_L_2D_map,
            ],
            z_labels=[
                "S (RIU-1)",
                "Snr (RIU-1)",
                "Se",
                "alpha x L (dB)",
                "alpha_bend x L (dB)",
                "alpha_prop x L (dB)",
            ],
        )
        logger(f"Wrote '{filename.with_suffix('.xlsx')}'.")


def _plot_mrr_optimization_results(
    models: Models,
    mrr: Mrr,
    plotting_extrema: dict,
    filename_path: Path,
    logger=print,
):
    """ """

    #
    # Plot of sensing parameters
    #

    # max{S}, S_NR, Se, a, u, gamma, Finesse
    fig, axs = plt.subplots(7)
    fig.suptitle(
        "MRR - Sensing parameters\n"
        + f"{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm\n"
        + rf"max{{max{{$S$}}}} = {mrr.max_S:.0f} (RIU$^{{-1}}$)"
        + rf" @ $R$ = {mrr.max_S_radius:.0f} μm"
    )

    # max{S}
    axs_index: int = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)")
    axs[axs_index].loglog(models.R, mrr.S)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [100, plotting_extrema["S_plot_max"]],
        "r--",
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(100, plotting_extrema["S_plot_max"])
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # S_NR @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.Snr)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [10, plotting_extrema["S_plot_max"]],
        "r--",
    )
    axs[axs_index].set_ylabel(r"S$_{NR}$" + "\n" + r"(RIU $^{-1}$)")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(10, plotting_extrema["S_plot_max"])
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Se @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Se * np.sqrt(mrr.a2))
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [0, plotting_extrema["Se_plot_max"]],
        "r--",
    )
    axs[axs_index].set_ylabel(r"S$_e \times a$")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(0, plotting_extrema["Se_plot_max"])
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, np.sqrt(mrr.a2))
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, 1], "r--")
    axs[axs_index].set_ylabel(r"$a$")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # u (h or w) @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.u)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [plotting_extrema["u_plot_min"], plotting_extrema["u_plot_max"]],
        "r--",
    )
    axs[axs_index].set_ylabel(f"{models.core_u_name} (μm)")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(
        plotting_extrema["u_plot_min"], plotting_extrema["u_plot_max"]
    )
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Gamma_fluid @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.gamma)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [plotting_extrema["gamma_plot_min"], plotting_extrema["gamma_plot_max"]],
        "r--",
    )
    axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(
        plotting_extrema["gamma_plot_min"], plotting_extrema["gamma_plot_max"]
    )
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # alpha_wg @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(
        models.R,
        np.asarray([mrr.models.alpha_wg(u) for u in mrr.u]) * PER_UM_TO_DB_PER_CM,
    )
    axs[axs_index].set_ylabel(r"α$_{wg}$")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )

    axs[axs_index].set_xlabel("Ring radius (μm)")
    filename: Path = filename_path.parent / f"{filename_path.stem}_MRR_sens_parms.png"
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    #
    # Plot of ring parameters
    #

    # max{S}, Q, Finesse, FWHM, FSR, contrast
    fig, axs = plt.subplots(6)
    fig.suptitle(
        "MRR - Ring parameters"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μm\n"
        + rf"max{{max{{$S$}}}} = {mrr.max_S:.0f} (RIU$^{{-1}}$)"
        + rf" @ $R$ = {mrr.max_S_radius:.0f} μm"
    )
    # max{S}
    axs_index = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$")
    axs[axs_index].loglog(models.R, mrr.S)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius],
        [100, plotting_extrema["S_plot_max"]],
        "r--",
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylim(100, plotting_extrema["S_plot_max"])
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Contrast, tau & a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.tau, color="blue", label="τ")
    axs[axs_index].semilogx(models.R, np.sqrt(mrr.a2), color="green", label="a")
    axs[axs_index].semilogx(models.R, mrr.contrast, color="red", label="contrast")
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, 1], "r--")
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylabel(r"Contrast, $a$, $\tau$")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])
    axs[axs_index].legend(loc="upper right")

    # ER @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.ER, label="Q")
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [0, np.amax(mrr.ER)], "r--"
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylabel("Extinction\nratio\n(dB)")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Q @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.Q, label="Q")
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [0, np.amax(mrr.Q)], "r--"
    )
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylabel("Q")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # (Finesse/2pi) / Se*a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(
        models.R,
        mrr.Finesse / (2 * np.pi) / (mrr.Se * np.sqrt(mrr.a2)),
    )
    axs[axs_index].set_ylim(0, 2.5)
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylabel(r"$\frac{Finesse/2\pi}{S_e\times a}$")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # FWHM, FSR, Finesse/2pi @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.FWHM * 1e6, "b", label="FWHM")
    axs[axs_index].loglog(models.R, mrr.FSR * 1e6, "g", label="FSR")
    axs[axs_index].set_xlim(
        plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]
    )
    axs[axs_index].set_ylabel("FWHM and FSR\n(pm)")
    axs[axs_index].set_xlabel("Ring radius (μm)")
    axR = axs[axs_index].twinx()
    axR.semilogx(models.R, mrr.Finesse / (2 * np.pi), "k--", label="Finesse/2π")
    axR.set_ylabel("Finesse/2π")
    axR.grid(visible=False)
    ax_lines = (
        axs[axs_index].get_legend_handles_labels()[0]
        + axR.get_legend_handles_labels()[0]
    )
    ax_labels = (
        axs[axs_index].get_legend_handles_labels()[1]
        + axR.get_legend_handles_labels()[1]
    )
    axs[axs_index].legend(ax_lines, ax_labels, loc="upper right")
    axs[axs_index].patch.set_visible(False)
    axR.patch.set_visible(True)
    axs[axs_index].set_zorder(axR.get_zorder() + 1)

    # Write figure to file
    filename = filename_path.parent / (filename_path.stem + "_MRR_ring_parms.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")


def _plot_combined_linear_spiral_mrr_results(
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    plotting_extrema: dict,
    filename_path: Path,
    logger=print,
):
    """
    Overlaid plots of linear, spiral and MRR waveguide sensitivity as function of R
    """

    # Calculate minimum sensitivity required to detect the minimum resolvable
    # change in ni for a given transmission measurement SNR
    S_min: float = (
        10 ** (-models.parameters["T_SNR"] / 10) / models.parameters["min_delta_ni"]
    )

    # Plot...
    fig, ax = plt.subplots()
    ax.set_title(
        "Maximum sensitivity for MRR and linear sensors"
        + f"\n{models.pol}"
        + f", λ = {models.lambda_res:.3f} μm"
        + rf", min(α$_{{wg}}$) = {models.alpha_wg_dB_per_cm:.1f} dB/cm"
        + f", {models.core_v_name} = {models.core_v_value:.3f} μ"
        if models.parameters["no_spiral"]
        else "Maximum sensitivity for MRR, spiral, and linear sensors"
    )

    # MRR
    ax.set_xlabel("Ring radius (μm)")
    ax.set_ylabel(r"Maximum sensitivity (RIU$^{-1}$)")
    ax.loglog(models.R, mrr.S, color="b", label="MRR")

    # Linear waveguide
    ax.loglog(models.R, linear.S, color="g", label=r"Linear waveguide ($L = 2R$)")
    ax.loglog(
        [plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"]],
        [S_min, S_min],
        "r--",
        label="".join(
            [
                r"min$\{S\}$ to resolve $\Delta n_{i}$",
                f" = {models.parameters['min_delta_ni']:.0E} "
                + f"@ SNR = {models.parameters['T_SNR']:.0f} dB",
            ]
        ),
    )
    ax.set_xlim(plotting_extrema["r_plot_min"], plotting_extrema["r_plot_max"])
    ax.set_ylim(100, plotting_extrema["S_plot_max"])

    # Spiral and MRR/spiral sensitivity ratio, if required
    if not models.parameters["no_spiral"]:
        ax.loglog(
            models.R[spiral.S > 1],
            spiral.S[spiral.S > 1],
            color="k",
            label=f"Spiral (spacing = {spiral.spacing:.0f} μm"
            + f", min turns = {spiral.turns_min:.2f})",
        )
        axR = ax.twinx()
        axR.semilogx(
            models.R,
            mrr.S / spiral.S,
            "k--",
            label=r"max$\{S_{MRR}\}$ / max$\{S_{SPIRAL}\}$",
        )
        axR.set_ylabel(r"max$\{S_{MRR}\}$ / max$\{S_{SPIRAL}\}$")
        axR.set_ylim(0, 30)
        axR.grid(visible=False)
        ax_lines = (
            ax.get_legend_handles_labels()[0] + axR.get_legend_handles_labels()[0]
        )
        ax_labels = (
            ax.get_legend_handles_labels()[1] + axR.get_legend_handles_labels()[1]
        )
        ax.legend(ax_lines, ax_labels, loc="lower right")
        ax.patch.set_visible(False)
        axR.patch.set_visible(True)
        ax.set_zorder(axR.get_zorder() + 1)
    else:
        ax.legend(loc="lower right")

    # Save figure
    filename = filename_path.parent / f"{filename_path.stem}_MRR_VS_SPIRAL_VS_SWGD.png"
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")


def plot_results(
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    filename_path: Path,
    logger=print,
):
    """

    :param models:
    :param mrr:
    :param linear:
    :param spiral:
    :param filename_path:
    :param logger:
    :return: None
    """

    # Calculate plotting extrema and max{S} vertical marker position
    plotting_extrema: dict = _calc_plotting_extrema(models, mrr=mrr)

    # Plot/save MRR optimization results
    _plot_mrr_optimization_results(
        models,
        mrr=mrr,
        plotting_extrema=plotting_extrema,
        filename_path=filename_path,
        logger=logger,
    )

    # Plot/save MRR 2D result maps
    if models.parameters["write_2D_maps"]:
        _plot_2D_maps(
            models,
            mrr=mrr,
            plotting_extrema=plotting_extrema,
            filename_path=filename_path,
            logger=logger,
        )

    # Plot/save spiral results, if required
    if not models.parameters["no_spiral"]:
        _plot_spiral_results(
            models,
            spiral=spiral,
            plotting_extrema=plotting_extrema,
            filename_path=filename_path,
            logger=logger,
        )

    # Plot/save overlaid graphs of linear, spiral and MRR sensitivity as function of R
    _plot_combined_linear_spiral_mrr_results(
        models=models,
        mrr=mrr,
        linear=linear,
        spiral=spiral,
        plotting_extrema=plotting_extrema,
        filename_path=filename_path,
        logger=logger,
    )
