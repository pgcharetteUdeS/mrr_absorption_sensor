"""

Plotting utilities

Exposed methods:
   - plot_results()

"""


# Standard library packages
from colorama import Fore, Style
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


def _calc_plotting_extrema(
    models: Models, mrr: Mrr
) -> tuple[float, float, float, float, float, float, float, float, float]:

    # R domain extrema (complete decades)
    r_plot_min: float = 10 ** (np.floor(np.log10(models.Rmin)))
    r_plot_max: float = 10 ** (np.ceil(np.log10(models.Rmax)))

    # u domain extrema
    u: np.ndarray = np.asarray(list(models.bending_loss_data.keys()))
    u_plot_min: float = u[0] * 0.9
    u_plot_max: float = u[-1] * 1.1

    # max{S} vertical marker
    S_plot_max: float = 10 ** np.ceil(np.log10(mrr.max_S))

    # Other extrema for Mrr plots
    Se_plot_max: float = np.ceil(np.amax(mrr.Se * np.sqrt(mrr.a2)) * 1.1 / 10) * 10
    Finesse_plot_max: float = (
        np.ceil(np.amax(mrr.Finesse / (2 * np.pi)) * 1.1 / 10) * 10
    )
    gamma_plot_min: float = np.floor(np.amin(mrr.gamma) * 0.9 / 10) * 10
    gamma_plot_max: float = np.ceil(np.amax(mrr.gamma) * 1.1 / 10) * 10

    return (
        r_plot_min,
        r_plot_max,
        u_plot_min,
        u_plot_max,
        S_plot_max,
        Se_plot_max,
        Finesse_plot_max,
        gamma_plot_min,
        gamma_plot_max,
    )


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
        np.argmax(spiral.n_turns[:biggest_spiral_index] > (spiral.turns_min * 1.01))
    )
    index_max: int = (
        int(
            np.argmax(spiral.n_turns[biggest_spiral_index:] < (spiral.turns_min * 1.01))
        )
        + biggest_spiral_index
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
        fig = spiral.draw_spiral(
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


def _plot_spiral_results(
    models: Models,
    mrr: Mrr,
    spiral: Spiral,
    filename_path: Path,
    draw_largest_spiral: bool = False,
    write_spiral_sequence_to_file: bool = False,
    logger=print,
):

    # Calculate plotting extrema and max{S} vertical marker
    (
        r_plot_min,
        r_plot_max,
        u_plot_min,
        u_plot_max,
        S_plot_max,
        *_,
    ) = _calc_plotting_extrema(models=models, mrr=mrr)

    # Plot max{S}, u, gamma, n turns mas, L
    fig, axs = plt.subplots(6)
    fig.suptitle(
        "Archimedes spiral "
        + f" ({models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
        + "".join([f", spacing = {spiral.spacing:.0f} ", r"$\mu$m"])
        + f", min turns = {spiral.turns_min:.2}"
        + ")"
        + "\n"
        + "".join(
            [
                r"max$\{$max$\{S\}\}$ = ",
                f"{spiral.max_S:.0f}",
                r" (RIU$^{-1}$)",
            ]
        )
        + "".join([" @ $R$ = ", f"{spiral.max_S_radius:.0f}", r" $\mu$m"])
    )
    # max{S}
    axs_index = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)")
    axs[axs_index].loglog(models.R, spiral.S)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [100, S_plot_max], "--"
    )
    axs[axs_index].set_ylim(100, S_plot_max)
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # u @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel("".join([f"{models.core_u_name}", r" ($\mu$m)"]))
    axs[axs_index].semilogx(models.R, spiral.u)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [u_plot_min, u_plot_max], "--"
    )
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(u_plot_min, u_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # gamma_fluid @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
    axs[axs_index].semilogx(models.R, spiral.gamma)
    axs[axs_index].plot([spiral.max_S_radius, spiral.max_S_radius], [0, 100], "--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, 100)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # n turns @ max{S}
    axs_index += 1
    n_turns_plot_max: float = np.ceil(np.amax(spiral.n_turns) * 1.1 / 10) * 10
    axs[axs_index].set_ylabel(r"n turns")
    axs[axs_index].semilogx(models.R, spiral.n_turns)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [0, n_turns_plot_max], "--"
    )
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, n_turns_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # a2 @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"$a^2$")
    axs[axs_index].semilogx(models.R, spiral.a2)
    axs[axs_index].plot([spiral.max_S_radius, spiral.max_S_radius], [0, 1], "--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # L @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"L ($\mu$m)")
    axs[axs_index].loglog(models.R, spiral.L)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [100, S_plot_max], "--"
    )
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(100, S_plot_max)
    axs[axs_index].set_xlabel(r"Ring radius ($\mu$m)")
    filename = filename_path.parent / f"{filename_path.stem}_SPIRAL.png"
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # Draw the spiral with the greatest number of turns found in the optimization
    if draw_largest_spiral:
        largest_spiral_index: int = int(np.argmax(spiral.n_turns))
        fig = spiral.draw_spiral(
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

    # Write sequence of consecutive spirals with n turns > spiral.n_turns_min
    if write_spiral_sequence_to_file:
        _write_spiral_sequence_to_file(
            models=models, spiral=spiral, filename_path=filename_path, logger=logger
        )


def plot_results(
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    T_SNR: float,
    min_delta_ni: float,
    filename_path: Path,
    n_2D_grid_points: int = 500,
    write_excel_files: bool = False,
    colormap2D: str = "viridis",
    no_spiral: bool = False,
    draw_largest_spiral: bool = False,
    write_spiral_sequence_to_file: bool = False,
    logger=print,
):
    """

    :param models:
    :param mrr:
    :param linear:
    :param spiral:
    :param T_SNR:
    :param min_delta_ni:
    :param filename_path:
    :param n_2D_grid_points:
    :param write_excel_files:
    :param colormap2D:
    :param no_spiral:
    :param draw_largest_spiral:
    :param write_spiral_sequence_to_file:
    :param logger:
    :return: None
    """

    # Calculate plotting extrema and max{S} vertical marker
    (
        r_plot_min,
        r_plot_max,
        u_plot_min,
        u_plot_max,
        S_plot_max,
        Se_plot_max,
        Finesse_plot_max,
        gamma_plot_min,
        gamma_plot_max,
    ) = _calc_plotting_extrema(models=models, mrr=mrr)

    #
    # MRR results
    #

    # max{S}, S_NR, Se, a, u, gamma, Finesse
    fig, axs = plt.subplots(7)
    fig.suptitle(
        "Micro-ring resonator "
        + f"({models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join(
            [f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m", ")\n"]
        )
        + "".join(
            [
                r"max$\{$max$\{S\}\}$ = ",
                f"{mrr.max_S:.0f}",
                r" (RIU$^{-1}$)",
            ]
        )
        + "".join([r" @ $R$ = ", f"{mrr.max_S_radius:.0f}", r" $\mu$m"])
    )

    # max{S}
    axs_index: int = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$" + "\n" + r"(RIU$^{-1}$)")
    axs[axs_index].loglog(models.R, mrr.S)
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [100, S_plot_max], "r--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(100, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # S_NR @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.Snr)
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [10, S_plot_max], "r--")
    axs[axs_index].set_ylabel(r"S$_{NR}$" + "\n" + r"(RIU $^{-1}$)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(10, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Se @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Se * np.sqrt(mrr.a2))
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, Se_plot_max], "r--")
    axs[axs_index].set_ylabel(r"S$_e \times a$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, Se_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, np.sqrt(mrr.a2))
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, 1], "r--")
    axs[axs_index].set_ylabel(r"$a$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # u @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.u)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [u_plot_min, u_plot_max], "r--"
    )
    axs[axs_index].set_ylabel(f"{models.core_u_name}" + r" ($\mu$m)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(u_plot_min, u_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Gamma_fluid @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.gamma)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [gamma_plot_min, gamma_plot_max], "r--"
    )
    axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(gamma_plot_min, gamma_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Finesse/2pi (number of turns in the ring) @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Finesse / (2 * np.pi))
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [0, Finesse_plot_max], "r--"
    )
    axs[axs_index].set_ylabel(r"Finesse/$2\pi$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, Finesse_plot_max)

    axs[axs_index].set_xlabel(r"Ring radius ($\mu$m)")
    filename: Path = filename_path.parent / f"{filename_path.stem}_MRR_ALL.png"
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # max{S}, Q, Finesse, FWHM, FSR, contrast
    fig, axs = plt.subplots(5)
    fig.suptitle(
        "MRR - Ring resonator parameters"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
        + "".join(
            [
                "\n" r"max$\{$max$\{S\}\}$ = ",
                f"{mrr.max_S:.0f}",
                r" (RIU$^{-1}$)",
            ]
        )
        + "".join([r" @ $R$ = ", f"{mrr.max_S_radius:.0f}", r" $\mu$m"])
    )
    # max{S}
    axs_index = 0
    axs[axs_index].set_ylabel(r"max$\{S\}$ (RIU$^{-1}$)")
    axs[axs_index].loglog(models.R, mrr.S)
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [100, S_plot_max], "r--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(100, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Contrast, tau & a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.tau, color="blue", label=r"$\tau$")
    axs[axs_index].semilogx(models.R, np.sqrt(mrr.a2), color="green", label="a")
    axs[axs_index].semilogx(models.R, mrr.contrast, color="red", label="contrast")
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, 1], "r--")
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel(r"Contrast, $a$, $\tau$")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])
    axs[axs_index].legend(loc="upper right")

    # Q @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Q, label="Q")
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [0, np.amax(mrr.Q)], "r--"
    )
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel("Q")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # (Finesse/2pi) / Se*a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(
        models.R,
        mrr.Finesse / (2 * np.pi) / (mrr.Se * np.sqrt(mrr.a2)),
    )
    axs[axs_index].set_ylim(0, 2.5)
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel(r"$\frac{Finesse/2\pi}{S_e\times a}$")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # FWHM, FSR, Finesse/2pi @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.FWHM * 1e6, "b", label="FWHM")
    axs[axs_index].loglog(models.R, mrr.FSR * 1e6, "g", label="FSR")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel("FWHM and FSR (pm)")
    axs[axs_index].set_xlabel(r"Ring radius ($\mu$m)")
    axR = axs[axs_index].twinx()
    axR.semilogx(models.R, mrr.Finesse / (2 * np.pi), "k--", label=r"Finesse/2$\pi$")
    axR.set_ylabel(r"Finesse/2$\pi$")
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

    #
    # MRR 2D maps
    #

    # Generate 2D map data X,Y data arrays
    R_2D_map = np.linspace(
        np.log10(models.R[0]), np.log10(models.R[-1]), n_2D_grid_points
    )
    u_2D_map = np.linspace(
        list(models.bending_loss_data)[0],
        list(models.bending_loss_data)[-1],
        n_2D_grid_points,
    )
    gamma_2D_map = np.asarray([models.gamma_of_u(u) * 100 for u in u_2D_map])

    # Indices for dashed lines at radii for max(Smrr)
    R_max_Smrr_index: int = int((np.abs(models.R - mrr.max_S_radius)).argmin())
    R_max_Smrr_u: float = mrr.u[R_max_Smrr_index]
    R_max_Smrr_gamma: float = mrr.gamma[R_max_Smrr_index]

    # 2D map of S(u, R)
    S_2D_map = np.asarray(
        [
            [mrr.calc_sensitivity(r=10**log10_R, u=u)[0] for log10_R in R_2D_map]
            for u in u_2D_map
        ]
    )
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(R_2D_map, u_2D_map, S_2D_map, cmap=colormap2D)
    ax.invert_yaxis()
    ax.set_title(
        f"MRR sensitivity as a function of {models.core_u_name} and R"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(f"{models.core_u_name}" + r" ($\mu$m)")
    fig.colorbar(cm, label=r"S (RIU $^{-1}$)")
    ax.plot(np.log10(models.R), mrr.u, color="black", label=r"max$\{S(h, R)\}$")
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [u_2D_map[-1], R_max_Smrr_u],
        "w--",
        label="".join(
            [r"max$\{$max$\{S_{MRR}\}\}$", f" = {mrr.max_S:.0f} RIU", r"$^{-1}$"]
        )
        + "".join([f" @ R = {mrr.max_S_radius:.0f} ", r"$\mu$", "m"])
        + "".join([f", {models.core_u_name} = {R_max_Smrr_u:.3f} ", r"$\mu$m"]),
    )
    ax.plot(
        [R_2D_map[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_u, R_max_Smrr_u],
        "w--",
    )
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent
        / f"{filename_path.stem}_MRR_2DMAP_S_VS_{models.core_u_name}_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")
    if write_excel_files:
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

    # 2D map of Smrr(gamma, R)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, S_2D_map, cmap=colormap2D)
    ax.set_title(
        r"MRR sensitivity, $S_{MRR}$, as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"$S_{MRR}$ (RIU $^{-1}$)")
    ax.plot(
        np.log10(models.R),
        mrr.gamma,
        color="black",
        label=r"max$\{S_{MRR}(\Gamma_{fluid}, R)\}$",
    )
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [gamma_2D_map[-1], R_max_Smrr_gamma],
        "w--",
        label="".join(
            [r"max$\{$max$\{S_{MRR}\}\}$", f" = {mrr.max_S:.0f} RIU", r"$^{-1}$"]
        )
        + "".join([f" @ R = {mrr.max_S_radius:.0f} ", r"$\mu$", "m"])
        + "".join([r", $\Gamma$ = ", f"{R_max_Smrr_gamma:.0f}", r"$\%$"]),
    )
    ax.plot(
        [R_2D_map[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_gamma, R_max_Smrr_gamma],
        "w--",
    )
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        "g--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw), mrr.gamma_resampled * 100, "g", label=r"Rw$(\Gamma_{fluid})$"
    )
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
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
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, Snr_2D_map, cmap=colormap2D)
    ax.plot(np.log10(models.R), mrr.gamma, "r--", label=r"max$\{S_{MRR}\}$")
    ax.set_title(
        r"MRR $S_{NR}$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$")
    fig.colorbar(cm, label=r"$S_{NR}$ (RIU$^{-1}$)")
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
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
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, Se_2D_map, cmap=colormap2D)
    ax.plot(np.log10(models.R), mrr.gamma, "r--", label=r"max$\{S_{MRR}\}$")
    ax.set_title(
        r"MRR $S_e$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$")
    fig.colorbar(cm, label=r"$S_e$")
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
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
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, Se_times_a_2D_map, cmap=colormap2D)
    ax.plot(np.log10(models.R), mrr.gamma, "r--", label=r"max$\{S_{MRR}\}$")
    ax.set_title(
        r"MRR $S_e \times a$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$")
    fig.colorbar(cm, label=r"$S_e \times a$")
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
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
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, a2_2D_map, cmap=colormap2D)
    ax.plot(np.log10(models.R), mrr.gamma, "r--", label=r"max$\{S_{MRR}\}$")
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        "g--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw), mrr.gamma_resampled * 100, "g", label=r"Rw$(\Gamma_{fluid})$"
    )
    ax.set_title(
        r"MRR $a^2$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$")
    fig.colorbar(cm, label=r"$a^2$")
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
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
    cm = ax.pcolormesh(R_2D_map, gamma_2D_map, alpha_L_2D_map, cmap=colormap2D)
    ax.plot(np.log10(models.R), mrr.gamma, "r--", label=r"max$\{S_{MRR}\}$")
    ax.plot(
        np.log10(mrr.Re),
        mrr.gamma_resampled * 100,
        "g--",
        label=r"Re$(\Gamma_{fluid})$",
    )
    ax.plot(
        np.log10(mrr.Rw), mrr.gamma_resampled * 100, "g", label=r"Rw$(\Gamma_{fluid})$"
    )
    ax.set_title(
        r"MRR $\alpha L$ as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$")
    fig.colorbar(cm, label=r"$\alpha L$ (dB)")
    ax.set_xlim(left=np.log10(r_plot_min), right=np.log10(r_plot_max))
    ax.set_ylim(bottom=mrr.gamma_resampled[0] * 100, top=mrr.gamma_resampled[-1] * 100)
    ax.legend(loc="lower right")
    filename = (
        filename_path.parent
        / f"{filename_path.stem}_MRR_2DMAP_alphaL_VS_GAMMA_and_R.png"
    )
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # Save 2D map data as a function of gamma and R to output Excel file, if required
    if write_excel_files:
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

    # Plot spiral results, if required
    if not no_spiral:
        _plot_spiral_results(
            models=models,
            mrr=mrr,
            spiral=spiral,
            filename_path=filename_path,
            draw_largest_spiral=draw_largest_spiral,
            write_spiral_sequence_to_file=write_spiral_sequence_to_file,
            logger=logger,
        )

    #
    # Overlaid plots of linear, spiral and MRR waveguide sensitivity as function of R
    #

    # Calculate minimum sensitivity required to detect the minimum resolvable
    # change in ni for a given transmission measurement SNR
    S_min: float = 10 ** (-T_SNR / 10) / min_delta_ni

    # Plot...
    fig, ax = plt.subplots()
    ax.set_title(
        "Maximum sensitivity for MRR, spiral, and linear sensors"
        if not no_spiral
        else "Maximum sensitivity for MRR and linear sensors"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", {models.core_v_name} = {models.core_v_value:.3f} ", r"$\mu$m"])
    )

    # MRR
    ax.set_xlabel(r"Ring radius ($\mu$m)")
    ax.set_ylabel(r"Maximum sensitivity ($RIU^{-1}$)")
    ax.loglog(models.R, mrr.S, color="b", label="MRR")

    # Linear waveguide
    ax.loglog(models.R, linear.S, color="g", label=r"Linear waveguide ($L = 2R$)")
    ax.loglog(
        [r_plot_min, r_plot_max],
        [S_min, S_min],
        "r--",
        label="".join(
            [
                r"min$\{S\}$ to resolve $\Delta n_{i}$",
                f" = {min_delta_ni:.0E} @ SNR = {T_SNR:.0f} dB",
            ]
        ),
    )
    ax.set_xlim(r_plot_min, r_plot_max)
    ax.set_ylim(100, S_plot_max)

    # Spiral and MRR/spiral sensitivity ratio, if required
    if not no_spiral:
        ax.loglog(
            models.R[spiral.S > 1],
            spiral.S[spiral.S > 1],
            color="k",
            label=f"Spiral (spacing = {spiral.spacing:.0f}"
            + r" $\mu$m"
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
