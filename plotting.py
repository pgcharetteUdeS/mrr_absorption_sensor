#
# Plotting utilities
#
# Exposed methods:
#   - plot_results()
#
#

# Standard library packages
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Package modules
from .modeling import Models
from .mrr import Mrr
from .linear import Linear
from .spiral import Spiral
from .fileio import write_2D_data_to_Excel


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

    # h domain extrema
    h: np.ndarray = np.asarray(list(models.bending_loss_data.keys()))
    h_plot_min: float = h[0] * 0.9
    h_plot_max: float = h[-1] * 1.1

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
        h_plot_min,
        h_plot_max,
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


def _write_spiral_sequence_to_file(models: Models, spiral: Spiral, filename_path: Path):
    """
    Write sequence of consecutive spirals with n turns > spiral.n_turns_min
    """

    # Calculate spiral sequence looping indices (min, max, index)
    biggest_spiral_index: int = int(np.argmax(spiral.n_turns))
    index_min: int = int(
        np.argmax(spiral.n_turns[:biggest_spiral_index] > (spiral.turns_min * 1.001))
    )
    index_max: int = (
        int(
            np.argmax(
                spiral.n_turns[biggest_spiral_index:] < (spiral.turns_min * 1.001)
            )
        )
        + biggest_spiral_index
    )
    index_inc: int = int((index_max - index_min) / 25)

    # Loop to write generate the spiral images in the sequence
    fig, _ = plt.subplots(1, 1)
    images: list = []
    for index in range(index_min, index_max, index_inc):
        fig = spiral.draw_spiral(
            r_outer=models.R[index],
            h=spiral.h[index],
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
        h_plot_min,
        h_plot_max,
        S_plot_max,
        *_,
    ) = _calc_plotting_extrema(models=models, mrr=mrr)

    # Plot max{S}, h, gamma, n turns mas, L
    fig, axs = plt.subplots(5, 1)
    fig.suptitle(
        "Archimedes spiral "
        + f" ({models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m"])
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

    # h @ max{S}
    axs_index += 1
    axs[axs_index].set_ylabel(r"$h$ ($\mu$m)")
    axs[axs_index].semilogx(models.R, spiral.h)
    axs[axs_index].plot(
        [spiral.max_S_radius, spiral.max_S_radius], [h_plot_min, h_plot_max], "--"
    )
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(h_plot_min, h_plot_max)
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
            h=spiral.h[largest_spiral_index],
            n_turns=spiral.n_turns[largest_spiral_index],
            r_window=(models.R[largest_spiral_index] // 25 + 1) * 25,
        )
        filename = filename_path.parent / f"{filename_path.stem}_SPIRAL_SCHEMATIC.png"
        fig.savefig(fname=filename)
        logger(f"Wrote '{filename}'.")

    # Write sequence of consecutive spirals with n turns > spiral.n_turns_min
    if write_spiral_sequence_to_file:
        _write_spiral_sequence_to_file(
            models=models, spiral=spiral, filename_path=filename_path
        )


def plot_results(
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    T_SNR: float,
    min_delta_ni: float,
    filename_path: Path,
    write_excel_files: bool = False,
    no_spiral: bool = False,
    draw_largest_spiral: bool = False,
    write_spiral_sequence_to_file: bool = False,
    logger=print,
):

    # Calculate plotting extrema and max{S} vertical marker
    (
        r_plot_min,
        r_plot_max,
        h_plot_min,
        h_plot_max,
        S_plot_max,
        Se_plot_max,
        Finesse_plot_max,
        gamma_plot_min,
        gamma_plot_max,
    ) = _calc_plotting_extrema(models=models, mrr=mrr)

    #
    # Intermediate MRR results figures
    #

    # max{S}, S_NR, Se, a, h, gamma, Finesse
    fig, axs = plt.subplots(7, 1)
    fig.suptitle(
        "Micro-ring resonator "
        + f"({models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m", ")\n"])
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
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [100, S_plot_max], "--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(5000, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # S_NR @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.Snr)
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [10, S_plot_max], "--")
    axs[axs_index].set_ylabel(r"S$_{NR}$" + "\n" + r"(RIU $^{-1}$)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(10, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Se @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Se * np.sqrt(mrr.a2))
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, Se_plot_max], "--")
    axs[axs_index].set_ylabel(r"S$_e \times a$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, Se_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, np.sqrt(mrr.a2))
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [0, 1], "--")
    axs[axs_index].set_ylabel(r"$a$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, 1)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # h @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.h)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [h_plot_min, h_plot_max], "--"
    )
    axs[axs_index].set_ylabel(r"$h$ ($\mu$m)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(h_plot_min, h_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Gamma_fluid @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.gamma)
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [gamma_plot_min, gamma_plot_max], "--"
    )
    axs[axs_index].set_ylabel(r"$\Gamma_{fluide}$ ($\%$)")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(gamma_plot_min, gamma_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # Finesse/2pi (number of turns in the ring) @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(models.R, mrr.Finesse / (2 * np.pi))
    axs[axs_index].plot(
        [mrr.max_S_radius, mrr.max_S_radius], [0, Finesse_plot_max], "--"
    )
    axs[axs_index].set_ylabel(r"Finesse/$2\pi$")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(0, Finesse_plot_max)

    axs[axs_index].set_xlabel(r"Ring radius ($\mu$m)")
    filename: Path = filename_path.parent / (filename_path.stem + "_MRR.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    # max{S}, Finesse, FWHM, FSR
    fig, axs = plt.subplots(3, 1)
    fig.suptitle(
        "MRR - Ring resonator parameters"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m"])
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
    axs[axs_index].plot([mrr.max_S_radius, mrr.max_S_radius], [100, S_plot_max], "--")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylim(1000, S_plot_max)
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # (Finesse/2pi) / Se*a @ max{S}
    axs_index += 1
    axs[axs_index].semilogx(
        models.R,
        mrr.Finesse / (2 * np.pi) / (mrr.Se * np.sqrt(mrr.a2)),
        "b",
        label="FWHM",
    )
    axs[axs_index].set_ylim(0, 2.5)
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel(r"$\frac{Finesse/2\pi}{S_e\times a}$")
    axs[axs_index].axes.get_xaxis().set_ticklabels([])

    # FWHM, FSR @ max{S}
    axs_index += 1
    axs[axs_index].loglog(models.R, mrr.FWHM * 1e6, "b", label="FWHM")
    axs[axs_index].loglog(models.R, mrr.FSR * 1e6, "g", label="FSR")
    axs[axs_index].set_xlim(r_plot_min, r_plot_max)
    axs[axs_index].set_ylabel("FWHM and FSR (pm)")
    axs[axs_index].set_xlabel(r"Ring radius ($\mu$m)")

    # Finesse @ max{S}
    axR = axs[axs_index].twinx()
    axR.semilogx(models.R, mrr.Finesse, "k--", label="Finesse")
    axR.set_ylabel(r"Finesse")
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
    filename = filename_path.parent / (filename_path.stem + "_MRR_ring_parms.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")

    #
    # Figures 6 and 6b: 2D maps of sensitivity as a functions of h, gamma, and R
    #

    # Generate 2D map data
    n_grid_points: int = 500
    R_fig_6 = np.linspace(np.log10(models.R[0]), np.log10(models.R[-1]), n_grid_points)
    h_fig_6 = np.linspace(
        list(models.bending_loss_data)[0],
        list(models.bending_loss_data)[-1],
        n_grid_points,
    )
    S_fig_6 = np.asarray(
        [
            [mrr.calc_sensitivity(r=10 ** log10_R, h=hh)[0] for log10_R in R_fig_6]
            for hh in h_fig_6
        ]
    )
    gamma_fig_6 = np.asarray([models.gamma(hh) * 100 for hh in h_fig_6])

    # Indices for dashed lines at radii for S_mrr = 5X and 10X S_spiral, and max(Smrr)
    R_max_Smrr_index: int = int((np.abs(models.R - mrr.max_S_radius)).argmin())
    R_max_Smrr_h: float = mrr.h[R_max_Smrr_index]
    R_max_Smrr_gamma: float = mrr.gamma[R_max_Smrr_index]

    # Fig.6b: plot 2D map of S(h, R)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(R_fig_6, h_fig_6, S_fig_6)
    ax.plot(np.log10(models.R), mrr.h, label=r"max$\{S(h, R)\}$")
    ax.invert_yaxis()
    ax.set_title(
        r"Fig.6b : MRR sensitivity as a function of $h$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$h$ ($\mu$m)")
    ax.legend(loc="upper right")
    fig.colorbar(cm, label=r"S (RIU $^{-1}$)")

    # Plot dashed lines at radii for S_mrr = 5X and 10X S_spiral, and max(Smrr)
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [h_fig_6[-1], R_max_Smrr_h],
        "w--",
        label="".join([r"$max\{S_{MRR}\}$", f" = {mrr.max_S:.0f} RIU", r"$^{-1}$"])
        + "".join([f" @ R = {mrr.max_S_radius:.0f} ", r"$\mu$", "m"])
        + "".join([f", h = {R_max_Smrr_h:.3f} ", r"$\mu$m"]),
    )
    ax.plot(
        [R_fig_6[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_h, R_max_Smrr_h],
        "w--",
    )
    if not no_spiral:
        R_5X_index, R_5X, R_5X_S, R_10X_index, R_10X, R_10X_S = _calc_5X_10X_comp_data(
            R=models.R, R_max_Smrr_index=R_max_Smrr_index, mrr=mrr, spiral=spiral
        )
        R_5X_h: float = mrr.h[R_5X_index]
        R_10X_h: float = mrr.h[R_10X_index]
        ax.plot(
            [np.log10(R_5X), np.log10(R_5X)],
            [h_fig_6[-1], R_5X_h],
            "k--",
            label="".join([r"$S_{MRR} = 5\times S_{SPIRAL}$"])
            + "".join([r"$, S_{MRR}/max\{S_{MRR}\} = $", f"{R_5X_S / mrr.max_S:.2f}"])
            + "".join([f" @ R = {R_5X:.0f} ", r"$\mu$", "m"])
            + "".join([f", h = {R_5X_h:.3f} ", r"$\mu$m"]),
        )
        ax.plot([R_fig_6[0], np.log10(R_5X)], [R_5X_h, R_5X_h], "k--")
        ax.plot(
            [np.log10(R_10X), np.log10(R_10X)],
            [h_fig_6[-1], R_10X_h],
            "r--",
            label="".join([r"$S_{MRR} = 10\times S_{SPIRAL}$"])
            + "".join([r"$, S_{MRR}/max\{S_{MRR}\} = $", f"{R_10X_S / mrr.max_S:.2f}"])
            + "".join([f" @ R = {R_10X:.0f} ", r"$\mu$", "m"])
            + "".join([f", h = {R_10X_h:.3f} ", r"$\mu$m"]),
        )
        ax.plot([R_fig_6[0], np.log10(R_10X)], [R_10X_h, R_10X_h], "r--")
        ax.legend(loc="lower right")

    # Save Figure 6b to .png file and data to Excel file
    filename = filename_path.parent / (filename_path.stem + "_FIG6b.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")
    if write_excel_files:
        write_2D_data_to_Excel(
            filename=str(filename.with_suffix(".xlsx")),
            X=10 ** R_fig_6,
            Y=h_fig_6,
            S=S_fig_6,
            y_label="h",
        )
        logger(f"Wrote '{filename.with_suffix('.xlsx')}'.")

    # Fig.6: plot 2D map of S(gamma, R)
    fig, ax = plt.subplots()
    cm = ax.pcolormesh(R_fig_6, gamma_fig_6, S_fig_6)
    ax.plot(np.log10(models.R), mrr.gamma, label=r"max$\{S(\Gamma_{fluid}, R)\}$")
    ax.set_title(
        r"Fig.6 : MRR sensitivity as a function of $\Gamma_{fluid}$ and $R$"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"log(R) ($\mu$m)")
    ax.set_ylabel(r"$\Gamma_{fluid}$ ($\%$)")
    fig.colorbar(cm, label=r"S (RIU $^{-1}$)")

    # Plot dashed lines at radii for S_mrr = 5X and 10X S_spiral, and max(Smrr)
    ax.plot(
        [np.log10(mrr.max_S_radius), np.log10(mrr.max_S_radius)],
        [gamma_fig_6[-1], R_max_Smrr_gamma],
        "w--",
        label="".join([r"$max\{S_{MRR}\}$", f" = {mrr.max_S:.0f} RIU", r"$^{-1}$"])
        + "".join([f" @ R = {mrr.max_S_radius:.0f} ", r"$\mu$", "m"])
        + "".join([r", $\Gamma$ = ", f"{R_max_Smrr_gamma:.0f}", r"$\%$"]),
    )
    ax.plot(
        [R_fig_6[0], np.log10(mrr.max_S_radius)],
        [R_max_Smrr_gamma, R_max_Smrr_gamma],
        "w--",
    )
    if not no_spiral:
        R_5X_index, R_5X, R_5X_S, R_10X_index, R_10X, R_10X_S = _calc_5X_10X_comp_data(
            R=models.R, R_max_Smrr_index=R_max_Smrr_index, mrr=mrr, spiral=spiral
        )
        R_5X_gamma: float = mrr.gamma[R_5X_index]
        R_10X_gamma: float = mrr.gamma[R_10X_index]
        ax.plot(
            [np.log10(R_5X), np.log10(R_5X)],
            [gamma_fig_6[-1], R_5X_gamma],
            "k--",
            label="".join([r"$S_{MRR} = 5\times S_{SPIRAL}$"])
            + "".join([r"$, S_{MRR}/max\{S_{MRR}\} = $", f"{R_5X_S / mrr.max_S:.2f}"])
            + "".join([f" @ R = {R_5X:.0f} ", r"$\mu$", "m"])
            + "".join([r", $\Gamma$ = ", f"{R_5X_gamma:.0f}", r"$\%$"]),
        )
        ax.plot([R_fig_6[0], np.log10(R_5X)], [R_5X_gamma, R_5X_gamma], "k--")
        ax.plot(
            [np.log10(R_10X), np.log10(R_10X)],
            [gamma_fig_6[-1], R_10X_gamma],
            "r--",
            label="".join([r"$S_{MRR} = 10\times S_{SPIRAL}$"])
            + "".join([r"$, S_{MRR}/max\{S_{MRR}\} = $", f"{R_10X_S / mrr.max_S:.2f}"])
            + "".join([f" @ R = {R_10X:.0f} ", r"$\mu$", "m"])
            + "".join([r", $\Gamma$ = ", f"{R_10X_gamma:.0f}", r"$\%$"]),
        )
        ax.plot([R_fig_6[0], np.log10(R_10X)], [R_10X_gamma, R_10X_gamma], "r--")
    ax.legend(loc="lower right")

    # Save figure 6 to .png file and data to Excel file ** LONG **
    filename = filename_path.parent / (filename_path.stem + "_FIG6.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")
    if write_excel_files:
        write_2D_data_to_Excel(
            filename=str(filename.with_suffix(".xlsx")),
            X=10 ** R_fig_6,
            Y=h_fig_6,
            S=S_fig_6,
            y_label="h",
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
    # Fig.7: Overlay plots of linear, spiral and MRR waveguide sensitivity(R)
    #

    # Calculate minimum sensitivity required to detect the minimum resolvable
    # change in ni for a given transmission measurement SNR
    S_min: float = 10 ** (-T_SNR / 10) / min_delta_ni

    # Plot...
    fig, ax = plt.subplots()
    ax.set_title(
        "Fig.7 : Maximum sensitivity for MRR, spiral, and linear sensors"
        + f"\n{models.pol}"
        + "".join([r", $\lambda$", f" = {models.lambda_res:.3f} ", r"$\mu$m"])
        + "".join([r", $\alpha_{wg}$", f" = {models.alpha_wg_dB_per_cm:.1f} dB/cm"])
        + "".join([f", w = {models.core_width:.3f} ", r"$\mu$m"])
    )
    ax.set_xlabel(r"Ring radius ($\mu$m)")
    ax.set_ylabel(r"Maximum sensitivity ($RIU^{-1}$)")
    ax.loglog(models.R, mrr.S, color="b", label="MRR")
    ax.loglog(
        models.R[spiral.S > 1],
        spiral.S[spiral.S > 1],
        color="k",
        label=f"Spiral (spacing = {spiral.spacing:.0f}"
        + r" $\mu$m"
        + f", min turns = {spiral.turns_min:.2f})",
    )
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
    ax_lines = ax.get_legend_handles_labels()[0] + axR.get_legend_handles_labels()[0]
    ax_labels = ax.get_legend_handles_labels()[1] + axR.get_legend_handles_labels()[1]
    ax.legend(ax_lines, ax_labels, loc="lower right")
    ax.patch.set_visible(False)
    axR.patch.set_visible(True)
    ax.set_zorder(axR.get_zorder() + 1)

    # Save figure
    filename = filename_path.parent / (filename_path.stem + "_FIG7.png")
    fig.savefig(filename)
    logger(f"Wrote '{filename}'.")
