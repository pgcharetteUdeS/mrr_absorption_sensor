"""

Plot figures in the article

Data in files "*_MRR_2DMAPS_VS_GAMMA_and_R.xlsx" and "*_ALL_RESULTS.xlsx"

"""

import io
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook, Workbook


def _determine_y_plotting_extrema(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    results_file_mrr_sheet_col_names: dict,
) -> tuple[float, float]:
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        results_file_mrr_sheet_col_names ():

    Returns:
        y_max_s
        y_max_αl

    """

    # Calculate y axis limit for αl plots
    αl_max = max(
        max(cell.value for cell in row)
        for row in wb_2d_map["alpha x L (dB)"].iter_rows(min_row=2)
    )
    y_max_αl = np.ceil(αl_max / 50) * 50

    # Calculate y axis limit for sensitivity plots
    max_s_max = max(
        np.asarray(
            [
                val[0].value
                for val in wb_all_results["MRR"].iter_rows(
                    min_row=2,
                    min_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
                    max_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
                )
            ]
        )
    )
    y_max_s: float = (
        np.ceil(max_s_max / 50000) * 50000
        if max_s_max < 100000
        else np.ceil(max_s_max / 500000) * 500000
    )

    # Return y extrema
    return y_max_s, y_max_αl


def _get_re_rw(wb_all_results: Workbook, gamma: float) -> tuple[float, float]:
    """

    Args:
        wb_all_results ():
        gamma ():

    Returns:

    """
    gammas = np.asarray(
        [
            val[0].value
            for val in wb_all_results["Re and Rw"].iter_rows(min_row=2, max_col=1)
        ]
    )
    index = int(np.argmin(np.abs(gammas - gamma)))
    re, rw = np.asarray(
        [
            (val[0].value, val[1].value)
            for val in wb_all_results["Re and Rw"].iter_rows(
                min_row=2, min_col=3, max_col=4
            )
        ]
    ).T

    return re[index], rw[index]


def figure_3b(
    wb_all_results: Workbook,
    out_filename_path: Path,
):
    """

    Args:
        wb_all_results ():
        out_filename_path ():

    Returns:

    """

    # fetch u and alpha_wg(u) data
    u, alpha_wg = np.asarray(
        [
            (val[0].value, val[1].value)
            for val in wb_all_results["alpha_wg_interp"].iter_rows(
                min_row=2, min_col=1, max_col=2
            )
        ]
    ).T

    # Create the figure
    fig, axs = plt.subplots()
    fig.suptitle(f"Figure 3b ('{out_filename_path.stem}*')")
    axs.plot(u, alpha_wg)
    axs.set_ylim(bottom=0)
    if wb_all_results["alpha_wg"]["A1"].value == "height_um":
        axs.set_xlabel("Height (μm)")
    else:
        axs.set_xlabel("Width (μm)")
    axs.set_ylabel(r"$\alpha_{wg}$ (dB/cm)")

    # Save figure to file
    fig.savefig(out_filename_path.parent / f"{out_filename_path.stem}_FIG3b.png")
    print(
        f"Wrote '{str(out_filename_path.parent)}/{out_filename_path.stem}_FIG3b.png'."
    )


def figure_4(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    gamma: float,
    out_filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        gamma ():
        out_filename_path: Path,

    Returns:

    """
    # Fetch R and gamma arrays from their respective sheets in the 2D map workbook
    r = np.asarray([c.value for c in wb_2d_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2d_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Fetch line profiles for S, Snr, and Se at gamma from the the 2D map workbook
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)
    s = np.asarray([c.value for c in wb_2d_map["S (RIU-1)"][index]])
    s_nr = np.asarray([c.value for c in wb_2d_map["Snr (RIU-1)"][index]])
    s_e = np.asarray([c.value for c in wb_2d_map["Se"][index]])

    # Get corresponding Re and Rw values from the all_results workbook
    re, rw = _get_re_rw(
        wb_all_results=wb_all_results,
        gamma=gamma,
    )

    # Create the figure
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f"Figure 4 ('{out_filename_path.stem}*')")

    # PLot line profiles
    ax.semilogx(r, s, "b-", label=r"$S_{MRR}$")
    ax.semilogx(r, s_nr, "r--", label=r"$S_{NR}$")
    index = int(np.argmin(np.abs(r - rw)) + 1)
    ax.semilogx([rw, rw], [0, s_nr[index]], "k")
    ax.text(rw * 0.95, s_nr[index] * 1.05, "RW")
    ax_r = ax.twinx()
    ax_r.semilogx(r, s_e, "g--", label=r"$S_{e}$")
    index = int(np.argmin(np.abs(r - re)) + 1)
    ax_r.semilogx([re, re], [0, s_e[index]], "k")
    ax_r.text(re * 0.95, s_e[index] * 1.05, "Re")

    # PLot formatting, title, labels, etc.
    y_max: float = (np.ceil((max(s) / 5000)) + 1) * 5000
    ax.set_title(
        "Sensitivity components as a function of radius"
        + rf" at $\Gamma_{{fluid}}$ = {gamma:.0f}$\%$"
    )
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel(r"$S_{MRR}$ and $S_{NR}$ (RIU$_{-1}$)")
    ax_r.set_ylabel(r"$S_e$")
    ax.set_ylim(0, y_max)
    ax_r.set_ylim(0, y_max / 1000)
    ax.set_xlim(r[0], r[-1])

    # Combine legends
    ax_lines = ax.get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
    ax_labels = ax.get_legend_handles_labels()[1] + ax_r.get_legend_handles_labels()[1]
    ax.legend(ax_lines, ax_labels, loc="upper left")

    # Save figure to file
    fig.savefig(out_filename_path.parent / f"{out_filename_path.stem}_FIG4.png")
    print(f"Wrote '{str(out_filename_path.parent)}/{out_filename_path.stem}_FIG4.png'.")


def _figure_6_line_profile_plot(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    gamma: float,
    r: np.ndarray,
    gammas: np.ndarray,
    ax: plt.Axes,
    y_max_s: float,
    y_max_αl: float,
    last: bool,
):
    # Calculate the row index in the sheets for the requested gamma value
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)

    # Fetch line profiles data from the worksheets
    α_l = np.asarray([c.value for c in wb_2d_map["alpha x L (dB)"][index]])
    α_bend_l = np.asarray([c.value for c in wb_2d_map["alpha_bend x L (dB)"][index]])
    α_prop_l = np.asarray([c.value for c in wb_2d_map["alpha_prop x L (dB)"][index]])
    s = np.asarray([c.value for c in wb_2d_map["S (RIU-1)"][index]])

    # Get corresponding Re and Rw values from the all_results workbook
    re, rw = _get_re_rw(
        wb_all_results=wb_all_results,
        gamma=gamma,
    )

    # Plot the line profile
    ax.semilogx(r, α_l, "k", label="αL")
    ax.semilogx(r, α_bend_l, "g--", label="αbendL")
    ax.semilogx(r, α_prop_l, "r--", label="αpropL")
    ax.semilogx([rw, rw], [0, 50], "k")
    ax.text(rw * 1.05, y_max_αl * 1.025, "Rw")
    ax.semilogx([re, re], [0, 50], "k")
    ax.text(re * 0.85, y_max_αl * 1.025, "Re")
    ax_r = ax.twinx()
    ax_r.semilogx(r, s, "b", label=r"$S_{MRR}$")

    # PLot formatting, title, labels, etc.
    ax.set_title(rf"$\Gamma_{{fluid}}$ = {gamma:.0f} $\%$")
    ax.set_ylabel("Roundtrip losses (dB)")
    ax_r.set_ylabel(r"$S_{MRR}$ (RIU$^{-1}$)")
    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(0, y_max_αl)
    ax_r.set_ylim(0, y_max_s)
    if last:
        ax.set_xlabel("Radius (μm)")
    else:
        ax.axes.get_xaxis().set_ticklabels([])

    # Combine legends
    ax_lines = ax.get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
    ax_labels = ax.get_legend_handles_labels()[1] + ax_r.get_legend_handles_labels()[1]
    ax.legend(ax_lines, ax_labels, loc="upper left")


def figure_6(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    line_profile_gammas: np.ndarray,
    y_max_s: float,
    y_max_αl: float,
    out_filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        line_profile_gammas ():
        y_max_s ():
        y_max_αl ():
        out_filename_path: Path,

    Returns:

    """

    # Fetch R and gamma arrays from their respective sheets in the workbook
    r = np.asarray([c.value for c in wb_2d_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2d_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Create figure with required number of subplots for the requested line profiles
    fig, axs = plt.subplots(
        nrows=len(line_profile_gammas), figsize=(9, 10), constrained_layout=True
    )
    fig.suptitle(f"Figure 6 ('{out_filename_path.stem}*')")

    # Loop to generate the subplots of the line profiles in "line_profile_gammas"
    for i, gamma in enumerate(line_profile_gammas):
        _figure_6_line_profile_plot(
            wb_2d_map=wb_2d_map,
            wb_all_results=wb_all_results,
            gamma=gamma,
            r=r,
            gammas=gammas,
            ax=axs[i],
            y_max_s=y_max_s,
            y_max_αl=y_max_αl,
            last=i >= len(line_profile_gammas) - 1,
        )

    # Save figure to file
    fig.savefig(out_filename_path.parent / f"{out_filename_path.stem}_FIG6.png")
    print(f"Wrote '{str(out_filename_path.parent)}/{out_filename_path.stem}_FIG6.png'.")


def figure_7(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    results_file_mrr_sheet_col_names: dict,
    line_profile_gammas: np.ndarray,
    y_max_s: float,
    out_filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        results_file_mrr_sheet_col_names ():
        line_profile_gammas ():
        y_max_s ():
        out_filename_path ():

    Returns:

    """

    # Create the figure
    fig, axs = plt.subplots(4, figsize=(9, 12))
    fig.suptitle(f"Figure 7 ('{out_filename_path.stem}*')")

    #
    # 6a
    #

    # Fetch R and gamma arrays from their respective sheets in the workbook
    r = np.asarray([c.value for c in wb_2d_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2d_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Loop to generate the subplots of the line profiles in "line_profile_gammas"
    for g in line_profile_gammas:
        index = int(np.argmin(np.abs(gammas - g)) + 1)
        s = np.asarray([c.value for c in wb_2d_map["S (RIU-1)"][index]])
        axs[0].semilogx(r, s, label=rf"$\Gamma$ = {g:.0f}%")

    # PLot formatting, title, labels, etc.
    axs[0].set_ylabel(r"S$_{MRR}$")
    axs[0].set_xlim(r[0], r[-1])
    axs[0].set_ylim(0, y_max_s)
    axs[0].legend(loc="upper left")

    #
    # 6b & 6c
    #
    r = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2, max_col=results_file_mrr_sheet_col_names["R_um"]
            )
        ]
    )
    s_max = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
                max_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
            )
        ]
    )
    α_bend = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["alpha_bend_dB_per_cm"],
                max_col=results_file_mrr_sheet_col_names["alpha_bend_dB_per_cm"],
            )
        ]
    )
    α_wg = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["alpha_wg_dB_per_cm"],
                max_col=results_file_mrr_sheet_col_names["alpha_wg_dB_per_cm"],
            )
        ]
    )
    h = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["h_um"],
                max_col=results_file_mrr_sheet_col_names["h_um"],
            )
        ]
    )
    gamma = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["gamma_percent"],
                max_col=results_file_mrr_sheet_col_names["gamma_percent"],
            )
        ]
    )
    max_s_max_r: float = r[int(np.argmax(s_max))]

    # Add max(max(Smax)) vertical line to 6a
    axs[0].semilogx([max_s_max_r, max_s_max_r], [0, y_max_s], "r--")
    axs[0].text(max_s_max_r * 1.05, y_max_s, r"max(max($S_{MRR}$))", color="red")
    axs[0].axes.get_xaxis().set_ticklabels([])

    # Smax(R)
    axs[1].loglog(r, s_max, "k")
    axs[1].set_ylabel("max(S$_{MRR}$)")
    axs[1].set_xlim(r[0], r[-1])
    axs[1].set_ylim(10, y_max_s)
    axs[1].loglog([max_s_max_r, max_s_max_r], [10, y_max_s], "r--")
    axs[1].text(max_s_max_r * 1.05, y_max_s, r"max(max($S_{MRR}$))", color="red")
    axs[1].axes.get_xaxis().set_ticklabels([])

    # h(R) and gamma(R) @ Smax
    y_min_h: float = 0.1
    y_max_h: float = 0.9
    axs[2].semilogx(r, h, "b", label="h")
    axs[2].set_ylabel(r"h (μm) @ max($S_{MRR}$)")
    axs[2].set_xlim(r[0], r[-1])
    axs[2].set_ylim(y_min_h, y_max_h)
    axs[2].semilogx([max_s_max_r, max_s_max_r], [y_min_h, y_max_h], "r--")
    ax_r = axs[2].twinx()
    ax_r.semilogx(r, gamma, "g--", label=r"$\Gamma_{fluid}$")
    ax_r.set_ylabel(r"$\Gamma_{fluid}$ $(\%)$ @ max($S_{MRR}$)")
    ax_r.set_ylim(0, 80)
    ax_lines = (
        axs[2].get_legend_handles_labels()[0] + ax_r.get_legend_handles_labels()[0]
    )
    ax_labels = (
        axs[2].get_legend_handles_labels()[1] + ax_r.get_legend_handles_labels()[1]
    )
    axs[2].legend(ax_lines, ax_labels, loc="upper left")
    axs[2].text(max_s_max_r * 1.05, 0.45, r"max(max($S_{MRR}$))", color="red")
    axs[2].axes.get_xaxis().set_ticklabels([])

    # α_bend & α_wg
    axs[3].semilogx(r, α_bend, label=r"α$_{bend}$")
    axs[3].semilogx(r, α_wg, label=r"α$_{wg}$")
    axs[3].set_ylabel(r"α$_{bend}$ and α$_{wg}$ (dB/cm)")
    axs[3].set_xlim(r[0], r[-1])
    axs[3].set_ylim(0, 5)
    axs[3].legend(loc="upper right")

    # Bottom horizontal axis labels
    axs[3].set_xlabel("Radius (μm)")

    # Save figure to file
    fig.savefig(out_filename_path.parent / f"{out_filename_path.stem}_FIG7.png")
    print(f"Wrote '{str(out_filename_path.parent)}/{out_filename_path.stem}_FIG7.png'.")


def figure_x(
    wb_all_results: Workbook,
    results_file_mrr_sheet_col_names: dict,
    out_filename_path: Path,
):
    """

    Args:
        wb_all_results ():
        results_file_mrr_sheet_col_names ():
        out_filename_path ():

    Returns:

    """
    # fetch radius data
    r_linear = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["R_um"],
                max_col=results_file_mrr_sheet_col_names["R_um"],
            )
        ]
    )

    # Generate log-sampled array of indices for r values
    r_samples_per_decade: float = 1
    r_sampled: np.ndarray = 10 ** (
        np.arange(
            np.log10(r_linear[0]),
            np.log10(r_linear[-1]) + 1 / r_samples_per_decade,
            1 / r_samples_per_decade,
        )
    )
    r_sampled_indices: list = [np.where(r_linear >= r)[0][0] for r in r_sampled]
    r_sampled_min: float = r_linear[r_sampled_indices[0]]

    # fetch SMRR data
    s_max: np.ndarray = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
                max_col=results_file_mrr_sheet_col_names["maxS_RIU_inv"],
            )
        ]
    )
    smax_sampled: np.ndarray = s_max[r_sampled_indices]

    # Create the figure
    fig, axs = plt.subplots(5, figsize=(9, 12))
    fig.suptitle(f"Figure X ('{out_filename_path.stem}*')")

    #
    # Figure α_bend
    #
    α_bend: np.ndarray = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["alpha_dB_per_cm"],
                max_col=results_file_mrr_sheet_col_names["alpha_dB_per_cm"],
            )
        ]
    )
    axs[0].plot(s_max, α_bend)
    axs[0].plot(
        smax_sampled, α_bend[r_sampled_indices], "o", label=r"R ($\mu$m, powers of 10)"
    )
    axs[0].plot(
        smax_sampled[0],
        α_bend[r_sampled_indices[0]],
        color="green",
        marker="o",
        label=f"Rmin = {r_sampled_min:.1f} " + r"$\mu$m",
    )
    axs[0].set_ylabel(r"${\alpha}$ (R)")
    axs[0].axes.xaxis.set_ticklabels([])
    axs[0].set_ylim(bottom=0)
    axs[0].legend()

    #
    # Figure 1/α_bend
    #
    axs[1].plot(s_max, 1 / α_bend)
    axs[1].plot(
        smax_sampled,
        1 / α_bend[r_sampled_indices],
        "o",
        label=r"R ($\mu$m, powers of 10)",
    )
    axs[1].plot(
        smax_sampled[0],
        1 / α_bend[r_sampled_indices[0]],
        color="green",
        marker="o",
        label=f"Rmin = {r_sampled_min:.1f} " + r"$\mu$m",
    )
    axs[1].set_ylabel(r"1/${\alpha}$ (R)")
    axs[1].axes.xaxis.set_ticklabels([])
    axs[1].set_ylim(bottom=0)
    axs[1].legend()

    #
    # Figure αl (dB)
    #
    αl: np.ndarray = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["alphaL"],
                max_col=results_file_mrr_sheet_col_names["alphaL"],
            )
        ]
    ) * (10 * np.log10(np.e))
    axs[2].plot(s_max, αl)
    axs[2].plot(
        smax_sampled, αl[r_sampled_indices], "o", label=r"R ($\mu$m, powers of 10)"
    )
    axs[2].plot(
        smax_sampled[0],
        αl[r_sampled_indices[0]],
        color="green",
        marker="o",
        label=f"Rmin = {r_sampled_min:.1f} " + r"$\mu$m",
    )
    axs[2].set_ylabel(r"${\alpha}$L (R) (dB)")
    axs[2].axes.xaxis.set_ticklabels([])
    axs[2].set_ylim(bottom=0)
    axs[2].legend()

    #
    # Figure 1/αl
    #
    axs[3].plot(s_max, 1 / αl)
    axs[3].plot(
        smax_sampled, 1 / αl[r_sampled_indices], "o", label=r"R ($\mu$m, powers of 10)"
    )
    axs[3].plot(
        smax_sampled[0],
        1 / αl[r_sampled_indices[0]],
        color="green",
        marker="o",
        label=f"Rmin = {r_sampled_min:.1f} " + r"$\mu$m",
    )
    axs[3].set_ylabel(r"1/${\alpha}$L (R)")
    axs[3].axes.xaxis.set_ticklabels([])
    axs[3].set_ylim(bottom=0)
    axs[3].legend()

    #
    # Figure a2
    #
    a2: np.ndarray = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2,
                min_col=results_file_mrr_sheet_col_names["a2"],
                max_col=results_file_mrr_sheet_col_names["a2"],
            )
        ]
    )
    axs[4].plot(s_max, a2)
    axs[4].plot(
        smax_sampled, a2[r_sampled_indices], "o", label=r"R ($\mu$m, powers of 10)"
    )
    axs[4].plot(
        smax_sampled[0],
        a2[r_sampled_indices[0]],
        color="green",
        marker="o",
        label=f"Rmin = {r_sampled_min:.1f} " + r"$\mu$m",
    )
    axs[4].set_xlabel(r"S$_{MRR}$(R)")
    axs[4].set_ylabel(r"$a^2$ (R)")
    axs[4].set_ylim(bottom=0)
    axs[4].legend()

    # Save figure to file
    fig.savefig(out_filename_path.parent / f"{out_filename_path.stem}_FIGX.png")
    print(f"Wrote '{str(out_filename_path.parent)}/{out_filename_path.stem}_FIGX.png'.")


def plot_article_figures(results_file_name: str, maps_file_name: str):
    """

    Returns:

    """

    # matplotlib initializations
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.grid.which": "both",
        },
    )
    plt.ion()

    # Load Excel workbooks (into memory, to reduce risk of conflict)
    with open(results_file_name, "rb") as f:
        in_mem_results_file = io.BytesIO(f.read())
    with open(maps_file_name, "rb") as f:
        in_mem_maps_file = io.BytesIO(f.read())
    wb_all_results: Workbook = load_workbook(in_mem_results_file, read_only=True)
    wb_2d_map: Workbook = load_workbook(in_mem_maps_file, read_only=True)
    out_filename_path: Path = Path(results_file_name)

    # Create dictionary of column names for the MRR worksheet in the Excel results file
    results_file_mrr_sheet_col_names: dict = {
        col.value: index for index, col in enumerate(wb_all_results["MRR"][1], start=1)
    }

    # User-defined discrete gamma value array for figure 5
    line_profile_gammas_fig_5: np.ndarray = np.asarray([20, 45, 65, 75])

    # Generate discrete gamma value array for figure 6
    gamma = np.asarray(
        [val[0].value for val in wb_2d_map["gamma (%)"].iter_rows(max_col=1)]
    )
    line_profile_gammas_fig_6: np.ndarray = np.arange(
        np.ceil(gamma[-1] / 10) * 10, np.floor(gamma[0] / 10) * 10 + 10, 10
    )
    if int(gamma[-1]) != line_profile_gammas_fig_6[0]:
        line_profile_gammas_fig_6 = np.insert(
            line_profile_gammas_fig_6, 0, int(gamma[-1])
        )
    if int(gamma[0]) != line_profile_gammas_fig_6[-1]:
        line_profile_gammas_fig_6 = np.append(line_profile_gammas_fig_6, int(gamma[0]))

    # Generate figures
    y_max_s, y_max_αl = _determine_y_plotting_extrema(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        results_file_mrr_sheet_col_names=results_file_mrr_sheet_col_names,
    )
    figure_3b(
        wb_all_results=wb_all_results, out_filename_path=out_filename_path
    )
    figure_4(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        gamma=30,
        out_filename_path=out_filename_path,
    )
    figure_6(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        line_profile_gammas=line_profile_gammas_fig_5,
        y_max_s=y_max_s,
        y_max_αl=y_max_αl,
        out_filename_path=out_filename_path,
    )
    figure_7(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        results_file_mrr_sheet_col_names=results_file_mrr_sheet_col_names,
        line_profile_gammas=line_profile_gammas_fig_6,
        y_max_s=y_max_s,
        out_filename_path=out_filename_path,
    )
    figure_x(
        wb_all_results=wb_all_results,
        results_file_mrr_sheet_col_names=results_file_mrr_sheet_col_names,
        out_filename_path=out_filename_path,
    )
    plt.show()


def plot_article_figures_wrapper():
    """
    Wrapper function for calling plot_article_figures() with command line arguments

    Returns: N/A

    """

    # Parser for command line parameter input (ex: running from .bat file)
    parser = argparse.ArgumentParser(description="Plot article figures")
    parser.add_argument("--results_file", type=str)
    parser.add_argument("--maps_file", type=str)
    parser.add_argument("--no_pause", action="store_true")
    args = parser.parse_args()

    # Plot article figures
    plot_article_figures(
        results_file_name=args.results_file, maps_file_name=args.maps_file
    )

    # If running from the command line, either pause for user input to keep figures
    # visible. If running in PyCharm debugger, set breakpoint here.
    if sys.gettrace() is not None:
        print("Breakpoint here to keep figures visible in IDE!")
    elif not args.no_pause:
        input("Script paused to display figures, press any key to exit")
    else:
        print("Done!")


# Run the script...
if __name__ == "__main__":
    plot_article_figures_wrapper()
