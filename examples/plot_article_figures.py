"""

Plot figures 3, 5, 6 in the article

Data in files "*_MRR_2DMAPS_VS_GAMMA_and_R.xlsx" and "*_ALL_RESULTS.xlsx"

"""

import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook
from pathlib import Path
import sys


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


def figure_3(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    gamma: float,
    filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        gamma ():
        filename_path: Path,

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
    re, rw = _get_re_rw(wb_all_results=wb_all_results, gamma=gamma)

    # Create the figure
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f"Figure 3 ('{filename_path.stem}*')")

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
    fig.savefig(filename_path.parent / f"{filename_path.stem}_FIG3.png")


def _figure_5_line_profile_plot(
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
    # Calcule the row index in the sheets for the requested gamma value
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)

    # Fetch line profiles data from the worksheets
    α_l = np.asarray([c.value for c in wb_2d_map["alpha x L (dB)"][index]])
    α_bend_l = np.asarray([c.value for c in wb_2d_map["alpha_bend x L (dB)"][index]])
    α_prop_l = np.asarray([c.value for c in wb_2d_map["alpha_prop x L (dB)"][index]])
    s = np.asarray([c.value for c in wb_2d_map["S (RIU-1)"][index]])

    # Get corresponding Re and Rw values from the all_results workbook
    re, rw = _get_re_rw(wb_all_results=wb_all_results, gamma=gamma)

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


def figure_5(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    line_profile_gammas: np.ndarray,
    y_max_s: float,
    y_max_αl: float,
    filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        line_profile_gammas ():
        y_max_s ():
        y_max_αl ():
        filename_path: Path,

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
    fig.suptitle(f"Figure 5 ('{filename_path.stem}*')")

    # Loop to generate the subplots of the line profiles in "line_profile_gammas"
    for i, gamma in enumerate(line_profile_gammas):
        _figure_5_line_profile_plot(
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
    fig.savefig(filename_path.parent / f"{filename_path.stem}_FIG5.png")


def figure_6(
    wb_2d_map: Workbook,
    wb_all_results: Workbook,
    line_profile_gammas: np.ndarray,
    y_max_s: float,
    filename_path: Path,
):
    """

    Args:
        wb_2d_map ():
        wb_all_results ():
        line_profile_gammas ():
        y_max_s ():
        filename_path ():

    Returns:

    """

    # Create the figure
    fig, axs = plt.subplots(4, figsize=(9, 12))
    fig.suptitle(f"Figure 6 ('{filename_path.stem}*')")

    #
    # 6a
    #

    # Fetch R and gamma arrays from their respective sheets in the workbook
    r = np.asarray([c.value for c in wb_2d_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2d_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Loop to generate the subplots of the line profiles in "line_profile_gammas"
    for g in np.insert(line_profile_gammas, 0, min(gammas)):
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
        [val[0].value for val in wb_all_results["MRR"].iter_rows(min_row=2, max_col=1)]
    )
    s_max = np.asarray(
        [
            val[0].value
            for val in wb_all_results["MRR"].iter_rows(min_row=2, min_col=3, max_col=3)
        ]
    )
    α_bend, α_wg = np.asarray(
        [
            (val[0].value, val[1].value)
            for val in wb_all_results["MRR"].iter_rows(min_row=2, min_col=6, max_col=7)
        ]
    ).T
    h, gamma = np.asarray(
        [
            (val[0].value, val[1].value)
            for val in wb_all_results["MRR"].iter_rows(
                min_row=2, min_col=14, max_col=15
            )
        ]
    ).T
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
    axs[3].set_ylim(0, 10)
    axs[3].legend(loc="upper right")

    # Bottom horizontal axis labels
    axs[3].set_xlabel("Radius (μm)")

    # Save figure to file
    fig.savefig(filename_path.parent / f"{filename_path.stem}_FIG6.png")


def _determine_y_plotting_extrema(
    wb_2d_map: Workbook, wb_all_results: Workbook
) -> tuple[float, float]:
    """

    Args:
        wb_2d_map ():
        wb_all_results ():

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
                    min_row=2, min_col=3, max_col=3
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


def main():
    """

    Returns:

    """

    # Discrete gamma value arrays for plotting
    line_profile_gammas_fig_5: np.ndarray = np.asarray([20, 45, 65, 75])
    line_profile_gammas_fig_6: np.ndarray = np.asarray([20, 30, 45, 55, 60, 65, 70, 75])

    # Read Excel filenames from the command line, else define them locally
    if len(sys.argv) == 3:
        filename_all_results: str = sys.argv[1]
        filename_mrr_2d_maps_vs_gamma_and_r: str = sys.argv[2]
    else:
        filename_all_results: str = (
            "data/Tableau_REDUCED_TE_w07_dbrutes_R_ALL_RESULTS.xlsx"
        )
        filename_mrr_2d_maps_vs_gamma_and_r: str = (
            "data/Tableau_REDUCED_TE_w07_dbrutes_R_MRR_2DMAPS_VS_GAMMA_and_R"
            + "__alpha_wg_variable.xlsx"
        )
    filename_path: Path = Path(filename_all_results)

    # matplotlib initializations
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.grid.which": "both",
        },
    )
    plt.ion()

    # Load Excel workbooks
    wb_all_results: Workbook = load_workbook(filename_all_results, read_only=True)
    wb_2d_map: Workbook = load_workbook(
        filename_mrr_2d_maps_vs_gamma_and_r, read_only=True
    )

    # Generate figures
    y_max_s, y_max_αl = _determine_y_plotting_extrema(
        wb_2d_map=wb_2d_map, wb_all_results=wb_all_results
    )
    figure_3(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        gamma=30,
        filename_path=filename_path,
    )
    figure_5(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        line_profile_gammas=line_profile_gammas_fig_5,
        y_max_s=y_max_s,
        y_max_αl=y_max_αl,
        filename_path=filename_path,
    )
    figure_6(
        wb_2d_map=wb_2d_map,
        wb_all_results=wb_all_results,
        line_profile_gammas=line_profile_gammas_fig_6,
        y_max_s=y_max_s,
        filename_path=filename_path,
    )
    plt.show()

    # If running from the command line, either pause for user input to keep figures
    # visible. If running in PyCharm debugger, set breakpoint here.
    if sys.gettrace() is not None:
        print("Breakpoint here to keep figures visible in IDE!")
    else:
        input("Script paused to display figures, press any key to exit")


# Run the script...
main()
