"""

Plot figure 5, 6a in the article

Data in file "*_R_MRR_2DMAPS_VS_GAMMA_and_R.xlsx"

"""

import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook


def _get_Re_Rw(wb_all_results: Workbook, gamma: float) -> tuple[float, float]:
    gammas = np.asarray(
        [
            val[0].value
            for val in wb_all_results["Re and Rw"].iter_rows(min_row=2, max_col=1)
        ]
    )
    index = int(np.argmin(np.abs(gammas - gamma)))
    Res, Rws = np.asarray(
        [
            (val[0].value, val[1].value)
            for val in wb_all_results["Re and Rw"].iter_rows(
                min_row=2, min_col=3, max_col=4
            )
        ]
    ).T

    return Res[index], Rws[index]


def plot_figure_3(wb_2D_map: Workbook, wb_all_results: Workbook, gamma: float):
    # Fetch R and gamma arrays from their respective sheets in the 2D map workbook
    R = np.asarray([c.value for c in wb_2D_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2D_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Fetch line profiles for S, Snr, and Se at gamma from the the 2D map workbook
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)
    S = np.asarray([c.value for c in wb_2D_map["S (RIU-1)"][index]])
    Snr = np.asarray([c.value for c in wb_2D_map["Snr (RIU-1)"][index]])
    Se = np.asarray([c.value for c in wb_2D_map["Se"][index]])

    # Get corresponding Re and Rw values from the all_results workbook
    Re, Rw = _get_Re_Rw(wb_all_results=wb_all_results, gamma=gamma)

    # Create the figure
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Figure 3")

    # PLot line profiles
    ax.semilogx(R, S, "b-", label=r"$S_{MRR}$")
    ax.semilogx(R, Snr, "r--", label=r"$S_{NR}$")
    index = int(np.argmin(np.abs(R - Rw)) + 1)
    ax.semilogx([Rw, Rw], [0, Snr[index]], "k")
    ax.text(Rw * 0.95, Snr[index] * 1.05, "RW")
    axR = ax.twinx()
    axR.semilogx(R, Se, "g--", label=r"$S_{e}$")
    index = int(np.argmin(np.abs(R - Re)) + 1)
    axR.semilogx([Re, Re], [0, Se[index]], "k")
    axR.text(Re * 0.95, Se[index] * 1.05, "Re")

    # PLot formatting, title, labels, etc.
    ax.set_title(
        "Sensitivity components as a function of radius"
        + rf" at $\Gamma_{{fluid}}$ = {gamma:.0f}$\%$"
    )
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel(r"$S_{MRR}$ and $S_{NR}$ (RIU$_{-1}$)")
    axR.set_ylabel(r"$S_e$")
    ax.set_ylim(0, 25000)
    axR.set_ylim(0, 25)
    ax.set_xlim(R[0], R[-1])
    ax.grid(visible=True)

    # Combine legends
    ax_lines = ax.get_legend_handles_labels()[0] + axR.get_legend_handles_labels()[0]
    ax_labels = ax.get_legend_handles_labels()[1] + axR.get_legend_handles_labels()[1]
    ax.legend(ax_lines, ax_labels, loc="upper left")


def _plot_figure_5_sub_plot(
    wb_2D_map: Workbook,
    wb_all_results: Workbook,
    gamma: float,
    R: np.ndarray,
    gammas: np.ndarray,
    ax: plt.Axes,
    last: bool,
):
    # Calcule the row index in the sheets for the requested gamma value
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)

    # Fetch line profiles data from the worksheets
    αL = np.asarray([c.value for c in wb_2D_map["alpha x L (dB)"][index]])
    αbendL = np.asarray([c.value for c in wb_2D_map["alpha_bend x L (dB)"][index]])
    αpropL = np.asarray([c.value for c in wb_2D_map["alpha_prop x L (dB)"][index]])
    S = np.asarray([c.value for c in wb_2D_map["S (RIU-1)"][index]])

    # Get corresponding Re and Rw values from the all_results workbook
    Re, Rw = _get_Re_Rw(wb_all_results=wb_all_results, gamma=gamma)

    # Plot the line profiles
    ax.semilogx(R, αL, "k", label="αL")
    ax.semilogx(R, αbendL, "g--", label="αbendL")
    ax.semilogx(R, αpropL, "r--", label="αpropL")
    ax.semilogx([Rw, Rw], [0, 50], "k")
    ax.text(Rw * 1.05, 50 * 1.025, "Rw")
    ax.semilogx([Re, Re], [0, 50], "k")
    ax.text(Re * 0.85, 50 * 1.025, "Re")
    axR = ax.twinx()
    axR.semilogx(R, S, "b", label=r"$S_{MRR}$")

    # PLot formatting, title, labels, etc.
    ax.set_title(rf"$\Gamma_{{fluid}}$ = {gamma:.0f} $\%$")
    ax.set_ylabel("Roundtrip losses (dB)")
    axR.set_ylabel(r"$S_{MRR}$ (RIU$^{-1}$)")
    ax.set_xlim(R[0], R[-1])
    ax.set_ylim(0, 60)
    axR.set_ylim(0, 50000)
    ax.grid(visible=True)
    if last:
        ax.set_xlabel("Radius (μm)")
    else:
        ax.axes.get_xaxis().set_ticklabels([])

    # Combine legends
    ax_lines = ax.get_legend_handles_labels()[0] + axR.get_legend_handles_labels()[0]
    ax_labels = ax.get_legend_handles_labels()[1] + axR.get_legend_handles_labels()[1]
    ax.legend(ax_lines, ax_labels, loc="upper left")


def plot_figure_5(
    wb_2D_map: Workbook, wb_all_results: Workbook, line_profile_gammas: np.ndarray
):
    # Fetch R and gamma arrays from their respective sheets in the workbook
    R = np.asarray([c.value for c in wb_2D_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2D_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Create figure with required number of subplots for the line profiles
    fig, axs = plt.subplots(len(line_profile_gammas), constrained_layout=True)
    fig.suptitle("Figure 5")

    # Loop to generate the subplots of the line profiles
    for i, gamma in enumerate(line_profile_gammas):
        _plot_figure_5_sub_plot(
            wb_2D_map=wb_2D_map,
            wb_all_results=wb_all_results,
            gamma=gamma,
            R=R,
            gammas=gammas,
            ax=axs[i],
            last=i >= len(line_profile_gammas) - 1,
        )


def plot_figure_6a(wb_2D_map: Workbook, line_profile_gammas: np.ndarray):
    # Fetch R and gamma arrays from their respective sheets in the workbook
    R = np.asarray([c.value for c in wb_2D_map["R (um)"][1]])
    gammas = np.asarray(
        [val[0].value for val in wb_2D_map["gamma (%)"].iter_rows(max_col=1)]
    )

    # Create the figure
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Figure 6a")

    # Loop to generate the subplots of the line profiles
    for g in line_profile_gammas:
        index = int(np.argmin(np.abs(gammas - g)) + 1)
        S = np.asarray([c.value for c in wb_2D_map["S (RIU-1)"][index]])
        ax.semilogx(R, S, label=f"{g:.0f}%")

    # PLot formatting, title, labels, etc.
    ax.set_title(r"Sensitivity as a function of $\Gamma_{fluid}$ (linear)")
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel(r"S$_{MRR}$")
    ax.set_xlim(R[0], R[-1])
    ax.set_ylim(0, 50000)
    ax.grid(visible=True)
    ax.legend(loc="upper left")


def main():
    plt.ion()

    # Define Excel filenames
    filename_ALL_RESULTS: str = "data/Tableau_REDUCED_TE_w07_dbrutes_R_ALL_RESULTS.xlsx"
    filename_MRR_2DMAPS_VS_GAMMA_AND_R: str = (
        "data/Tableau_REDUCED_TE_w07_dbrutes_R_MRR_2DMAPS_VS_GAMMA_and_R"
        + "__alpha_wg_variable.xlsx"
    )

    # Load Excel workbooks
    wb_all_results: Workbook = load_workbook(filename_ALL_RESULTS, read_only=True)
    wb_2D_map: Workbook = load_workbook(
        filename_MRR_2DMAPS_VS_GAMMA_AND_R, read_only=True
    )

    # Generate figures
    plot_figure_3(wb_2D_map=wb_2D_map, wb_all_results=wb_all_results, gamma=30)
    plot_figure_5(
        wb_2D_map=wb_2D_map,
        wb_all_results=wb_all_results,
        line_profile_gammas=np.asarray([20, 45, 65, 75]),
    )
    plot_figure_6a(
        wb_2D_map=wb_2D_map,
        line_profile_gammas=np.asarray([20, 30, 45, 55, 60, 65, 70, 75]),
    )
    plt.show()


main()
print("Done!")
