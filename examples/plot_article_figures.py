"""

Plot figure 5, 6a in the article from data in file "*_R_MRR_2DMAPS_VS_GAMMA_and_R.xlsx"

"""

import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook, Workbook


def plot_figure_5_sub_plot(
    wb: Workbook,
    gamma: float,
    R: np.ndarray,
    gammas: np.ndarray,
    ax: plt.Axes,
    last: bool,
):
    # Calcule the row index in the sheets for the requested gamma value
    index = int(np.argmin(np.abs(gammas - gamma)) + 1)

    # Fetch line profiles data from the worksheets
    αL = np.asarray([c.value for c in wb["alpha x L (dB)"][index]])
    αbendL = np.asarray([c.value for c in wb["alpha_bend x L (dB)"][index]])
    αpropL = np.asarray([c.value for c in wb["alpha_prop x L (dB)"][index]])
    S = np.asarray([c.value for c in wb["S (RIU-1)"][index]])

    # Plot the line profiles
    ax.semilogx(R, αL, "k", label="αL")
    ax.semilogx(R, αbendL, "g--", label="αbendL")
    ax.semilogx(R, αpropL, "r--", label="αpropL")
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


def plot_figure_5(wb: Workbook, line_profile_gammas: np.ndarray):
    # Fetch R and gamma arrays from their respective sheets in the workbook
    R = np.asarray([c.value for c in wb["R (um)"][1]])
    gammas = np.asarray([val[0].value for val in wb["gamma (%)"].iter_rows(max_col=1)])

    # Create figure with required number of subplots for the line profiles
    fig, axs = plt.subplots(len(line_profile_gammas), constrained_layout=True)
    fig.suptitle("Figure 5")

    # Loop to generate the subplots of the line profiles
    for i, gamma in enumerate(line_profile_gammas):
        plot_figure_5_sub_plot(
            wb=wb,
            gamma=gamma,
            R=R,
            gammas=gammas,
            ax=axs[i],
            last=i >= len(line_profile_gammas) - 1,
        )
    plt.show()


def plot_figure_6a(wb: Workbook, line_profile_gammas: np.ndarray):
    # Fetch R and gamma arrays from their respective sheets in the workbook
    R = np.asarray([c.value for c in wb["R (um)"][1]])
    gammas = np.asarray([val[0].value for val in wb["gamma (%)"].iter_rows(max_col=1)])

    # Create the figure
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle("Figure 6a")

    # Loop to generate the subplots of the line profiles
    for g in line_profile_gammas:
        index = int(np.argmin(np.abs(gammas - g)) + 1)
        S = np.asarray([c.value for c in wb["S (RIU-1)"][index]])
        ax.semilogx(R, S, label=f"{g:.0f}%")

    # PLot formatting, title, labels, etc.
    ax.set_title(r"Sensitivity as a function of $\Gamma_{fluid}$ (linear)")
    ax.set_xlabel("Radius (μm)")
    ax.set_ylabel(r"S$_{MRR}$")
    ax.set_xlim(R[0], R[-1])
    ax.set_ylim(0, 50000)
    ax.grid(visible=True)
    ax.legend(loc="upper left")
    plt.show()


def main():
    plt.ion()
    fname: str = (
        "data/Tableau_REDUCED_TE_w07_dbrutes" + "_R_MRR_2DMAPS_VS_GAMMA_and_R.xlsx"
    )
    plot_figure_5(
        wb=load_workbook(fname, read_only=True),
        line_profile_gammas=np.asarray([20, 45, 65, 75]),
    )
    plot_figure_6a(
        wb=load_workbook(fname, read_only=True),
        line_profile_gammas=np.asarray([20, 30, 45, 55, 60, 65, 70, 75]),
    )


main()
print("Done!")
