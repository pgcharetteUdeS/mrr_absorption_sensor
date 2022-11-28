"""example_light.py

Simplified use of mrr_absorption_sensor package using data from "example_light.toml"

See:
    P. Girault et al., "Influence of Losses, Device Size, and Mode Confinement on
    Integrated Micro-Ring Resonator Performance for Absorption Spectroscopy Using
    Evanescent Field Sensing," in Journal of Lightwave Technology, 2022
    doi: 10.1109/JLT.2022.3220982.

"""

import matplotlib.pyplot as plt
import sys

from plot_article_figures import plot_article_figures

# Add the parent directory so that the mrr_absorption_sensor directory can be found
sys.path.append("..\\")
from mrr_absorption_sensor import analyze


plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 6,
        "lines.linewidth": 0.5,
        "axes.linewidth": 0.5,
        "figure.max_open_warning": 25,
    },
)

analyze("example_light.toml")
plot_article_figures(
    results_file_name="data\\example_light_ALL_RESULTS.xlsx",
    maps_file_name="data\\example_light_MRR_2DMAPS_VS_GAMMA_and_R.xlsx",
)
print("Done example_light.py!")
