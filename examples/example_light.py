"""
Simplified use of the mrr_absorption_sensor package using "example_light.toml"
"""

import matplotlib.pyplot as plt
from plot_article_figures import plot_article_figures
import sys

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
