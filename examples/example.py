"""

To run:
   - run the script from the SAME directory as the mrr_absorption_sensor package.
 OR:
   - run the script from anywhere, but the PYTHONDIR environment variable must contain
     the PARENT directory of the mrr_absorption_sensor package.

"""

import matplotlib.pyplot as plt
from plot_article_figures import plot_article_figures

# If running from .bat file, use FIRST import statement, else SECOND statement (in IDE)
from mrr_absorption_sensor.mrr_absorption_sensor import analyze
#from mrr_absorption_sensor import analyze

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 6,
        "lines.linewidth": 0.5,
        "axes.linewidth": 0.5,
    },
)
analyze("example.toml")
# analyze("Tableau_TE_h03.toml")
plot_article_figures(
    results_file_name="data\\example_ALL_RESULTS.xlsx",
    maps_file_name="data\\example_MRR_2DMAPS_VS_GAMMA_and_R.xlsx",
)

print("Done!")
