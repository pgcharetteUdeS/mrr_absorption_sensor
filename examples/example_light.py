"""

To run:
   - run the script from the SAME directory as the mrr_absorption_sensor package.
 OR:
   - run the script from anywhere, but the PYTHONDIR environment variable must contain
     the PARENT directory of the mrr_absorption_sensor package.

"""

import sys
import matplotlib.pyplot as plt
from plot_article_figures import plot_article_figures

# KLUDGE? Choose correct import statement depending on if running from .bat file
# or from an IDE.
if sys.gettrace() is not None:
    from mrr_absorption_sensor import analyze
else:
    from mrr_absorption_sensor.mrr_absorption_sensor import analyze

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
