# To run:
#   - run the script from the SAME directory as the mrr_absorption_sensor package.
# OR:
#   - run the script from anywhere, but the PYTHONDIR environment variable must contain
#     the PARENT directory of the mrr_absorption_sensor package.
#

import matplotlib.pyplot as plt
from mrr_absorption_sensor import analyze

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 6,
        "lines.linewidth": 0.5,
        "axes.linewidth": 0.5,
    },
)
analyze("example.toml")
