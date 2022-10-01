"""

constants.PY

"""

from collections import namedtuple

# Package version
__version__: str = "V20221001_08H30"

# Global constants in a namedTuple class
Constants = namedtuple("Constants", ["PER_UM_TO_DB_PER_CM"])
constants = Constants(4.34 * 10000)

# Define extra line styles, see:
# "https://matplotlib.org/3.5.1/gallery/lines_bars_and_markers/linestyles.html"
LINE_STYLES: dict = {
    "loosely dotted": (0, (1, 10)),
    "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "loosely dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}
