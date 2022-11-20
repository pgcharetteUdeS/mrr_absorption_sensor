"""constants.PY

Global constants

"""
__all__ = ["constants", "LINE_STYLES"]

from typing import NamedTuple
import numpy as np


class Constants(NamedTuple):
    """Global constants
    - PER_UM_TO_DB_PER_CM: covert losses from um-1 in exponent form to
                           dB/cm in DB form.
    """

    PER_UM_TO_DB_PER_CM: float


constants: Constants = Constants(np.log10(np.e) * 10 * 10000)


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
