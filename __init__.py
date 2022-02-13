#
# mrr_absorption_sensor package
#
# Classes:
#   - Models
#   - sensors: Mrr, Linear, Spiral
#
# Notes:
#   1) On windows, the colorama package must be initiazed with colorama.init()
#      prior to calling the package modules if colored text in the console is desired.
#   2) "Auto-removal of grids by pcolor() and pcolormesh() is deprecated..."
#      warnings caused by a bug in matplotlib 3.5.1 can be suppressed
#      by calling "warnings.filterwarnings("ignore", category=DeprecationWarning)"

from .fileio import (
    load_toml_file,
    validate_excel_output_file,
    write_excel_output_file,
    write_image_data_to_Excel,
)
from .linear import Linear
from .models import Models
from .mrr import Mrr
from .plotting import plot_results
from .spiral import Spiral
from .version import version
