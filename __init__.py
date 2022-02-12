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

from .models import Models
from .mrr import Mrr
from .linear import Linear
from .spiral import Spiral
from .fileio import (
    load_toml_file,
    define_excel_output_fname,
    write_2D_data_to_Excel,
    append_image_to_seq,
)
from .plotting import plot_results

# Show the package version number
from colorama import Fore, Style
from .version import version

print(f"{Fore.BLUE}mrr_absorption_sensor package {version()}{Style.RESET_ALL}")
