"""

mrr_absorption_sensor package

Main interface methods:
    - analyze.analyze()

Classes:
    - Models
    - sensors: Mrr, Linear, Spiral

Notes:
   1) On windows, the colorama package must be initialized with colorama.init()
      prior to calling the package modules if colored text in the console is desired.

   2) "Auto-removal of grids by pcolor() and pcolormesh() is deprecated..."
      warnings caused by a bug in matplotlib 3.5.1 can be suppressed
      by calling "warnings.filterwarnings("ignore", category=DeprecationWarning)"

"""
from .analyze import analyze
from .version import version
