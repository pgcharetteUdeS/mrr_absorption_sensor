mrr_absorption_sensor package

Purpose:
    Calculate the maximum achievable sensitivities over a range of radii for micro-ring
    resonator, spiral, and linear waveguide absorption sensors.

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
