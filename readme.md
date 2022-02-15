mrr_absorption_sensor package

Purpose:
    Calculate the maximum achievable sensitivities over a range of waveguide bending
    radii for micro-ring resonator, spiral, and linear waveguide absorption sensors,
    where the waveguide core has a fixed width and the core height is allowed
    to vary over a specified range in the optimization at each bending radius.

Main interface methods:
    - analyze.analyze()

Classes:
    - Models: polynomial interpolation of gamma(h), neff(h), alpha_bend(r, h)
    - Mrr, Linear, Spiral: sensor-specific variables and methods

Notes:
   1) The problem specifications are normally read in from a .toml file containing
      the problem specification and data parameters, see "example.toml" for explanations
      of the key/value pairs, see fileio.load_toml_file() and the file example.toml.

   2) The model for alpha_bend(r, h) is hardcoded in Models.fit_alpha_bend_model()
      but the code is structured in such a way that it is relatively easy to change, 
      see the "USER-DEFINABLE MODEL-SPECIFIC SECTION" code section.

   3) On windows, the colorama package must be initialized with colorama.init()
      prior to calling the package modules if colored text in the console is desired.

Known issues:
   1) "Auto-removal of grids by pcolor() and pcolormesh() is deprecated..."
      warnings caused by a bug in matplotlib 3.5.1 can be suppressed
      by calling "warnings.filterwarnings("ignore", category=DeprecationWarning)"
