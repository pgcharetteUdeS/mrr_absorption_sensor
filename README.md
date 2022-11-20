mrr_absorption_sensor package

## Purpose:
Calculate the maximum achievable sensitivities over a range of waveguide bending radii
for micro-ring resonator, spiral, and linear waveguide absorption sensors. The waveguide
core has either a fixed width or height and the other (free) core geometry parameter is
allowed to vary over a specified range in the optimization at each bending radius.

1. Fixed height: supply a "core_width" field and dictionary of "h" keys/value pairs in
   the .toml file.

2. Fixed width:  supply a "core_height" field and dictionary of "w" keys/value pairs in
   the .toml file.

Main interface method:
- analyze.analyze()

Classes:
- Models: interpolation of gamma(h), neff(h), alpha_prop(u), alpha_bend(r, u)
- Mrr, Linear, Spiral: sensor-specific variables and methods

## Notes:

1. The problem specifications are read in from a .toml file containing the problem
   parameters and data, see "example.toml" for explanations of the key/value pairs and
   fileio.load_toml_file().

2. The return value from analyze.analyze() MUST be stored in a local variable in the
   calling script for the buttons to work in the 3D graph of alpha_bend(r, u).

3. The alpha_bend(r, u) model is hardcoded in models.Models.fit_alpha_bend_model()
   but the code is structured in such a way that it is relatively easy to change, see
   the "USER-DEFINABLE MODEL-SPECIFIC SECTION" code section.

4. On windows, the colorama package must be initialized with colorama.init()
   prior to calling the package modules if colored text on the console is desired.

## Known issues:

...
