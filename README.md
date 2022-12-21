mrr_absorption_sensor package

## Purpose:
Calculate the maximum achievable sensitivities over a range of curvature radii
for micro-ring resonator, spiral, and linear waveguide absorption sensors, as a function
of waveguide core geometry. The package was written to analyse and plot the results
described in:

P. Girault et al., "*Influence of Losses, Device Size, and Mode Confinement on
Integrated Micro-Ring Resonator Performance for Absorption Spectroscopy Using
Evanescent Field Sensing,*" Journal of Lightwave Technology, 2022,
doi: [10.1109/JLT.2022.3220982](https://doi.org/10.1109/JLT.2022.3220982).

The strip waveguide core has either a fixed width or height and the other (free) core
geometry parameter is allowed to vary over a specified range in the optimization
at each bending radius:

1. Fixed height: supply a "core_width" field and a dictionary of keys/value pairs
   with mode solver data as a function of height in the .toml file.

2. Fixed width:  supply a "core_height" field and a dictionary of keys/value pairs
   with mode solver data as a function of width in the .toml file.

Main interface method:
- analyze.analyze()

Classes:
- Models: interpolation of gamma(h), neff(h), alpha_prop(u), alpha_bend(r, u)
- Mrr, Linear, Spiral: sensor-specific variables and methods

## Notes:

1. The problem specifications are read in from a .toml file containing the problem
   parameters and data, see "example.toml" for explanations of the key/value pairs.

2. The return value from analyze.analyze() must be stored in a local variable in the
   calling script for the buttons to work in the 3D graph of alpha_bend(r, u).

3. The model for the radiative bending losses as a function of ring radius r and 
   waveguide core geometry free parameter u, alpha_bend(r, u), is hardcoded in
   models.Models._fit_α_bend_model() but the code is structured in such a way that it is
   relatively easy to change, see "USER-DEFINABLE MODEL-SPECIFIC SECTION" code section.

## Known issues:

None
