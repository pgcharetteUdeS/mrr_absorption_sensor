""" mrr_absorption_sensor

Calculate the maximum achievable sensitivities over a range of waveguide bending radii
for micro-ring resonator, spiral, and linear waveguide absorption sensors. The waveguide
core has either a fixed width or height and the other (free) core geometry parameter
is allowed to vary over a specified range in the optimization at each bending radius.

See:
    P. Girault et al., "Influence of Losses, Device Size, and Mode Confinement on
    Integrated Micro-Ring Resonator Performance for Absorption Spectroscopy Using
    Evanescent Field Sensing," in Journal of Lightwave Technology, 2022
    doi: 10.1109/JLT.2022.3220982.

Main interface method: analyze.analyze()

"""

from .analyze import analyze
from .linear import Linear
from .models import Models
from .mrr import Mrr
from .spiral import Spiral
from .version import __version__
