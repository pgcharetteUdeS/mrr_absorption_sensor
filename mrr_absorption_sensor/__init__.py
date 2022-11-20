""" mrr_absorption_sensor

Calculate the maximum achievable sensitivities over a range of waveguide bending radii
for micro-ring resonator, spiral, and linear waveguide absorption sensors. The waveguide
core has either a fixed width or height and the other (free) core geometry parameter
is allowed to vary over a specified range in the optimization at each bending radius.

Main interface method: analyze.analyze()

"""

from .analyze import analyze
from .linear import Linear
from .models import Models
from .mrr import Mrr
from .spiral import Spiral
from .version import __version__
