"""CONSTANTS.PY

Global CONSTANTS

"""
__all__ = ["CONSTANTS", "LINE_STYLES", "InputParameters"]

from dataclasses import dataclass, field
from dacite import from_dict
from typing import NamedTuple
import numpy as np
from pathlib import Path
from typing import cast


class Constants(NamedTuple):
    """Global CONSTANTS
    - per_um_to_db_per_cm: covert losses from um-1 in exponent form to
                           dB/cm in DB form.
    """

    per_um_to_db_per_cm: float


CONSTANTS: Constants = Constants(np.log10(np.e) * 10 * 10000)


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


class Missing:
    """Sentinel"""

    pass


MISSING = Missing()


@dataclass
class Waveguide:
    """
    Waveguide parameters
    """

    lambda_resonance: float
    n_clad: float
    n_core: float
    n_sub: float
    core_height: float | Missing = MISSING
    core_width: float | Missing = MISSING
    roughness_lc: float = 50e-9
    roughness_sigma: float = 6e-9
    ni_op_point: float = 1.0e-6
    polarization: str = "TE"

    def __post_init__(self):
        # Determine if this is a "fixed core height" or "fixed core width" analysis
        if self.core_width is not MISSING and self.core_height is MISSING:
            self.u_coord_name: str = "h"
            self.v_coord_name: str = "w"
            self.v_coord_value: float = cast(float, self.core_width)
        elif self.core_height is not MISSING and self.core_width is MISSING:
            self.u_coord_name = "w"
            self.v_coord_name = "h"
            self.v_coord_value = cast(float, self.core_height)
        else:
            raise KeyError("EITHER 'core_width' OR 'core_height' is required!")


@dataclass
class Ring:
    """
    RIng parameters
    """

    coupling_length: float = 0


@dataclass
class SpiralParms:
    """
    Spiral parameters
    """

    spacing: float = 5.0
    turns_min: float = 0.5
    turns_max: float = 25.0


@dataclass
class Fitting:
    """
    Model fitting parameters
    """

    alpha_wg_order: int = 3
    alpha_wg_exponential_model: bool = False
    gamma_order: int = 4
    neff_order: int = 3
    optimization_local: bool = True
    optimization_method: str = "SLSQP"


@dataclass
class Limits:
    """
    Analysis limit parameters
    """

    r_min: float
    r_max: float
    r_samples_per_decade: int = 50
    alpha_bend_threshold: float = 0.01
    min_delta_ni: float = 1.0e-6
    T_SNR: float = 20.0
    u_min: float = 0
    u_max: float = 0
    gamma_min: float = 0
    gamma_max: float = 0


@dataclass
class IO:
    """
    Graphing and file I/O and parameters
    """

    map_line_profiles: list = field(default_factory=lambda: [20, 45, 65, 75])
    draw_largest_spiral: bool = True
    map2D_colormap: str = "viridis"
    map2D_n_grid_points: int = 500
    map2D_overlay_color_dark: str = "white"
    map2D_overlay_color_light: str = "white"
    output_sub_dir: str = ""
    write_2D_maps: bool = True
    write_excel_files: bool = True
    write_spiral_sequence_to_file: bool = True


@dataclass
class Debug:
    """
    Debugging and other flags
    """

    alpha_wg_exponential_model: bool = False
    disable_u_search_lower_bound: bool = False
    disable_R_domain_check: bool = False
    models_only: bool = False
    analyze_spiral: bool = True


@dataclass
class Geom:
    """
    Mode solver data class
    """

    u: float
    gamma: float
    neff: float
    r: list
    alpha_bend: list
    alpha_wg: float = 0
    r_alpha_bend_threshold: float = 0


@dataclass
class InputParameters:
    """
    Problem data loaded from a .toml file dictionary
    """

    wg: Waveguide
    ring: Ring
    spiral: SpiralParms
    fit: Fitting
    limits: Limits
    debug: Debug
    io: IO
    geom: dict = field(default_factory=dict)
    filename: Path = Path("")

    def __post_init__(self):
        # Convert geom dictionary values from dict to dataclass objects
        if len(self.geom) == 0:
            raise KeyError(f"[yellow]{self.filename}: no 'geom' mode solver data!")
        for key in self.geom:
            self.geom[key] = from_dict(data_class=Geom, data=self.geom[key])

        # Load the gamma and u extrema into the class variables
        self.limits.u_min = min(val.u for val in self.geom.values())
        self.limits.u_max = max(val.u for val in self.geom.values())
        self.limits.gamma_min = min(val.gamma for val in self.geom.values())
        self.limits.gamma_max = max(val.gamma for val in self.geom.values())

        # Check the input data
        self.check_input_data()

    def check_input_data(self):
        """
        Perform a number of consistency and value range check on the input data.
        """

        # Check selected parameters for valid content
        if self.wg.polarization not in ["TE", "TM"]:
            raise KeyError(f"{self.filename}: Invalid 'pol' key value!")

        # Check that u values are in ascending order and positive
        u_bending_loss: np.ndarray = np.asarray([val.u for val in self.geom.values()])
        if (
            not np.all(u_bending_loss[:-1] < u_bending_loss[1:])
            or u_bending_loss[0] <= 0
        ):
            raise KeyError(
                f"{self.filename}: 'geom' keys are not in ascending order of 'u'!"
            )

        # Check that gamma values are in descending order and reasonably valued
        if np.any(np.diff(np.asarray([val.gamma for val in self.geom.values()])) > 0):
            raise KeyError(f"{self.filename}: 'gamma' values not in descending order!")
        if max(val.gamma for val in self.geom.values()) > 1:
            raise KeyError(f"{self.filename}: 'gamma' must be in [0..1] range!")

        # Check that neff are reasonable
        if min(val.neff for val in self.geom.values()) < 1:
            raise KeyError(f"{self.filename}: neff values must be greater than 1!")

        # Check bending loss data (R and alpha_bend arrays)
        for val in self.geom.values():
            r_array: np.ndarray = np.asarray(val.r)
            alpha_bend_array: np.ndarray = np.asarray(val.alpha_bend)
            if len(r_array) < 3:
                raise KeyError(
                    f"{self.filename}: Insufficient number of bending loss data"
                    f" entries at u = {val.u:.3f} (number must be > 3)!"
                )
            elif len(r_array) != len(alpha_bend_array):
                raise KeyError(
                    f"{self.filename}: Unequal length radius and bending loss data"
                    f" arrays at u = {val.u:.3f}!"
                )
            elif len(np.unique(r_array)) != len(r_array):
                raise KeyError(
                    f"{self.filename}: Duplicate radius value in bending loss data"
                    f" at u = {val.u:.3f}!"
                )
            elif not np.all(r_array[:-1] < r_array[1:]):
                raise KeyError(
                    f"{self.filename}: Radius values are not in ascending order in"
                    f" bending loss data at u = {val.u:.3f}!"
                )
            elif r_array[0] <= 0:
                raise KeyError(
                    f"{self.filename}:  Negative radius value in bending loss data"
                    f" at u = {val.u:.3f}!"
                )
