"""fileio.py

File I/O utilities

All lengths are in units of um

"""
__all__ = ["load_toml_file", "validate_excel_results_file", "write_excel_results_file"]

from openpyxl.workbook import Workbook
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import toml
from rich import print
import sys

from .constants import constants
from .linear import Linear
from .models import Models
from .mrr import Mrr
from .spiral import Spiral
from .version import __version__


def _check_mode_solver_data(modes_data: dict, bending_loss_data: dict, filename: Path):
    """
    For each u entry in the dictionary, check that the mode solver data are ordered,
    positive, and without duplicates.
    """

    # Check that the mode solver data dictionary is not empty
    if not bending_loss_data:
        raise ValueError(f"No bending loss data loaded from '{filename}'!")

    # Check that u values are in ascending order and positive
    u_bending_loss: np.ndarray = np.asarray(list(bending_loss_data.keys()))
    if not np.all(u_bending_loss[:-1] < u_bending_loss[1:]) or u_bending_loss[0] <= 0:
        raise ValueError(
            f"Bending loss data in '{filename}' are not in ascending h order!"
        )

    # Check that gamma values are in monotonically descending order
    if np.any(np.diff(np.asarray([v.get("gamma") for v in modes_data.values()])) > 0):
        raise ValueError("Gamma values not in monotonically decreasing order!")

    # Check that radius is ordered, positive, and without duplicates
    for u, value in bending_loss_data.items():
        r_array: np.ndarray = np.asarray(value["R"])
        if len(r_array) < 3:
            raise ValueError(f"Invalid R array for u = {u:.3f} in '{filename}'!")
        if (
            len(np.unique(r_array)) != len(r_array)
            or not np.all(r_array[:-1] < r_array[1:])
            or r_array[0] <= 0
        ):
            raise ValueError(f"Invalid R array for u {u:.3f} in '{filename}'!")

    # Check that neff and gamma values are reasonable
    for mode in modes_data.values():
        if mode["neff"] < 1 or mode["gamma"] > 1:
            raise ValueError(f"Invalid 'modes' data in '{filename}'!")


def load_toml_file(
    filename: Path, logger: Callable = print
) -> Tuple[dict, dict, dict,]:
    """

    Load problem data from .toml file and parse it into internal dictionaries.

    :param filename: .toml input file containing the problem data
    :param logger: console logger (optional)
    :return: parameters, modes_data, bending_loss_data
             --
             parameters (dict): problem parameters,
             bending_loss_data (dict): R(u) and alpha_bend(u) mode solver data

    """
    # Console message
    logger(f"Loading information from '{filename}'...")

    # Load dictionary from the .toml file
    toml_data = toml.load(str(filename))

    # Parse fields from .toml file into the parameters{} dictionary
    parameters: dict = {
        # Waveguide physical parameters
        "core_height": toml_data.get("core_height"),
        "core_width": toml_data.get("core_width"),
        "n_clad": toml_data.get("n_clad"),
        "n_core": toml_data.get("n_core"),
        "n_sub": toml_data.get("n_sub"),
        "roughness_lc": toml_data.get("roughness_lc", 50e-9),
        "roughness_sigma": toml_data.get("roughness_sigma", 10e-9),
        "lambda_res": toml_data.get("lambda_res", 0.633),
        "ni_op": toml_data.get("ni_op", 1.0e-6),
        "pol": toml_data.get("pol", "TE"),
        # MRR physical parameters
        "lc": toml_data.get("lc", 0),
        # Spiral physical parameters
        "spiral_spacing": toml_data.get("spiral_spacing", 5.0),
        "spiral_turns_min": toml_data.get("spiral_turns_min", 0.5),
        "spiral_turns_max": toml_data.get("spiral_turns_max", 25.0),
        # Model fitting parameters
        "alpha_wg_order": toml_data.get("alpha_wg_order", 3),
        "gamma_order": toml_data.get("gamma_order", 4),
        "neff_order": toml_data.get("neff_order", 3),
        "optimization_local": toml_data.get("optimization_local", True),
        "optimization_method": toml_data.get("optimization_method", "Powell"),
        # Analysis parameters
        "alpha_bend_threshold": toml_data.get("alpha_bend_threshold", 0.01),
        "min_delta_ni": toml_data.get("min_delta_ni", 1.0e-6),
        "Rmin": toml_data.get("Rmin", 25.0),
        "Rmax": toml_data.get("Rmax", 10000.0),
        "R_samples_per_decade": toml_data.get("R_samples_per_decade", 100),
        "T_SNR": toml_data.get("T_SNR", 20.0),
        # Graphing and file I/O and parameters
        "draw_largest_spiral": toml_data.get("draw_largest_spiral", True),
        "map2D_colormap": toml_data.get("map2D_colormap", "viridis"),
        "map2D_n_grid_points": toml_data.get("map2D_n_grid_points", 500),
        "map2D_overlay_color_dark": toml_data.get("map2D_overlay_color_dark", "white"),
        "map2D_overlay_color_light": toml_data.get(
            "map2D_overlay_color_light", "black"
        ),
        "map_line_profiles": toml_data.get("map_line_profiles", []),
        "output_sub_dir": toml_data.get("output_sub_dir", ""),
        "write_2D_maps": toml_data.get("write_2D_maps", True),
        "write_excel_files": toml_data.get("write_excel_files", True),
        "write_spiral_sequence_to_file": toml_data.get(
            "write_spiral_sequence_to_file", True
        ),
        # Debugging and other flags
        "alpha_wg_exponential_model": toml_data.get(
            "alpha_wg_exponential_model", False
        ),
        "disable_u_search_lower_bound": toml_data.get(
            "disable_u_search_lower_bound", False
        ),
        "disable_R_domain_check": toml_data.get("disable_R_domain_check", False),
        "models_only": toml_data.get("models_only", False),
        "analyze_spiral": toml_data.get("analyze_spiral", True),
    }

    # Check if .toml file contains unsupported keys, if so exit
    valid_keys: list = list(parameters.keys()) + ["h", "w"]
    if invalid_keys := [key for key in toml_data.keys() if key not in valid_keys]:
        valid_keys.sort(key=lambda x: x.lower())
        raise ValueError(
            f"File '{filename}' contains unsupported keys: "
            + f"{invalid_keys} - Valid keys are : {valid_keys}"
        )

    # Determine if this is a "fixed core height" or "fixed core width" analysis
    if (
        toml_data.get("core_width") is not None
        and toml_data.get("core_height") is not None
    ) or (toml_data.get("core_width") is None and toml_data.get("core_height") is None):
        raise ValueError(
            "EITHER 'core_width' OR 'core_height' fields nust be specified!"
        )
    if toml_data.get("core_width") is not None:
        logger("Fixed waveguide core width analysis.")
        parameters["core_u_name"] = "h"
        parameters["core_v_name"] = "w"
        parameters["core_v_value"] = toml_data.get("core_width")
        if toml_data.get("h") is None:
            raise ValueError("No 'h' fields specified!")
    else:
        logger("Fixed waveguide core height analysis.")
        parameters["core_u_name"] = "w"
        parameters["core_v_name"] = "h"
        parameters["core_v_value"] = toml_data.get("core_height")
        if toml_data.get("w") is None:
            raise ValueError("No 'w' fields specified!")

    # Check selected parameters for presence and/or valid content
    required_parameters: list = [
        "n_clad",
        "n_core",
        "n_sub",
    ]
    for key in required_parameters:
        if key not in parameters:
            raise ValueError(f"Missing '{key}' parameter in '{filename}'!")
    if parameters["pol"] not in ["TE", "TM"]:
        raise ValueError(
            f"Invalid 'pol' field value '{parameters['pol']}' in '{filename}'!"
        )

    # Copy the neff(u) and gamma(u) mode solver data to the "modes_data{}" dictionary,
    # and the alpha_bend(R, u) mode solver data to the "bending_loss_data{}" dictionary
    modes_data: dict = {}
    bending_loss_data: dict = {}
    for k, value in toml_data[parameters["core_u_name"]].items():
        u_key_um = float(k) / 1000
        modes_data[u_key_um] = {
            "u": u_key_um,
            "neff": value["neff"],
            "gamma": value["gamma"],
        }
        bending_loss_data[u_key_um] = {
            "R": value["R"],
            "alpha_bend": value["alpha_bend"],
        }

    # Check the validity of the mode solver data
    _check_mode_solver_data(
        modes_data=modes_data,
        bending_loss_data=bending_loss_data,
        filename=filename,
    )

    # Console message
    logger(f"Loaded information from '{filename}'.")

    # Write out the parameters dictionary to a text file
    filename_txt: Path = (
        filename.parent
        / parameters["output_sub_dir"]
        / f"{filename.stem}_parameters.toml"
    )
    with open(filename_txt, "w") as f:
        f.write(f"# mrr_absorption_sensor package {__version__}\n")
        toml.dump(parameters, f)
    logger(f"Wrote input parameters to '{filename_txt}'.")

    # Return dictionaries of loaded values
    return (
        parameters,
        modes_data,
        bending_loss_data,
    )


def validate_excel_results_file(filename_path: Path) -> Path:
    """
    Define output Excel filename string from a Path object. Test to see if the file is
    already open, if so return an exception.

    This function is useful to run before any serious works starts because if the script
    tries to open a file that's already open, say in Excel, this causes the script to
    halt with an exception and the work done up to that point is lost.

    Args:
        filename_path (Path): filename path object to test for openness, conversion

    Returns:
        Path: filename path

    """

    excel_output_filename: Path = Path(
        filename_path.parent / f"{filename_path.stem}_ALL_RESULTS.xlsx"
    )

    try:
        with open(str(excel_output_filename), "w"):
            pass
    except IOError:
        print(
            f"Could not open '{excel_output_filename}', close it "
            + "if it's already open!"
        )
        sys.exit()

    return excel_output_filename


def write_excel_results_file(
    excel_output_path: Path,
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    parameters: dict,
    logger: Callable = print,
):
    """
    Write the analysis results to the output Excel file from a dictionary of
    key:value pairs, where the keys are the Excel file column header text strings
    and the values are the corresponding column data arrays

    Args:
        excel_output_path (Path):
        models (Models):
        mrr (Mrr):
        linear (Linear):
        spiral (Spiral):
        parameters (dict):
        logger (Callable):

    Returns: None

    """
    # Create Excel workbook
    wb = Workbook()

    # Save the MMR data to a sheet
    mrr_data_dict = {
        "R_um": models.r,
        "neff": mrr.n_eff,
        "maxS_RIU_inv": mrr.s,
        "Se": mrr.s_e,
        "Snr_RIU_inv": mrr.s_nr,
        "alpha_bend_dB_per_cm": mrr.α_bend * constants.PER_UM_TO_DB_PER_CM,
        "alpha_wg_dB_per_cm": mrr.α_wg * constants.PER_UM_TO_DB_PER_CM,
        "alpha_prop_dB_per_cm": mrr.α_prop * constants.PER_UM_TO_DB_PER_CM,
        "alpha_dB_per_cm": mrr.α * constants.PER_UM_TO_DB_PER_CM,
        "alphaL": mrr.αl,
        "a2": mrr.wg_a2,
        "tau": mrr.tau,
        "T_max": mrr.t_max,
        "T_min": mrr.t_min,
        "ER_dB": mrr.er,
        "contrast": mrr.contrast,
        f"{models.core_u_name}_um": mrr.u,
        "gamma_percent": mrr.gamma,
        "Finesse": mrr.finesse,
        "Q": mrr.q,
        "FWHM_um": mrr.fwhm,
        "FSR_um": mrr.fsr,
    }
    mrr_data: np.ndarray = np.asarray(list(mrr_data_dict.values())).T
    mrr_sheet = wb["Sheet"]
    mrr_sheet.title = "MRR"
    mrr_sheet.append(list(mrr_data_dict.keys()))
    for row in mrr_data:
        mrr_sheet.append(row.tolist())

    # Save the calculated and interpolated alpha_wg(u) values to a sheet
    alpha_wg_sheet = wb.create_sheet("alpha_wg")
    alpha_wg_interp_sheet = wb.create_sheet("alpha_wg_interp")
    if models.core_v_name == "w":
        alpha_wg_sheet.append(["height_um", "alpha_wg_dB_per_cm"])
    else:
        alpha_wg_sheet.append(["width_um", "alpha_wg_dB_per_cm"])
    if models.core_v_name == "w":
        alpha_wg_interp_sheet.append(["height_ump", "alpha_wg_dB_per_cm"])
    else:
        alpha_wg_interp_sheet.append(["width_um", "alpha_wg_dB_per_cm"])
    for value in models.modes_data.values():
        alpha_wg_sheet.append([value["u"], value["alpha_wg"]])
    u_data = np.asarray([value.get("u") for value in models.modes_data.values()])
    u_interp: np.ndarray = np.linspace(u_data[0], u_data[-1], 100)
    alpha_wg_modeled = np.asarray([models.α_wg_of_u(u) for u in u_interp]) * (
        constants.PER_UM_TO_DB_PER_CM
    )
    for u, alpha_wg in zip(u_interp, alpha_wg_modeled):
        alpha_wg_interp_sheet.append([u, alpha_wg])

    # Save the Re(gamma) & Rw(gamma) arrays to a sheet
    re_rw_sheet = wb.create_sheet("Re and Rw")
    re_rw_sheet.append(
        [
            "gamma_percent",
            f"{models.core_u_name}_um",
            "Re_um",
            "Rw_um",
            "A_um_inv",
            "B_um_inv",
        ]
    )
    for line in zip(
        mrr.gamma_resampled * 100,
        mrr.u_resampled,
        mrr.r_e,
        mrr.r_w,
        mrr.α_bend_a,
        mrr.α_bend_b,
    ):
        re_rw_sheet.append(line)

    # Save the linear waveguide data to a sheet
    linear_data_dict = {
        "R_um": models.r,
        "maxS_RIU_inv": linear.s,
        f"{models.core_u_name}_um": linear.u,
        "gamma_percent": linear.gamma,
        "L_um": 2 * models.r,
        "a2": linear.wg_a2,
    }
    linear_data: np.ndarray = np.asarray(list(linear_data_dict.values())).T
    linear_sheet = wb.create_sheet("Linear")
    linear_sheet.append(list(linear_data_dict.keys()))
    for row in linear_data:
        linear_sheet.append(row.tolist())

    # If required, save the spiral data to a sheet
    if parameters["analyze_spiral"]:
        spiral_data_dict = {
            "R_um": models.r,
            "maxS_RIU_inv": spiral.s,
            f"{models.core_u_name}_um": spiral.u,
            "gamma_percent": spiral.gamma,
            "n_revs": spiral.n_turns * 2,
            "Rmin_um": spiral.outer_spiral_r_min,
            "L_um": spiral.l,
            "a2": spiral.wg_a2,
        }
        spiral_data: np.ndarray = np.asarray(list(spiral_data_dict.values())).T
        spiral_sheet = wb.create_sheet("Spiral")
        spiral_sheet.append(list(spiral_data_dict.keys()))
        for row in spiral_data:
            spiral_sheet.append(row.tolist())

    # Save the Excel file to disk
    wb.save(filename=str(excel_output_path))
    logger(f"Wrote '{excel_output_path}'.")
