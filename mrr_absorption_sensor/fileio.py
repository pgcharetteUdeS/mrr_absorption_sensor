"""fileio.py

File I/O utilities

All lengths are in units of um

"""
__all__ = [
    "load_toml_file",
    "validate_excel_results_file",
    "write_excel_results_file",
]

import dacite
from dacite import from_dict
from dataclasses import asdict
from openpyxl.workbook import Workbook
from pathlib import Path
from typing import Callable

import numpy as np
import toml
from rich import print
import sys

from .constants import constants, InputParameters
from .version import __version__


from .linear import Linear
from .models import Models
from .mrr import Mrr
from .spiral import Spiral


def load_toml_file(filename: Path, logger: Callable = print) -> InputParameters:
    """
    Load problem data from .toml file into a dataclass

    Args:
        filename (Path): .toml input file containing the problem data
        logger (Callable): console logger (optional)

    Returns: parms (InputParameters)

    """

    # Load problem parameters from a .toml file into an InputParameters dataclass
    toml_dict: dict = toml.load(filename)
    toml_dict |= {"filename": filename, "logger": logger}
    parms: InputParameters = from_dict(
        data_class=InputParameters, data=toml_dict, config=dacite.Config(strict=True)
    )
    logger(f"Loaded information from '{filename}'.")

    # Write out the parameters to a .toml file, for reference purposes
    filename_txt: Path = (
        filename.parent / parms.io.output_sub_dir / f"{filename.stem}_parameters.toml"
    )
    with open(filename_txt, "w") as f:
        f.write(f"# mrr_absorption_sensor package {__version__}\n")
        toml.dump(asdict(parms), f)
    logger(f"Wrote input parameters to '{filename_txt}'.")

    # Return dataclass with problem parameters
    return parms


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
            "if it's already open!"
        )
        sys.exit()

    return excel_output_filename


def write_excel_results_file(
    excel_output_path: Path,
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    parms: InputParameters,
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
        parms (InputParameters):
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
        f"{models.parms.wg.u_coord_name}_um": mrr.u,
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
    if models.parms.wg.v_coord_name == "w":
        alpha_wg_sheet.append(["height_um", "alpha_wg_dB_per_cm"])
    else:
        alpha_wg_sheet.append(["width_um", "alpha_wg_dB_per_cm"])
    if models.parms.wg.v_coord_name == "w":
        alpha_wg_interp_sheet.append(["height_ump", "alpha_wg_dB_per_cm"])
    else:
        alpha_wg_interp_sheet.append(["width_um", "alpha_wg_dB_per_cm"])
    for value in models.parms.geom.values():
        alpha_wg_sheet.append([value.u, value.alpha_wg])
    u_data = np.asarray([value.u for value in models.parms.geom.values()])
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
            f"{models.parms.wg.u_coord_name}_um",
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
        f"{models.parms.wg.u_coord_name}_um": linear.u,
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
    if parms.debug.analyze_spiral:
        spiral_data_dict = {
            "R_um": models.r,
            "maxS_RIU_inv": spiral.s,
            f"{models.parms.wg.u_coord_name}_um": spiral.u,
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
