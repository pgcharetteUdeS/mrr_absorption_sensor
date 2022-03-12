"""

File I/O utilities

Exposed methods:
   - load_toml_file()
   - validate_excel_output_file()
   - write_excel_output_file()
   - write_image_data_to_Excel()

All lengths are in units of um

"""


# Standard library
from colorama import Fore, Style
import numpy as np
from openpyxl.workbook import Workbook
from pathlib import Path
import sys
import toml
from typing import Callable

# Package modules
from .models import Models
from .mrr import Mrr
from .linear import Linear
from .spiral import Spiral
from .version import __version__


def _check_mode_solver_data(
    modes_data: dict,
    bending_loss_data: dict,
    filename: Path,
    logger=print,
) -> bool:
    """
    For each h entry in the dictionary, check that the mode solver data are ordered,
    positive, and without duplicates. Check that the h dynamic range for the "modes"
    data covers that of the "bending loss" data
    """

    #
    # "Bending loss" mode solver data (alpha_bend, R) as a function of h
    #

    # Check that h values are in ascending order and positive
    h_bending_loss: np.ndarray = np.asarray(list(bending_loss_data.keys()))
    if not np.all(h_bending_loss[:-1] < h_bending_loss[1:]) or h_bending_loss[0] <= 0:
        logger(f"Bending loss data in '{filename}' are not in ascending h order!")
        return False

    # Check radii and alpha_bend arrays are ordered, positive, and without duplicates
    for h, value in bending_loss_data.items():
        Rs: np.ndarray = np.asarray(value["R"])
        ABs: np.ndarray = np.asarray(value["alpha_bend"])
        if len(Rs) != len(ABs) or len(Rs) < 3:
            logger(f"Invalid R/alpha_bend arrays for height {h:.3f} in '{filename}'!")
            return False
        if len(np.unique(Rs)) != len(Rs) or not np.all(Rs[:-1] < Rs[1:]) or Rs[0] <= 0:
            logger(f"Invalid R array for height {h:.3f} in '{filename}'!!")
            return False
        if (
            len(np.unique(ABs)) != len(ABs)
            or not np.all(ABs[:-1] > ABs[1:])
            or ABs[-1] <= 0
        ):
            logger(f"Invalid alpha_bend array for height {h:.3f} in '{filename}'!")
            return False

    #
    # "modes" mode solver data (neff, gamma) as a function of h
    #

    # Check that h values are in ascending order & positive, and that
    # the h dynamic range covers the bending loss data
    h_modes: np.ndarray = np.asarray(list(modes_data.keys()))
    if not np.all(h_modes[:-1] < h_modes[1:]) or h_modes[0] <= 0:
        logger(f"'modes' data in '{filename}' are not in ascending h order!")
        return False
    if h_bending_loss[0] < h_modes[0] or h_bending_loss[-1] > h_modes[-1]:
        logger(f"Mismatched modes and bending loss h dynamic range in '{filename}'!")
        return False

    # Check that neff and gamma values are reasonable
    for m in modes_data.values():
        if m["neff"] < 1 or m["gamma"] > 1:
            logger(f"Invalid 'modes' data in '{filename}'!")
            return False

    # Return success
    return True


def load_toml_file(
    filename: Path, logger: Callable = print
) -> tuple[dict, dict, dict,]:
    """

    Load problem data from .toml file and parse it into internal dictionaries.

    :param filename: .toml input file containing the problem data
    :param logger: console logger (optional)
    :return: parameters, modes_data, bending_loss_data
             --
             parameters (dict): problem parameters,
             modes_data (dict): gamma(h) mode solver data,
             bending_loss_data (dict): R(h) and alpha_bend(h) mode solver data

    """
    # Load dictionary from the .toml file
    toml_data = toml.load(str(filename))

    # Parse fields from .toml file into the parameters{} dictionary
    parameters: dict = {
        # Physical parameters
        "lambda_res": toml_data.get("lambda_res", 0.633),
        "pol": toml_data.get("pol", "TE"),
        "core_width": toml_data.get("core_width", 0.7),
        "ni_op": toml_data.get("ni_op", 1.0e-6),
        "alpha_wg": toml_data.get("alpha_wg", 1.0),
        "spiral_spacing": toml_data.get("spiral_spacing", 5.0),
        # Analysis parameters
        "Rmin": toml_data.get("Rmin", 25.0),
        "Rmax": toml_data.get("Rmax", 10000.0),
        "R_samples_per_decade": toml_data.get("R_samples_per_decade", 50),
        "spiral_turns_min": toml_data.get("spiral_turns_min", 1.0),
        "spiral_turns_max": toml_data.get("spiral_turns_max", 25.0),
        "T_SNR": toml_data.get("T_SNR", 20.0),
        "min_delta_ni": toml_data.get("min_delta_ni", 1.0e-6),
        "output_sub_dir": toml_data.get("output_sub_dir", ""),
        "alpha_bend_threshold": toml_data.get("alpha_bend_threshold", 0.001),
        "write_excel_files": toml_data.get("write_excel_files", True),
        "write_spiral_sequence_to_file": toml_data.get(
            "write_spiral_sequence_to_file", True
        ),
        # Fitting parameters
        "gamma_order": toml_data.get("gamma_order", 3),
        "neff_order": toml_data.get("neff_order", 3),
        # Debugging flags
        "models_only": toml_data.get("models_only", False),
        "disable_R_domain_check": toml_data.get("disable_R_domain_check", False),
        "no_spiral": toml_data.get("no_spiral", False),
    }

    # Check if .toml file contains unsupported keys, if so exit
    valid_keys: list = list(parameters.keys()) + ["modes", "h"]
    invalid_keys: list = [key for key in toml_data.keys() if key not in valid_keys]
    if invalid_keys:
        valid_keys.sort(key=lambda x: x.lower())
        logger(
            f"{Fore.YELLOW}File '{filename}' contains unsupported keys: "
            + f"{invalid_keys} - Valid keys are : {valid_keys}{Style.RESET_ALL}"
        )
        sys.exit()

    # Check selected keys for valid content
    if parameters["pol"] not in ["TE", "TM"]:
        logger(
            f"{Fore.YELLOW}Invalid 'pol' field value '{parameters['pol']}'"
            + f" in '{filename}'!{Style.RESET_ALL}"
        )
        sys.exit()

    # Copy the "neff" and "gamma" mode solver data to the modes_data{} dictionary.
    # If the "modes" field is specified, fetch the data from the associated field
    # value, else fetch the data from the "h" dictionary in the .toml file.
    modes_data: dict = {}
    if "modes" in toml_data.keys():
        for mode in toml_data["modes"]:
            h_key_um = mode[-1]
            if h_key_um in modes_data:
                print(
                    f"{Fore.YELLOW}Duplicate 'modes' entries in "
                    + f"'{filename}'!{Style.RESET_ALL}"
                )
                sys.exit()
            modes_data[h_key_um] = {
                "h": h_key_um,
                "neff": mode[0],
                "gamma": mode[1],
            }
    else:
        for k, value in toml_data["h"].items():
            h_key_um = float(k) / 1000
            modes_data[h_key_um] = {
                "h": h_key_um,
                "neff": value["neff"],
                "gamma": value["gamma"],
            }

    # Copy the "bending loss" mode solver data to the bending_loss_data{} dictionary
    bending_loss_data: dict = {}
    for k, value in toml_data["h"].items():
        h_key_um = float(k) / 1000
        bending_loss_data[h_key_um] = {
            "R": value["R"],
            "alpha_bend": value["alpha_bend"],
        }

    # Check the validity of the mode solver data, exit if problem found
    if not _check_mode_solver_data(
        modes_data=modes_data,
        bending_loss_data=bending_loss_data,
        filename=filename,
        logger=logger,
    ):
        logger(f"{Fore.YELLOW}Mode solver data validation fail!{Style.RESET_ALL}")
        sys.exit()

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

    return (
        parameters,
        modes_data,
        bending_loss_data,
    )


def validate_excel_output_file(filename_path: Path) -> str:
    """
    Define output Excel filename string from a Path object. Test to see if the file is
    already open, if so return an exception. This function is useful to run before any
    serious works starts because if the script tries to open a file that's already open,
    say in Excel, this causes the script to halt with an exception and the work done
    up to that point is lost.

    :param filename_path: Excel filename (Path)
    :return: Excel filename (str)
    """

    excel_output_filename: str = str(
        filename_path.parent / f"{filename_path.stem}_ALL_RESULTS.xlsx"
    )

    try:
        with open(excel_output_filename, "w"):
            pass
    except IOError:
        print(
            f"{Fore.YELLOW}Could not open '{excel_output_filename}', close it "
            + f"if it's already open!{Style.RESET_ALL}"
        )
        sys.exit()

    return excel_output_filename


def write_excel_output_file(
    excel_output_fname: str,
    models: Models,
    mrr: Mrr,
    linear: Linear,
    spiral: Spiral,
    no_spiral: bool = False,
    logger: Callable = print,
):
    """
    Write the analysis results to the output Excel file from a dictionary of
    key:value pairs, where the keys are the Excel file column header text strings
    and the values are the corresponding column data arrays

    :param excel_output_fname:
    :param models:
    :param mrr:
    :param linear:
    :param spiral:
    :param no_spiral:
    :param logger:
    :return: None
    """

    output_data_dict = {
        "R (um)": models.R,
        "neff": mrr.neff,
        "MRR max(S) (RIU-1)": mrr.S,
        "MRR Se": mrr.Se,
        "MRR Snr (RIU-1)": mrr.Snr,
        "MRR a2": mrr.a2,
        "MRR tau": mrr.tau,
        "MRR h (um)": mrr.h,
        "MRR gamma (%)": mrr.gamma,
        "MRR Finesse": mrr.Finesse,
        "MRR Q": mrr.Q,
        "FWHM": mrr.FWHM,
        "FSR": mrr.FSR,
        "LINEAR max(S) (RIU-1)": linear.S,
        "LINEAR h (um)": linear.h,
        "LINEAR gamma (%)": linear.gamma,
        "LINEAR L (um)": 2 * models.R,
        "LINEAR a2": linear.a2,
    }
    if not no_spiral:
        output_data_dict.update(
            {
                "SPIRAL max(S) (RIU-1)": spiral.S,
                "SPIRAL h (um)": spiral.h,
                "SPIRAL gamma (%)": spiral.gamma,
                "SPIRAL n turns": spiral.n_turns,
                "SPIRAL Rmin (um)": spiral.outer_spiral_r_min,
                "SPIRAL L (um)": spiral.L,
            }
        )
    output_data: np.ndarray = np.asarray(list(output_data_dict.values())).T
    wb = Workbook()
    wb.active.append(list(output_data_dict.keys()))
    for row in output_data:
        wb.active.append(row.tolist())
    wb.save(filename=excel_output_fname)
    logger(f"Wrote '{excel_output_fname}'.")


def write_image_data_to_Excel(
    filename: str,
    X: np.ndarray,
    x_label: str,
    Y: np.ndarray,
    y_label: str,
    Zs: list,
    z_labels: list,
):
    """
    Write image data to Excel file
    """

    wb = Workbook()

    # X sheet
    X_sheet = wb["Sheet"]
    X_sheet.title = x_label
    X_sheet.append(X.tolist())

    # Y sheet
    Y_sheet = wb.create_sheet(y_label)
    for y in Y:
        Y_sheet.append([y])

    # Z sheets
    for i, Z in enumerate(Zs):
        Z_sheet = wb.create_sheet(z_labels[i])
        for z in Z:
            Z_sheet.append(z.tolist())

    # Save file
    wb.save(filename=filename)
