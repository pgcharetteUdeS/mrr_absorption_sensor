"""

Analyze all sensor types

Exposed methods:
   - analyze()

"""


# Standard library
from colorama import Fore, Style
from pathlib import Path

# Package modules
from .models import Models
from .mrr import Mrr
from .linear import Linear
from .spiral import Spiral
from .plotting import plot_results
from .fileio import load_toml_file, validate_excel_output_file, write_excel_output_file
from .version import version


def analyze(
    toml_input_file_path: Path,
    logger=print,
):
    """
    Calculate the maximum achievable sensitivities over a range of radii for micro-ring
    resonator, spiral, and linear waveguide absorption sensors.

    The waveguides have a fixed core width and where the core height is allowed to vary
    over a specified range to achieve maximum sensitivity at each radius.

    :param toml_input_file_path: input file containing the problem data (.toml format)
    :param logger: console logger (optional), for example from the "logging" package
    :return: None

    """
    # Show the package version number
    print(f"{Fore.BLUE}mrr_absorption_sensor package {version()}{Style.RESET_ALL}")

    # Load the problem parameters from the input .toml file
    (parameters, modes_data, bending_loss_data,) = load_toml_file(
        filename=toml_input_file_path,
        logger=logger,
    )

    # Build the filename Path object for the output files
    output_filenames_path: Path = (
        toml_input_file_path.parent
        / parameters["output_sub_dir"]
        / toml_input_file_path.name
    )

    # Instantiate the Models class to build/fit the interpolation models for
    # gamma_fluid(h), neff(h), and alpha_bend(h, r)
    models: Models = Models(
        modes_data=modes_data,
        bending_loss_data=bending_loss_data,
        Rmin=parameters["Rmin"],
        Rmax=parameters["Rmax"],
        R_samples_per_decade=parameters["R_samples_per_decade"],
        lambda_res=parameters["lambda_res"],
        pol=parameters["pol"],
        core_width=parameters["core_width"],
        ni_op=parameters["ni_op"],
        alpha_wg_dB_per_cm=parameters["alpha_wg"],
        filename_path=output_filenames_path,
        alpha_bend_threshold=parameters["alpha_bend_threshold"],
        gamma_order=parameters["gamma_order"],
        neff_order=parameters["neff_order"],
        logger=logger,
        disable_R_domain_check=parameters["disable_R_domain_check"],
    )

    # If only model fitting was required, return
    if parameters["models_only"]:
        return

    # Define output Excel filename: if file is already open, halt with an exception
    # (better to halt here with an exception than AFTER having done the analysis...)
    if parameters["write_excel_files"]:
        excel_output_fname = validate_excel_output_file(output_filenames_path)
    else:
        excel_output_fname = ""

    # Instantiate, then analyze the MRR sensor
    mrr = Mrr(models=models, logger=logger)
    mrr.analyze()

    # Instantiate, then analyze the linear sensor
    linear = Linear(models=models, logger=logger)
    linear.analyze()

    # Instantiate, then if required analyze the spiral sensor
    spiral = Spiral(
        spacing=parameters["spiral_spacing"],
        turns_min=parameters["spiral_turns_min"],
        turns_max=parameters["spiral_turns_max"],
        models=models,
        logger=logger,
    )
    if not parameters["no_spiral"]:
        spiral.analyze()

    # Plot results
    plot_results(
        models=models,
        mrr=mrr,
        linear=linear,
        spiral=spiral,
        T_SNR=parameters["T_SNR"],
        min_delta_ni=parameters["min_delta_ni"],
        filename_path=output_filenames_path,
        write_excel_files=parameters["write_excel_files"],
        no_spiral=parameters["no_spiral"],
        write_spiral_sequence_to_file=parameters["write_spiral_sequence_to_file"],
        logger=logger,
    )

    # If required, write the analysis results to the output Excel file
    if parameters["write_excel_files"]:
        write_excel_output_file(
            excel_output_fname=excel_output_fname,
            models=models,
            mrr=mrr,
            linear=linear,
            spiral=spiral,
            no_spiral=parameters["no_spiral"],
            logger=logger,
        ),
