"""

Analyze the 3 sensor types

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
from .fileio import load_toml_file, validate_excel_output_file, write_excel_results_file
from .version import __version__


def analyze(
    toml_input_file_path: Path,
    logger=print,
) -> tuple[Models, Mrr | None, Linear | None, Spiral | None]:
    """
    Calculate the maximum achievable sensitivities over a range of radii for micro-ring
    resonator, spiral, and linear waveguide absorption sensors.

    The waveguides core geometry has a fixed dimension "v", the other "u" is allowed
    to vary over a specified range to find the maximum sensitivity at each radius.

    :param toml_input_file_path: input file containing the problem data (.toml format)
    :param logger: console logger (optional), for example from the "logging" package
    :return: None

    """
    # Show the package version number
    print(f"{Fore.YELLOW}mrr_absorption_sensor package {__version__}{Style.RESET_ALL}")

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
    # gamma_fluid(u), neff(u), and alpha_bend(u, r)
    models: Models = Models(
        modes_data=modes_data,
        bending_loss_data=bending_loss_data,
        core_u_name=parameters["core_u_name"],
        core_v_name=parameters["core_v_name"],
        core_v_value=parameters["core_v_value"],
        Rmin=parameters["Rmin"],
        Rmax=parameters["Rmax"],
        R_samples_per_decade=parameters["R_samples_per_decade"],
        lambda_res=parameters["lambda_res"],
        pol=parameters["pol"],
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
        return models, None, None, None

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
        colormap2D=parameters["colormap2D"],
        no_spiral=parameters["no_spiral"],
        draw_largest_spiral=parameters["write_spiral_sequence_to_file"],
        write_spiral_sequence_to_file=parameters["write_spiral_sequence_to_file"],
        logger=logger,
    )

    # If required, write the analysis results to the output Excel file
    if parameters["write_excel_files"]:
        write_excel_results_file(
            excel_output_fname=excel_output_fname,
            models=models,
            mrr=mrr,
            linear=linear,
            spiral=spiral,
            no_spiral=parameters["no_spiral"],
            logger=logger,
        ),

    # Return the instantiated models class
    return models, mrr, linear, spiral
