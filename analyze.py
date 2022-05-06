"""

Analyze the 3 sensor types

Exposed methods:
   - analyze()

"""


# Standard library
import colorama as colorama
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
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
    toml_input_file: str,
    logger=print,
) -> tuple[Models, Mrr | None, Linear | None, Spiral | None]:
    """
    Calculate the maximum achievable sensitivities over a range of radii for micro-ring
    resonator, spiral, and linear waveguide absorption sensors.

    The waveguides core geometry has a fixed dimension "v", the other "u" is allowed
    to vary over a specified range to find the maximum sensitivity at each radius.

    :param toml_input_file: input file containing the problem data (.toml format)
    :param logger: console logger (optional), for example from the "logging" package
    :return: None

    """

    # Initialize the colorama package to print colored text to the Windows console
    colorama.init()

    # Show the package version number
    logger(f"{Fore.YELLOW}mrr_absorption_sensor package {__version__}{Style.RESET_ALL}")

    # matplotlib initializations
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.grid.which": "both",
        },
    )

    # Load the problem parameters from the input .toml file
    toml_input_file_path: Path = Path(toml_input_file)
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
    # gamma_fluid(u), neff(u), alpha_wg(u), and alpha_bend(u, r)
    models: Models = Models(
        parameters=parameters,
        modes_data=modes_data,
        bending_loss_data=bending_loss_data,
        filename_path=output_filenames_path,
        logger=logger,
    )

    # Check that the array of radii to be analyzed is not empty
    if np.size(models.r) == 0:
        raise ValueError(f"{Fore.YELLOW}No radii to analyze!{Style.RESET_ALL}")

    # If only model fitting was required, return
    if parameters["models_only"]:
        plt.show()
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
        models=models,
        parameters=parameters,
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
        filename_path=output_filenames_path,
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
    plt.show()
    return models, mrr, linear, spiral
