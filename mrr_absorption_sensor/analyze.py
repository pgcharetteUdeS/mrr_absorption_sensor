"""analyse.py

Main package interface to analyze the 3 sensor types

"""
__all__ = ["analyze"]


from pathlib import Path
from rich import print, traceback
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .version import __version__
from .fileio import (
    load_toml_file,
    validate_excel_results_file,
    write_excel_results_file,
)
from .linear import Linear
from .models import Models
from .mrr import Mrr
from .spiral import Spiral

# matplotlib package initializations
plt.rcParams.update(
    {
        "axes.grid": True,
        "axes.grid.which": "both",
    },
)

# rich package initializations
traceback.install()


def analyze(
    toml_input_file: str | Path,
    block: bool = False,
    logger: Callable = print,
) -> Tuple[Models, Mrr | None, Linear | None, Spiral | None]:
    """
    Calculate the maximum achievable sensitivities over a range of radii for micro-ring
    resonator, spiral, and linear waveguide absorption sensors.

    The waveguide core has a fixed dimension "v" (width or height, according to the data
    in the input .toml file). The other dimension "u" is allowed to vary over a
    specified range to find the maximum sensitivity at each radius.

    Args:
        toml_input_file (str):
        block (bool):
        logger (Callable):

    Returns:
        models: Models class instance
        mrr: Mrr class instance
        linear: Linear class instance
        spiral: Spiral class instance

    """

    # Show the package version number
    logger(f"[magenta3]mrr_absorption_sensor package {__version__}")

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
        raise ValueError("No radii to analyze!")

    # If only model fitting was required, return
    if parameters["models_only"]:
        plt.show()
        return models, None, None, None

    # Define output Excel filename: if file is already open, halt with an exception
    # (better to halt here with an exception than AFTER having done the analysis...)
    excel_output_path: Path = Path("")
    if parameters["write_excel_files"]:
        excel_output_path = validate_excel_results_file(
            filename_path=output_filenames_path
        )

    # Instantiate sensor classes
    mrr = Mrr(models=models, logger=logger)
    linear = Linear(models=models, logger=logger)
    spiral = Spiral(models=models, logger=logger)

    # Analyze sensors
    mrr.analyze()
    linear.analyze()
    if parameters["analyze_spiral"]:
        spiral.analyze()

    # Plot results
    models.calculate_plotting_extrema(max_s=mrr.max_s)
    mrr.plot_optimization_results()
    linear.plot_optimization_results()
    if parameters["analyze_spiral"]:
        spiral.plot_optimization_results()
    mrr.plot_combined_sensor_optimization_results(linear=linear, spiral=spiral)

    # Write the analysis results to the output Excel file
    if parameters["write_excel_files"]:
        write_excel_results_file(
            excel_output_path=excel_output_path,
            models=models,
            mrr=mrr,
            linear=linear,
            spiral=spiral,
            parameters=parameters,
            logger=logger,
        ),

    # Show plots, return instantiated classes
    plt.show(block=block)
    return models, mrr, linear, spiral
