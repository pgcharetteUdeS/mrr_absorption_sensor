"""

FILENAME: example.py

PURPOSE: Determine the maximum achievable sensitivity as a function of the waveguide
         core geometry free parameter (height or width) over a range of ring/spiral
         radii for micro-ring resonator, Archimedes spiral, and linear (straight)
         waveguide sensors.

COMMAND LINE PARAMETERS (optional):
    --in_data_file <filename.toml>
        .toml file containing the problem specification parameters, see "example.toml"
        for explanations of the key/value pairs. If this parameter is absent,
        for example when running the script from an IDE, it must be specified in main().
    --no_pause
        disable "pause while waiting for user input" at the end of the script (this
        is useful when running multiple analyses from a .bat file so that the .bat
        file doesn't pause for user input after every call to the script).

NOTES:
    1) To include the "mrr_absorption_sensor" package, use one of the following options:
       - Copy the package directory to the project directory.
       - Add the parent directory of the package directory to the Python search path
         (on Windows, add it to the PYTHONPATH environment variable).
       - In PyCharm, add the directory to the IDE search path:
         <Settings><Project Interpreter>, wheel icon next to the interpreter,
         <Show all...>, "Show paths for selected interpreter" icon.
        - In the script, use the "sys.path.append(<package parent directory>)"


IDEs issues/notes:
    PyCharm:
      1) Sometimes the "assert" statements stop halting execution and dropping into the
         debugger. Solution: "Run/View Breakpoints" to open the Breakpoints dialog,
         enable "Python Exception Breakpoints" node in the tree on the left side menu.
      2) The keyboard stops working in Windows: go the Mouse settings dialog, then
         "Additional mouse options/Pointer Options", de-select or select/de-select
         the "Enhance pointer position" option, "Apply", "Ok".

    Spyder:
      1) Two solutions for interactive graphs (ex: rotating 3D scatter plot,
         x/y cursor locations in 2D graphs), BUT THIS HANGS THE DEBUGGER (known
         bug in current Spyder release):
           - Tools > preferences > IPython console > Graphics > Graphics backend
                                                                          > Automatic
           - At the console, type: "matplotlib auto"

"""

# Standard library packages
import argparse
import logging
from rich import print
from rich.logging import RichHandler
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

# mrr_absorption_sensor package
# KLUDGE? Choose correct import statement depending on if running from .bat file
# or from an IDE.
if sys.gettrace() is not None:
    from mrr_absorption_sensor import analyze
else:
    from mrr_absorption_sensor.mrr_absorption_sensor import analyze
from plot_article_figures import plot_article_figures


def mrr_absorption_sensor_vs_spiral(toml_input_filename: str):
    """

    :param toml_input_filename: input .toml filename
    :return: None
    """

    # Start execution timer
    global_start_time: float = time.perf_counter()

    # matplotlib initializations
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.size": 6,
            "lines.linewidth": 0.5,
            "axes.linewidth": 0.5,
            "figure.max_open_warning": 25,
        },
    )

    # Define the logger for console information messages
    logging.basicConfig(
        level="INFO", format="%(funcName)s():%(message)s", handlers=[RichHandler()]
    )
    logger: logging.Logger = logging.getLogger("rich")

    # Parser for command line parameter input (ex: running from .bat file)
    parser = argparse.ArgumentParser(description="Analyse waveguide sensors")
    parser.add_argument("--in_data_file", type=str)
    parser.add_argument("--no_pause", action="store_true")
    args = parser.parse_args()

    # Load input .toml file name from the command line if supplied, else use filename
    # passed as a parameter to the function. Verify that the file exists.
    toml_input_file: str = (
        toml_input_filename if args.in_data_file is None else args.in_data_file
    )
    if not Path(toml_input_file).is_file():
        print(f"File '{toml_input_file}' does not exist!")
        sys.exit()

    # Run the sensor analysis
    models, mrr, linear, spiral = analyze(
        toml_input_file=toml_input_file, logger=logger.info
    )

    # Generate the article figures
    print(f"Analyze sensors for data read from file '{models.filename_path.name}'")
    datafile_mantissa: str = (
        f"{models.parms.io.output_sub_dir}\\" f"{models.filename_path.stem}"
    )
    plot_article_figures(
        results_file_name=f"{datafile_mantissa}_ALL_RESULTS.xlsx",
        maps_file_name=f"{datafile_mantissa}_MRR_2DMAPS_VS_GAMMA_and_R.xlsx",
    )

    # Show execution time
    elapsed_time: float = time.perf_counter() - global_start_time
    print(
        f"[magenta3]Time: {time.strftime('%M:%S', time.gmtime(elapsed_time))}"
        " (MM:SS)"
    )


if __name__ == "__main__":
    mrr_absorption_sensor_vs_spiral("example.toml")
