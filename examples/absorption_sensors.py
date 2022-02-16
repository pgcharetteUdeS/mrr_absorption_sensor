"""

FILENAME: absorption_sensors.py

PURPOSE: Determine maximum achievable sensitivity as a function of waveguide
         core height at a fixed waveguide core width over a range of ring/spiral
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
import colorama as colorama
from colorama import Fore, Style
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import warnings

# mrr_absorption_sensor package
from mrr_absorption_sensor import analyze


def absorption_sensors_analysis(toml_input_filename: str):
    """

    :param toml_input_filename:
    :return:
    """
    # Start execution timer
    global_start_time: float = time.perf_counter()

    # Initialize the colorama package to print colored text to the Windows console
    colorama.init()

    # matplotlib initializations
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.size": 6,
            "axes.grid": True,
            "axes.grid.which": "both",
            "lines.linewidth": 0.5,
            "axes.linewidth": 0.5,
        },
    )
    plt.ion()

    # Define the logger for console information messages
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(funcName)s():%(message)s"))
    logger.addHandler(stream_handler)

    # Parser for command line parameter input (ex: running from .bat file)
    parser = argparse.ArgumentParser(description="Analyse waveguide sensors")
    parser.add_argument("--in_data_file", type=str)
    parser.add_argument("--no_pause", action="store_true")
    args = parser.parse_args()

    # Load input .toml file name from the command line if supplied, else use filename
    # passed as a parameter to the function. Verify that the file exists.
    toml_input_file_path: Path = Path(
        toml_input_filename if args.in_data_file is None else args.in_data_file
    )
    if not toml_input_file_path.is_file():
        print(
            f"{Fore.YELLOW}File '{toml_input_file_path}' "
            + f"does not exist!{Style.RESET_ALL}"
        )
        sys.exit()

    # Run the sensor analysis
    analyze(toml_input_file_path=toml_input_file_path, logger=logger.info)

    # Show execution time
    elapsed_time: float = time.perf_counter() - global_start_time
    print(
        f"{Fore.BLUE}Execution time: "
        + f"{time.strftime('%M:%S', time.gmtime(elapsed_time))} (MM:SS)"
        + f"{Style.RESET_ALL}"
    )

    # If running from the command line, pause for user input to keep figures visible.
    # If running in PyCharm debugger, skip this else figures don't render!
    if sys.gettrace() is not None:
        print("Breakpoint here to keep figures visible in IDE!")
    elif not args.no_pause:
        input("Script paused to display figures, press any key to exit")
    else:
        print(f"{Fore.BLUE}Done!{Style.RESET_ALL}")


if __name__ == "__main__":
    # TEMPORARY: suppress deprecation warnings caused by matplotlib bug, with message:
    # "Auto-removal of grids by pcolor() and pcolormesh() is deprecated..."
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # input_data_filename: str = "example_modes.toml"
    absorption_sensors_analysis("example.toml")