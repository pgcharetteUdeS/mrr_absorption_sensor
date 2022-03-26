# To run:
#   - run the script from the SAME directory as the mrr_absorption_sensor package.
# OR:
#   - run the script from anywhere, but the PYTHONDIR environment variable must contain
#     the PARENT directory of the mrr_absorption_sensor package.
#
# The colorama package enables pretty colors on the console messages, which makes the
# warnings easier to spot. On Windows a call to colorama.init() is required.
import colorama as colorama
from mrr_absorption_sensor import analyze

colorama.init()
stuff = analyze("example.toml")
