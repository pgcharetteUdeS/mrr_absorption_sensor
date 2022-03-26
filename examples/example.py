# To run:
#   - run the script from the SAME directory as the mrr_absorption_sensor package.
# OR:
#   - run the script from anywhere, but the PYTHONDIR environment variable must contain
#     the PARENT directory of the mrr_absorption_sensor package.
#
from mrr_absorption_sensor import analyze

analyze("example.toml")
