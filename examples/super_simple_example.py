import colorama as colorama
from mrr_absorption_sensor import analyze

colorama.init()
models, mrr, linear, spiral = analyze(toml_input_file_path="example.toml")
