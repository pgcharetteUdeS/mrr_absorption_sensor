@echo off
REM %PYTHONPATH% must include the parent directory of the mrr_absorption_sensor package
set PYTHONDIR=C:/Users/chap1202/AppData/Local/Programs/Python/Python310
%PYTHONDIR%/python.exe example_more.py --in_data_file "example.toml"
pause
echo Press any key to continue...
