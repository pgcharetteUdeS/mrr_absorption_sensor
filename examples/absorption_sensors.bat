@echo off
REM The %PYTHONPATH% environmental variable must include the the mrr_absorption_sensor package parent directory
set PYTHONDIR=C:/Users/chap1202/AppData/Local/Programs/Python/Python310
%PYTHONDIR%/python.exe absorption_sensors.py --in_data_file "example.toml"
pause
echo Press any key to continue...
