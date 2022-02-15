@echo off
REM The %PYTHONPATH% environmental variable must include the the mrr_absorption_sensor package parent directory
python.exe absorption_sensors.py --in_data_file "example.toml"
pause
echo Press any key to continue...
