@echo off
REM %PYTHONPATH% must include the parent directory of the mrr_absorption_sensor package
set PYTHONDIR="C:\Program Files\Python310"
%PYTHONDIR%\python.exe example.py
pause
echo Press any key to continue...
