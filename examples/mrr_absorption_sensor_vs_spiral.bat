@echo off
REM %PYTHONPATH% must include "\USherbrooke\Groupe Biophotonique UdeS - Python\D‚veloppement de code\"
set PYTHONDIR=C:/Users/chap1202/AppData/Local/Programs/Python/Python310
%PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "example.toml"
REM %PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "Tableau_TE_h03.toml"
REM %PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "Tableau_TE_h03_w05_min.toml"
pause
echo Press any key to continue...
