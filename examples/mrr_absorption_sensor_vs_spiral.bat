@echo off
REM %PYTHONPATH% must include "\USherbrooke\Groupe Biophotonique UdeS - Python\D‚veloppement de code\"
set PYTHONDIR=C:/Users/chap1202/AppData/Local/Programs/Python/Python310
REM %PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "example.toml"
REM %PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "Tableau_Pauline_h03_TE.toml"
%PYTHONDIR%/python.exe mrr_absorption_sensor_vs_spiral.py --in_data_file "Tableau_Theo_h03_TE.toml"
pause
echo Press any key to continue...
