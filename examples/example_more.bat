@echo off
REM %PYTHONPATH% must include "\USherbrooke\Groupe Biophotonique UdeS - Python\Développement de code\"
set PYTHONDIR=C:/Users/chap1202/AppData/Local/Programs/Python/Python310
%PYTHONDIR%/python.exe example.py --in_data_file "example.toml"
pause
echo Press any key to continue...
