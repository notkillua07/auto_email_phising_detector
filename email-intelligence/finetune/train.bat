@echo off
REM Launcher for fine-tuning using the conda environment at D:\miniconda3\envs\mlenv
REM Run this instead of "python train.py"

set CONDA_PYTHON=D:\miniconda3\envs\mlenv\python.exe
set SCRIPT_DIR=%~dp0

echo Using Python: %CONDA_PYTHON%
echo Working dir: %SCRIPT_DIR%
echo.

"%CONDA_PYTHON%" "%SCRIPT_DIR%train.py"