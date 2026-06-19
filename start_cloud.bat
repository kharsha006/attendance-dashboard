@echo off
title Camera Monitor (Cloud Auto Setup)

echo ===================================================
echo        CAMERA MONITOR AUTO STARTER
echo ===================================================
echo.

cd /d "%~dp0"

set PYTHON=venv\Scripts\python.exe

echo Checking virtual environment...

if not exist "%PYTHON%" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Upgrading pip...
%PYTHON% -m pip install --upgrade pip

echo Installing dependencies from requirements.txt...
%PYTHON% -m pip install -r requirements.txt

echo.
echo ===================================================
echo Starting Camera Monitor...
echo ===================================================
echo.

start "Camera Monitor" cmd /k "%PYTHON% monitor_only.py"

exit