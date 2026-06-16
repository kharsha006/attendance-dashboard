@echo off
echo ===================================================
echo     STARTING LOCAL CAMERA MONITOR (CLOUD MODE)
echo ===================================================
echo.
echo Launching Camera Monitor connected to Cloud Database...

:: Change to the directory where the script is located
cd /d "%~dp0"

:: Start the Python script automatically without prompting for input
start "Camera Monitor (Cloud)" cmd /k "python monitor_only.py"

echo.
echo ===================================================
echo   CAMERA MONITOR LAUNCHED SUCCESSFULLY!
echo   Face detections are now being processed.
echo ===================================================
