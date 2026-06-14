@echo off
echo ===================================================
echo     STARTING LOCAL CAMERA MONITOR (CLOUD MODE)
echo ===================================================
echo.

if "%DATABASE_URL%"=="" (
    echo [WARNING] No DATABASE_URL found.
    set /p DATABASE_URL="Please paste your Render PostgreSQL External URL: "
)

echo.
echo [1/1] Launching Camera Monitor connected to Cloud Database...
start "Camera Monitor (Cloud)" cmd /k "set DATABASE_URL=%DATABASE_URL% && python monitor_only.py"

echo.
echo ===================================================
echo   CAMERA MONITOR LAUNCHED SUCCESSFULLY!
echo   Face detections will now be sent directly to 
echo   your Cloud Dashboard.
echo ===================================================
pause
