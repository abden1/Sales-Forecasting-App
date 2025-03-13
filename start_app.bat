@echo off
title Sales Forecasting Application
echo Starting Sales Forecasting Application...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or newer.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check and install required packages
echo Checking required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Make sure any existing server is stopped first
echo Stopping any running server instances...
taskkill /f /im pythonw.exe >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

REM Set PYTHONPATH correctly - Making sure Python can find our modules
echo Setting environment...
set CURRENT_DIR=%CD%
set PYTHONPATH=%CURRENT_DIR%;%PYTHONPATH%
echo PYTHONPATH is now: %PYTHONPATH%

REM Show important files exist
echo Checking key files...
if exist "app\database.py" (
    echo - app\database.py: Found
) else (
    echo - app\database.py: MISSING!
)
if exist "app\data_loader.py" (
    echo - app\data_loader.py: Found
) else (
    echo - app\data_loader.py: MISSING!
)
if exist "app\app.py" (
    echo - app\app.py: Found
) else (
    echo - app\app.py: MISSING!
)
if exist "app\models\forecasting.py" (
    echo - app\models\forecasting.py: Found
) else (
    echo - app\models\forecasting.py: MISSING!
)

REM Start the application in background
echo Starting server in background...
start /B pythonw wsgi.py

REM Wait for server to start
echo Waiting for server to initialize...
timeout /t 5 /nobreak >nul

REM Open browser
echo Opening application in browser...
start http://localhost:8000

REM Display instructions
echo.
echo =====================================================
echo Application is now running in the background at http://localhost:8000
echo.
echo To stop the server later, run: taskkill /f /im pythonw.exe
echo =====================================================
echo.
echo Press any key to exit this window. The application will continue running in the background.
pause 
echo ===================================================== 