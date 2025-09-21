@echo off
echo Setting up Meesho E-commerce Data Analysis Project...
echo.

REM Check if virtual environment exists
if not exist "env_dice" (
    echo Creating virtual environment...
    python -m venv env_dice
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .\env_dice\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Setup complete! Virtual environment is now active.
echo.
echo To run the project:
echo   1. python dataset.py          (Generate synthetic dataset)
echo   2. python dataset_preprocess.py (Basic preprocessing)
echo   3. python data_prepro.py      (Advanced MFE analysis)
echo.
echo To deactivate virtual environment: deactivate
echo.
pause
