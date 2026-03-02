@echo off
echo ========================================
echo   Multi-PDF RAG - Quick Start (Windows)
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b
)

:: Create venv if not exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate
call venv\Scripts\activate

:: Install deps
echo Installing dependencies...
pip install -r requirements.txt --quiet

:: Run
echo.
echo Starting app at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py
