@echo off
REM Script to start the Magenta RT backend on Windows

echo 🚀 Starting Magenta RT Backend...

REM Navigate to project root
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if backend directory exists
if not exist "backend" (
    echo ❌ Backend directory not found.
    pause
    exit /b 1
)

REM Navigate to backend and start server
echo 🔧 Starting FastAPI server...
cd backend
python main.py
