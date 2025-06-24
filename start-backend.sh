#!/bin/bash

# Script to start the Magenta RT backend
echo "🚀 Starting Magenta RT Backend..."

# Navigate to project root
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Backend directory not found."
    exit 1
fi

# Navigate to backend and start server
echo "🔧 Starting FastAPI server..."
cd backend
python main.py
