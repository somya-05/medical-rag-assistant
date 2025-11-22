#!/bin/bash

# Launch script for Medical Symptom Diagnosis System

echo "=================================="
echo "Medical Diagnosis System Launcher"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check for API key
if [ -z "$GROQ_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: GROQ_API_KEY not set!"
    echo ""
    echo "Please set your Groq API key:"
    echo "  export GROQ_API_KEY='your-key-here'"
    echo ""
    echo "Or create a .env file with:"
    echo "  GROQ_API_KEY=your-key-here"
    echo ""
    read -p "Press Enter to continue anyway (will fail at diagnosis) or Ctrl+C to exit..."
fi

echo ""
echo "Launching web interface..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

.venv/bin/streamlit run app.py
