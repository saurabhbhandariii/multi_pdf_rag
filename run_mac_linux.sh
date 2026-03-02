#!/bin/bash
echo "========================================"
echo "  Multi-PDF RAG - Quick Start (Mac/Linux)"
echo "========================================"
echo ""

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Run
echo ""
echo "Starting app at http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""
streamlit run app.py
