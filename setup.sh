#!/bin/bash

# Setup script for AI-Powered Test Case Generator

set -e  # Exit on error

echo "🧪 AI-Powered Test Case Generator - Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "❌ Error: Python $required_version or higher is required."
    echo "   Current version: $python_version"
    exit 1
fi
echo "  Python version: $python_version ✓"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created ✓"
else
    echo "  Virtual environment already exists ✓"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "  Activated ✓"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "  pip upgraded ✓"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "  Dependencies installed ✓"
echo ""

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo "  NLTK data downloaded ✓"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p exports
mkdir -p examples
echo "  Directories created ✓"
echo ""

# Check for .env file
echo "🔑 Checking configuration..."
if [ ! -f ".env" ]; then
    echo "  Creating .env from template..."
    cp .env.example .env
    echo "  ⚠️  Please edit .env and add your OPENROUTER_API_KEY"
else
    echo "  .env file exists ✓"
fi
echo ""

# Final instructions
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenRouter API key"
echo "2. Run the application with: streamlit run app.py"
echo "3. Or activate venv and run: source venv/bin/activate && streamlit run app.py"
echo ""
echo "For more information, see README.md"
echo ""
echo "Happy testing! 🧪"
