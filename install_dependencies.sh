#!/bin/bash

# Script to set up a Python virtual environment and install dependencies.

set -e  # Exit immediately if a command exits with a non-zero status.
set -o pipefail  # Fail if any part of a pipe command fails.

echo "Starting the installation process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install it first."
    exit 1
fi

# Create a virtual environment (if not already created)
VENV_DIR="env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating a Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."

# Install the required dependencies
pip install --upgrade pip
pip install transformers
pip install python-dotenv
pip install bitsandbytes
pip install 'accelerate>=0.26.0'
pip install nvidia-ml-py
pip install openai

echo "All dependencies have been installed successfully."

# Deactivate the virtual environment
deactivate

echo "Installation complete. To activate the virtual environment, run:"
echo "source $VENV_DIR/bin/activate"
