#!/bin/bash

echo "Setting up Meesho E-commerce Data Analysis Project..."
echo

# Check if virtual environment exists
if [ ! -d "env_dice" ]; then
    echo "Creating virtual environment..."
    python -m venv env_dice
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env_dice/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

echo
echo "Setup complete! Virtual environment is now active."
echo
echo "To run the project:"
echo "  1. python dataset.py          (Generate synthetic dataset)"
echo "  2. python dataset_preprocess.py (Basic preprocessing)"
echo "  3. python data_prepro.py      (Advanced MFE analysis)"
echo
echo "To deactivate virtual environment: deactivate"
echo
