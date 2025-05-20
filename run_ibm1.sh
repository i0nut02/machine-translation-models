#!/bin/bash

# Define the name for your virtual environment
VENV_NAME="venv"

# Define the name of your main Python script
PYTHON_SCRIPT="models.ibm1"

# --- Step 1: Create a virtual environment ---
echo "Checking for virtual environment..."
if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' already exists."
else
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Do you have Python 3 installed?"
        exit 1
    fi
    echo "Virtual environment created."
fi

# --- Step 2: Activate the virtual environment ---
echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated."

# --- Step 3: Install requirements (if requirements.txt exists) ---
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        deactivate # Deactivate venv before exiting on error
        exit 1
    fi
    echo "Dependencies installed."
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

# --- Step 4: Run the Python file ---
echo "Running '$PYTHON_SCRIPT'..."
python -m "$PYTHON_SCRIPT"
if [ $? -ne 0 ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' failed to run."
    deactivate # Deactivate venv before exiting on error
    exit 1
fi
echo "Script finished."

# --- Step 5: Deactivate the virtual environment ---
echo "Deactivating virtual environment..."
deactivate
echo "Done."