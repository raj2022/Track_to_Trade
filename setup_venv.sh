#!/usr/bin/env bash
# This script sets up a virtual environment for the project.
# usage: ./setup_venv.sh


# set -e    # Exit immediately if a command exits with a non-zero status.

# PYTHON_BIN=${PYTHON_BIN:-python3}  # Use the specified Python binary or default to python3  

# ${PYTHON_BIN} -m venv .quant_work  # Create a virtual environment named 'venv'

# shellcheck disable=SC1091
source .quant_work/bin/activate  # Activate the virtual environment

# pip install --upgrade pip  # Upgrade pip to the latest version
# pip install -r requirements.txt  # Install dependencies from requirements.txt

echo ""
echo "Done. Activate with: source .quant_work/bin/activate"
echo "Verify with:          python -c \"import numpy, pandas, jax, torch, hmmlearn, arch; print('all imports OK')\""
echo "Deactivate with: deactivate"

