#!/bin/bash
# Simple script to plot the latest CSV file

echo "Installing plotting dependencies..."
pip install -r scripts/requirements_plotting.txt

echo "Running plotting script..."
# Change to workspace root directory to find logs
cd "$(dirname "$0")/.."
python scripts/plot_observations_actions.py "$@"
