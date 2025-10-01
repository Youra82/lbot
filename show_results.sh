#!/bin/bash
VENV_PATH=".venv/bin/activate"
RESULTS_SCRIPT="src/lbot/analysis/show_results.py"

# Umgebung aktivieren und Skript ausführen
source "$VENV_PATH"
python3 "$RESULTS_SCRIPT"
deactivate
