#!/bin/bash
# Interaktives Skript zur Auswahl und Aktivierung der besten Handelsstrategien

# --- Farbcodes ---
BLUE='\033[0;34m'
NC='\033[0m'

# --- Pfade ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
SELECTOR_SCRIPT="src/lbot/analysis/result_selector.py"

# --- Umgebung aktivieren ---
source "$VENV_PATH"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}        L-Bot Strategie-Aktivator gestartet            ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Python-Skript für die interaktive Auswahl ausführen
python3 "$SELECTOR_SCRIPT"

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "         Strategie-Aktivator beendet             "
echo -e "${BLUE}=======================================================${NC}"

deactivate
