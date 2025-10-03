#!/bin/bash
# Einfaches Wrapper-Skript zum Herunterladen von historischen Daten

# --- Farbcodes ---
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Pfade ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
DOWNLOADER_SCRIPT="scripts/download_data.py"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# --- Umgebung aktivieren ---
source "$VENV_PATH"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}            L-Bot Historien-Daten Downloader            ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# --- Interaktive Abfrage ---
read -p "Handelspaar(e) eingeben (z.B. BTC ETH SOL): " SYMBOLS
read -p "Zeitfenster eingeben (z.B. 1h 4h 1d): " TIMEFRAMES
read -p "Startdatum (JJJJ-MM-TT) [Standard: 2020-01-01]: " START_DATE
START_DATE="${START_DATE:-2020-01-01}"

echo -e "\n${CYAN}Starte Download für Symbole '${SYMBOLS}' und Zeitfenster '${TIMEFRAMES}' seit ${START_DATE}...${NC}\n"

# --- Python-Skript ausführen ---
"$VENV_PYTHON" "$DOWNLOADER_SCRIPT" \
    --symbols "$SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --start_date "$START_DATE"

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "            Download-Prozess abgeschlossen             "
echo -e "${BLUE}=======================================================${NC}"

deactivate
