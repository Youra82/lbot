#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}======================================================="
echo "      JaegerBot Automatisierte Trainings-Pipeline"
echo -e "=======================================================${NC}"

# --- Pfade definieren ---
VENV_PATH=".venv/bin/activate"
INSTALL_SCRIPT="install.sh"
TRAINER="src/jaegerbot/analysis/trainer.py"
OPTIMIZER="src/jaegerbot/analysis/optimizer.py"

# --- Automatischer Check der virtuellen Umgebung ---
if [ ! -f "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtuelle Umgebung nicht gefunden. Starte die Installation...${NC}"
    if [ -f "$INSTALL_SCRIPT" ]; then
        chmod +x "$INSTALL_SCRIPT"
        bash ./install.sh
    else
        echo -e "${RED}Fehler: install.sh nicht im selben Verzeichnis gefunden. Breche ab.${NC}"
        exit 1
    fi
fi

# --- Virtuelle Umgebung aktivieren ---
source "$VENV_PATH"
echo -e "${GREEN}✔ Virtuelle Umgebung wurde erfolgreich aktiviert.${NC}"

# --- Interaktive Abfrage der Parameter ---
read -p "Handelspaar(e) eingeben (ohne /USDT, z.B. BTC ETH): " SYMBOLS
read -p "Zeitfenster eingeben (z.B. 1h 4h): " TIMEFRAMES
read -p "Startdatum für den Datendownload (JJJJ-MM-TT): " START_DATE
read -p "Enddatum für den Datendownload (JJJJ-MM-TT): " END_DATE
read -p "Startkapital in USDT eingeben (Standard: 1000): " START_CAPITAL
START_CAPITAL=${START_CAPITAL:-1000}
read -p "Mit wie vielen CPU-Kernen soll optimiert werden? (Standard: 1): " N_CORES
N_CORES=${N_CORES:-1}
read -p "Anzahl zu testender Strategien (Standard: 200): " N_TRIALS
N_TRIALS=${N_TRIALS:-200}

echo ""
echo -e "${YELLOW}Wähle einen Optimierungs-Modus:${NC}"
echo "  1) Strenger Modus (alle Regeln müssen erfüllt sein)"
echo "  2) 'Finde das Beste'-Modus (nur max. Drawdown-Regel zählt)"
read -p "Auswahl (1-2, Standard: 1): " OPTIM_MODE
OPTIM_MODE=${OPTIM_MODE:-1}

if [ "$OPTIM_MODE" == "1" ]; then
    OPTIM_MODE_ARG="strict"
    read -p "Maximalen Drawdown in Prozent eingeben (z.B. 25): " MAX_DD
    MAX_DD=${MAX_DD:-30}
    read -p "Minimale Win-Rate in Prozent eingeben (z.B. 60): " MIN_WR
    MIN_WR=${MIN_WR:-55}
    read -p "Minimalen Gesamtgewinn (PnL) in % eingeben (z.B. 1000): " MIN_PNL
    MIN_PNL=${MIN_PNL:-0}
    
    echo -e "\n${YELLOW}Pipeline wird im 'Strengen Modus' ausgeführt:${NC}"
    echo -e "- Bedingung 1: Maximaler Drawdown < ${MAX_DD}%"
    echo -e "- Bedingung 2: Minimale Win-Rate > ${MIN_WR}%"
    echo -e "- Bedingung 3: Minimaler Profit > ${MIN_PNL}%"
else
    OPTIM_MODE_ARG="best_profit"
    read -p "Maximalen Drawdown in Prozent eingeben (z.B. 25): " MAX_DD
    MAX_DD=${MAX_DD:-30}
    MIN_WR=0
    MIN_PNL=-99999
    
    echo -e "\n${YELLOW}Pipeline wird im 'Finde das Beste'-Modus ausgeführt:${NC}"
    echo -e "- Einzige Bedingung: Maximaler Drawdown < ${MAX_DD}%"
fi

# --- Pipeline starten ---
echo -e "\n${GREEN}>>> STUFE 1/2: Starte finales Modelltraining...${NC}"
python3 "$TRAINER" --symbols "$SYMBOLS" --timeframes "$TIMEFRAMES" --start_date "$START_DATE" --end_date "$END_DATE"

echo -e "\n${GREEN}>>> STUFE 2/2: Starte Handelsparameter-Optimierung...${NC}"
python3 "$OPTIMIZER" \
    --symbols "$SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --start_date "$START_DATE" \
    --end_date "$END_DATE" \
    --jobs "$N_CORES" \
    --max_drawdown "$MAX_DD" \
    --start_capital "$START_CAPITAL" \
    --min_win_rate "$MIN_WR" \
    --trials "$N_TRIALS" \
    --min_pnl "$MIN_PNL" \
    --mode "$OPTIM_MODE_ARG"

deactivate
echo -e "\n${BLUE}✔ Pipeline erfolgreich abgeschlossen!${NC}"
