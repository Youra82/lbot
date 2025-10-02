#!/bin/bash
# Interaktive Pipeline zum Trainieren UND Optimieren neuer L-Bot LSTM-Modelle

# --- Farbcodes für eine schönere Ausgabe ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

# --- Pfade und Skripte ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
SETTINGS_FILE="$SCRIPT_DIR/settings.json"
TRAINER="src/lbot/analysis/trainer.py"
OPTIMIZER="src/lbot/analysis/optimizer.py"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# --- Umgebung aktivieren ---
source "$VENV_PATH"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}    L-Bot Interaktive Trainings- & Optimierungs-Pipeline   ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# --- Python-Helper zum Auslesen der Standardwerte aus settings.json ---
get_setting() {
    "$VENV_PYTHON" -c "import json; f=open('$SETTINGS_FILE'); print(json.load(f)$1); f.close()"
}

# --- Standardwerte aus settings.json laden ---
DEFAULT_SYMBOLS=$(get_setting "['optimization_settings']['symbols_to_optimize']" | tr -d "[]',\"")
DEFAULT_TIMEFRAMES=$(get_setting "['optimization_settings']['timeframes_to_optimize']" | tr -d "[]',\"")
DEFAULT_LOOKBACK=$(get_setting "['optimization_settings']['lookback_days']")
DEFAULT_TRIALS=$(get_setting "['optimization_settings']['num_trials']")
DEFAULT_JOBS=$(get_setting "['optimization_settings']['cpu_cores']")

# --- Interaktive Abfrage der Schlüsselwerte mit integrierter Automatik-Option ---
echo -e "\n${CYAN}--- Bitte Parameter für den Lauf festlegen ---${NC}"
echo -e "${CYAN}(Einfach Enter drücken, um den vorgeschlagenen Wert zu übernehmen)${NC}"

read -p "Handelspaar(e) eingeben [Vorschlag: ${DEFAULT_SYMBOLS}]: " INPUT_SYMBOLS
SYMBOLS="${INPUT_SYMBOLS:-$DEFAULT_SYMBOLS}"

read -p "Zeitfenster eingeben [Vorschlag: ${DEFAULT_TIMEFRAMES}]: " INPUT_TIMEFRAMES
TIMEFRAMES="${INPUT_TIMEFRAMES:-$DEFAULT_TIMEFRAMES}"

# NEU: Integrierte Automatik-Wahl beim Daten-Rückblick
read -p "Daten-Rückblick in Tagen (für Automatik 'a' eingeben) [Vorschlag: ${DEFAULT_LOOKBACK}]: " INPUT_LOOKBACK

if [[ "$INPUT_LOOKBACK" == "a" || "$INPUT_LOOKBACK" == "A" ]]; then
    echo -e "${YELLOW}Berechne optimalen Daten-Rückblick...${NC}"
    LOOKBACK=0
    for tf in $TIMEFRAMES; do
        current_lookback=0
        case $tf in
            5m)  current_lookback=30 ;;
            15m) current_lookback=90 ;;
            30m) current_lookback=180 ;;
            1h)  current_lookback=365 ;;
            2h)  current_lookback=540 ;;
            4h)  current_lookback=730 ;;
            6h)  current_lookback=1095 ;;
            1d)  current_lookback=1825 ;;
            *)   current_lookback=$DEFAULT_LOOKBACK ;;
        esac
        if [[ $current_lookback -gt $LOOKBACK ]]; then
            LOOKBACK=$current_lookback
        fi
    done
    echo -e "${GREEN}✔ Automatischer Daten-Rückblick für '${TIMEFRAMES}' auf ${LOOKBACK} Tage gesetzt.${NC}"
else
    # Standardverhalten, wenn nicht 'a' eingegeben wird
    LOOKBACK="${INPUT_LOOKBACK:-$DEFAULT_LOOKBACK}"
fi

read -p "Anzahl der Optimierungs-Versuche (Trials) [Vorschlag: ${DEFAULT_TRIALS}]: " INPUT_TRIALS
TRIALS="${INPUT_TRIALS:-$DEFAULT_TRIALS}"

read -p "Anzahl der CPU-Kerne (-1 für alle) [Vorschlag: ${DEFAULT_JOBS}]: " INPUT_JOBS
JOBS="${INPUT_JOBS:-$DEFAULT_JOBS}"

# --- Berechnungen und Zusammenfassung ---
START_DATE=$(date -d "$LOOKBACK days ago" +%F)

echo -e "\n${YELLOW}-------------------------------------------------------${NC}"
echo -e "${GREEN}✅ Pipeline wird mit folgenden Werten gestartet:${NC}"
echo -e "   - Symbole:          ${CYAN}${SYMBOLS}${NC}"
echo -e "   - Zeitfenster:      ${CYAN}${TIMEFRAMES}${NC}"
echo -e "   - Daten-Startdatum: ${CYAN}${START_DATE} (${LOOKBACK} Tage zurück)${NC}"
echo -e "   - Trials:           ${CYAN}${TRIALS}${NC}"
echo -e "   - CPU-Kerne:        ${CYAN}${JOBS}${NC}"
echo -e "${YELLOW}-------------------------------------------------------${NC}\n"
read -p "Soll der Prozess gestartet werden? (j/n): " confirm && [[ $confirm == [jJ] || $confirm == [jJ][aA] ]] || exit 1


# --- Pipeline starten ---
echo -e "\n${BLUE}>>> STUFE 1/2: Starte L-Bot LSTM-Modelltraining... <<<${NC}"
"$VENV_PYTHON" "$TRAINER" \
    --symbols "$SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --start_date "$START_DATE"

if [ $? -ne 0 ]; then
    echo -e "\n${RED}Fehler im Trainer-Skript. Pipeline wird abgebrochen.${NC}"
    deactivate
    exit 1
fi

echo -e "\n${BLUE}>>> STUFE 2/2: Starte Handelsparameter-Optimierung... <<<${NC}"

if [[ "$JOBS" -ne 1 ]]; then
    display_jobs=$JOBS
    if [[ "$JOBS" -eq -1 ]]; then
        display_jobs="allen"
    fi
    echo -e "${YELLOW}Initialisiere Parallelverarbeitung auf ${display_jobs} Kernen... Dies kann je nach System bis zu einer Minute dauern...${NC}"
fi

export JOBLIB_START_METHOD=spawn
"$VENV_PYTHON" "$OPTIMIZER" \
    --symbols "$SYMBOLS" \
    --timeframes "$TIMEFRAMES" \
    --start_date "$START_DATE" \
    --trials "$TRIALS" \
    --jobs "$JOBS"

if [ $? -ne 0 ]; then
    echo -e "\n${RED}Fehler im Optimizer-Skript. Pipeline wird abgebrochen.${NC}"
    deactivate
    exit 1
fi

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "      ✅ Pipeline erfolgreich abgeschlossen!      "
echo -e "${GREEN}=======================================================${NC}"

deactivate
