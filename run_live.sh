#!/bin/bash

# Pfade definieren
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"
SETTINGS_FILE="$SCRIPT_DIR/settings.json"
BOT_RUNNER="$SCRIPT_DIR/src/jaegerbot/strategy/run.py" # <-- DIESE ZEILE WURDE KORRIGIERT

# Virtuelle Umgebung aktivieren
source "$VENV_PATH"

echo "======================================================="
echo "JaegerBot Master Runner wird gestartet..."

# Lese die Einstellung für den Auto-Modus
USE_AUTO_RESULTS=$(python3 -c "import json; f=open('$SETTINGS_FILE'); print(str(json.load(f).get('live_trading_settings', {}).get('use_auto_optimizer_results', False)).lower()); f.close()")

if [ "$USE_AUTO_RESULTS" == "true" ]; then
    echo "Modus: Autopilot. Lese Strategien aus den Optimierungs-Ergebnissen..."
    STRATEGY_SOURCE_FILE="$SCRIPT_DIR/artifacts/results/optimization_results.json"
    STRATEGY_KEY="approved_strategies"
else
    echo "Modus: Manuell. Lese Strategien aus den manuellen Einstellungen..."
    STRATEGY_SOURCE_FILE="$SETTINGS_FILE"
    STRATEGY_KEY="active_strategies"
fi
echo "======================================================="

# Lese und starte jede Strategie aus der gewählten Quelle
# KORRIGIERTER PYTHON-BEFEHL
python3 -c "
import json
import sys
try:
    with open('$STRATEGY_SOURCE_FILE', 'r') as f:
        config = json.load(f)
    
    if '$STRATEGY_KEY' == 'active_strategies':
        strategies = config.get('live_trading_settings', {}).get('$STRATEGY_KEY', [])
    else:
        strategies = config.get('$STRATEGY_KEY', [])
        
    for strategy in strategies:
        # Gib Symbol, Zeitfenster und Budget sauber getrennt aus
        print(strategy['symbol'], strategy['timeframe'], strategy.get('budget_usdt', 0))
except FileNotFoundError:
    print(f'Warnung: Strategie-Datei ($STRATEGY_SOURCE_FILE) nicht gefunden.', file=sys.stderr)
" | while read -r symbol timeframe budget; do
    echo ""
    echo "--- Starte Bot für: ${symbol} (${timeframe}) mit Budget ${budget} USDT ---"
    # Übergebe die sauberen Variablen an run.py
    python3 "$BOT_RUNNER" --symbol "$symbol" --timeframe "$timeframe" --budget "$budget"
done

echo ""
echo "======================================================="
echo "JaegerBot Master Runner hat alle Aufgaben abgeschlossen."
echo "======================================================="

deactivate
