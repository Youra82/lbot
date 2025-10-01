#!/bin/bash
# Interaktive Pipeline zum Trainieren neuer L-Bot LSTM-Modelle

VENV_PATH=".venv/bin/activate"
source "$VENV_PATH"

echo "======================================================="
echo "       L-Bot LSTM Trainings-Pipeline (Interaktiv)      "
echo "======================================================="

read -p "Handelspaar(e) eingeben (ohne /USDT, z.B. BTC ETH): " SYMBOLS
read -p "Zeitfenster eingeben (z.B. 1h 4h): " TIMEFRAMES
# Weitere Abfragen könnten hier hinzugefügt werden

echo ">>> Starte Modelltraining für L-Bot... <<<"
python3 src/lbot/analysis/trainer.py --symbols "$SYMBOLS" --timeframes "$TIMEFRAMES"

echo "✔ Pipeline erfolgreich abgeschlossen!"
deactivate
