# src/jaegerbot/analysis/show_results.py
import os
import sys
import json
import pandas as pd
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.analysis.backtester import load_data, run_ann_backtest

def main():
    print("--- JaegerBot Ergebnis-Analyse ---")
    
    # --- NEU: Interaktive Abfrage für den Backtest-Zeitraum und das Kapital ---
    print("\n--- Bitte Konfiguration für den Backtest festlegen ---")

    # Abfrage Startdatum mit Standardwert
    default_start_date = "2023-01-01"
    start_date_input = input(f"Startdatum (JJJJ-MM-TT) eingeben [Standard: {default_start_date}]: ")
    start_date = start_date_input if start_date_input else default_start_date

    # Abfrage Enddatum mit dem heutigen Datum als Standardwert
    default_end_date = date.today().strftime("%Y-%m-%d")
    end_date_input = input(f"Enddatum (JJJJ-MM-TT) eingeben [Standard: {default_end_date}]: ")
    end_date = end_date_input if end_date_input else default_end_date

    # Abfrage Startkapital mit Standardwert und Fehlerbehandlung
    default_capital = 1000
    try:
        start_capital_input = input(f"Startkapital in USDT eingeben [Standard: {default_capital}]: ")
        start_capital = int(start_capital_input) if start_capital_input else default_capital
    except ValueError:
        print(f"Ungültige Eingabe, verwende Standardwert: {default_capital} USDT")
        start_capital = default_capital
    
    print("--------------------------------------------------")
    # --- ENDE DER ÄNDERUNG ---

    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
    if not os.path.isdir(configs_dir):
        print(f"Fehler: Konfigurations-Ordner nicht gefunden: {configs_dir}")
        return

    all_results = []

    # Gehe durch alle gefundenen Konfigurationsdateien
    for filename in os.listdir(configs_dir):
        if filename.startswith('config_') and filename.endswith('.json'):
            config_path = os.path.join(configs_dir, filename)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            symbol = config['market']['symbol']
            timeframe = config['market']['timeframe']
            
            print(f"\nAnalysiere Ergebnisse für: {symbol} ({timeframe})...")

            # Lade die Daten
            data = load_data(symbol, timeframe, start_date, end_date)
            if data.empty:
                print(f"Konnte keine Daten für {symbol} laden. Überspringe.")
                continue

            # Lade das passende Modell
            safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
            model_paths = {
                'model': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5'),
                'scaler': os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
            }

            # Führe den Backtest mit den Parametern aus der Config-Datei durch
            params = {**config['strategy'], **config['risk']}
            result = run_ann_backtest(data.copy(), params, model_paths, start_capital)
            
            # Speichere das Ergebnis
            result_summary = {
                "Strategie": f"{symbol} ({timeframe})",
                "PnL (%)": result['total_pnl_pct'],
                "Max DD (%)": result['max_drawdown_pct'] * 100,
                "Win-Rate (%)": result['win_rate'],
                "Trades": result['trades_count'],
                "Endkapital": result['end_capital']
            }
            all_results.append(result_summary)

    if not all_results:
        print("\nKeine gültigen Konfigurationen zum Analysieren gefunden.")
        return

    # Erstelle einen sauberen DataFrame und sortiere ihn nach Profit
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="PnL (%)", ascending=False)

    print("\n\n=======================================================")
    print(f"      Zusammenfassung (Startkapital: {start_capital} USDT)")
    print("=======================================================")
    # Formatiere die Ausgabe für bessere Lesbarkeit
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(results_df.to_string(index=False))
    print("=======================================================")

if __name__ == "__main__":
    main()
