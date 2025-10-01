# src/lbot/analysis/show_results.py
import os
import sys
import json
import pandas as pd
from datetime import date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.analysis.backtester import Backtester
from lbot.utils.exchange import Exchange
from lbot.utils.data_handler import get_market_data
from lbot.utils.lstm_model import create_ann_features, load_model_and_scaler

def run_backtest_for_config(config, start_date, end_date, start_capital, settings):
    """ Führt einen einzelnen Backtest für eine gegebene Konfiguration durch. """
    symbol = config['market']['symbol']
    timeframe = config['market']['timeframe']
    
    print(f"\nAnalysiere Ergebnisse für: {symbol} ({timeframe})...")

    # Lade Daten
    dummy_exchange = Exchange({'apiKey': 'dummy', 'secret': 'dummy'})
    data_raw = get_market_data(dummy_exchange, symbol, timeframe, start_date)
    data_for_backtest = data_raw[start_date:end_date]
    if data_for_backtest.empty:
        print(f"Konnte keine Daten für {symbol} im Zeitraum {start_date}-{end_date} laden. Überspringe.")
        return None

    data_with_features = create_ann_features(data_for_backtest[['open', 'high', 'low', 'close', 'volume']])
    
    # Lade Modell
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    if not model or not scaler:
        print(f"Modell/Scaler für {symbol} nicht gefunden. Überspringe.")
        return None

    # Führe Backtest aus
    backtest_settings = settings.get('backtest_settings', {})
    backtester = Backtester(
        data=data_with_features.copy(),
        model=model,
        scaler=scaler,
        params=config,
        start_capital=start_capital,
        sequence_length=settings['model_settings']['sequence_length'],
        fee_rate_pct=backtest_settings.get('fee_rate_pct', 0.06),
        slippage_pct=backtest_settings.get('slippage_pct', 0.02)
    )
    result = backtester.run()
    
    # Berechne Endkapital für die Zusammenfassung
    end_capital = start_capital * (1 + result['total_pnl_pct'] / 100)
    
    return {
        "Strategie": f"{symbol} ({timeframe})",
        "PnL (%)": result['total_pnl_pct'],
        "Max DD (%)": result['max_drawdown_pct'],
        "Win-Rate (%)": result['win_rate'],
        "Trades": result['num_trades'],
        "Endkapital": end_capital
    }

def main():
    print("--- L-Bot Ergebnis-Analyse ---")
    
    # Lade globale Einstellungen
    settings_path = os.path.join(PROJECT_ROOT, 'settings.json')
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    # Interaktive Abfrage
    print("\n--- Bitte Konfiguration für den Backtest festlegen ---")
    default_start_date = "2023-01-01"
    start_date_input = input(f"Startdatum (JJJJ-MM-TT) eingeben [Standard: {default_start_date}]: ")
    start_date = start_date_input if start_date_input else default_start_date

    default_end_date = date.today().strftime("%Y-%m-%d")
    end_date_input = input(f"Enddatum (JJJJ-MM-TT) eingeben [Standard: {default_end_date}]: ")
    end_date = end_date_input if end_date_input else default_end_date

    default_capital = settings['optimization_settings'].get('start_capital', 1000)
    try:
        start_capital_input = input(f"Startkapital in USDT eingeben [Standard: {default_capital}]: ")
        start_capital = int(start_capital_input) if start_capital_input else default_capital
    except ValueError:
        print(f"Ungültige Eingabe, verwende Standardwert: {default_capital} USDT")
        start_capital = default_capital
    
    print("--------------------------------------------------")

    configs_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    if not os.path.isdir(configs_dir):
        print(f"Fehler: Konfigurations-Ordner nicht gefunden: {configs_dir}")
        return

    all_results = []
    for filename in os.listdir(configs_dir):
        if filename.startswith('config_') and filename.endswith('.json'):
            config_path = os.path.join(configs_dir, filename)
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            result_summary = run_backtest_for_config(config, start_date, end_date, start_capital, settings)
            if result_summary:
                all_results.append(result_summary)

    if not all_results:
        print("\nKeine gültigen Konfigurationen zum Analysieren gefunden.")
        return

    # Erstelle einen sauberen DataFrame und sortiere ihn
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="PnL (%)", ascending=False)

    print("\n\n=======================================================")
    print(f"     Zusammenfassung (Startkapital: {start_capital} USDT)")
    print("=======================================================")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(results_df.to_string(index=False))
    print("=======================================================")

if __name__ == "__main__":
    main()
