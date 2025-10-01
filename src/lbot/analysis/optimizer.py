# src/lbot/analysis/optimizer.py
import os
import sys
import argparse
import json
import pandas as pd
import optuna
import logging
from joblib import Parallel, delayed

# Pfad-Konfiguration
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, load_model_and_scaler
from lbot.analysis.backtester import Backtester

# Logging-Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Globale Variablen für Daten, Modell und Scaler, um sie nicht in jedem Trial neu zu laden
DATA = None
MODEL = None
SCALER = None

def objective(trial):
    """ Die Optuna-Zielfunktion, die für jeden Versuch aufgerufen wird. """
    
    # 1. Definiere die zu testenden Parameter
    params = {
        "strategy": {
            "prediction_threshold": trial.suggest_float("prediction_threshold", 0.51, 0.75),
        },
        "risk": {
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 5.0),
            "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.0, 5.0),
            "leverage": trial.suggest_int("leverage", 1, 20),
            "margin_mode": "isolated"
        },
        "behavior": {
            "use_longs": True,
            "use_shorts": False # Vereinfachung: Optimierer testet nur Longs
        }
    }

    # 2. Führe den Backtest mit diesen Parametern aus
    backtester = Backtester(DATA, MODEL, SCALER, params)
    metrics = backtester.run()

    # 3. Überprüfe die Bedingungen (Constraints)
    if metrics['max_drawdown_pct'] > MAX_DRAWDOWN:
        # Harte Bedingung: Wenn der Drawdown zu hoch ist, ist der Versuch ungültig.
        return -999.0 # Gib einen sehr schlechten Wert zurück

    # 4. Definiere das Optimierungsziel
    # Wir wollen den Profit maximieren, aber Drawdown bestrafen.
    # Ein einfacher Score könnte sein: Profit / (Drawdown + 1)
    score = metrics['total_pnl_pct'] / (metrics['max_drawdown_pct'] + 1)
    
    return score if not pd.isna(score) else -999.0

def run_optimization_for_pair(symbol, timeframe, start_date, end_date, trials, jobs):
    """ Startet die komplette Optimierung für ein Handelspaar. """
    global DATA, MODEL, SCALER, MAX_DRAWDOWN
    
    logging.info(f"Starte Optimierung für {symbol} ({timeframe})...")

    # Lade Daten, Modell und Scaler einmalig
    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    data_raw = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=5000)
    DATA = create_ann_features(data_raw[['open', 'high', 'low', 'close', 'volume']])
    
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)

    if MODEL is None or SCALER is None:
        logging.error(f"Modell/Scaler für {symbol} ({timeframe}) nicht gefunden. Überspringe.")
        return None

    MAX_DRAWDOWN = 30.0 # Harter Filter, kann als Arg übergeben werden

    # Starte die Optuna-Studie
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, n_jobs=jobs)
    
    best_params = study.best_trial.params
    logging.info(f"Beste Parameter für {symbol} ({timeframe}) gefunden: {best_params}")
    logging.info(f"Bester Score: {study.best_trial.value}")

    # Speichere die beste Konfiguration
    config_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_{safe_filename}.json')
    
    final_config = {
        "market": {"symbol": symbol, "timeframe": timeframe},
        "strategy": {"prediction_threshold": best_params['prediction_threshold']},
        "risk": {
            "risk_per_trade_pct": best_params['risk_per_trade_pct'],
            "risk_reward_ratio": best_params['risk_reward_ratio'],
            "leverage": best_params['leverage'],
            "margin_mode": "isolated"
        },
        "behavior": {"use_longs": True, "use_shorts": False}
    }

    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=4)
        
    logging.info(f"Beste Konfiguration gespeichert in: {config_path}")
    return final_config

def main():
    parser = argparse.ArgumentParser(description="L-Bot Parameter Optimizer")
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()

    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()
    
    # Hier könnte man die Schleife mit Joblib parallelisieren, für den Anfang sequentiell
    for symbol in symbols:
        for timeframe in timeframes:
            run_optimization_for_pair(symbol, timeframe, None, None, args.trials, args.jobs)

if __name__ == "__main__":
    main()
