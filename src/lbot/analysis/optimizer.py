# src/lbot/analysis/optimizer.py
import os
import sys
import argparse
import json
import pandas as pd
import optuna
import logging
from optuna.integration import TqdmCallback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, load_model_and_scaler
from lbot.utils.data_handler import get_market_data
from lbot.analysis.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Globale Variablen, die pro Optimierungslauf gesetzt werden
DATA = None
MODEL = None
SCALER = None
SETTINGS = None

def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r') as f:
        return json.load(f)

def objective(trial):
    """ Die Optuna-Zielfunktion, die für jeden Versuch aufgerufen wird. """
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
        "behavior": { "use_longs": True, "use_shorts": False }
    }

    # Lade Backtest-Einstellungen aus der Hauptkonfiguration
    backtest_settings = SETTINGS.get('backtest_settings', {})
    opti_settings = SETTINGS.get('optimization_settings', {})

    backtester = Backtester(
        data=DATA, 
        model=MODEL, 
        scaler=SCALER, 
        params=params,
        start_capital=opti_settings.get('start_capital', 1000),
        sequence_length=SETTINGS['model_settings']['sequence_length'],
        fee_rate_pct=backtest_settings.get('fee_rate_pct', 0.06),
        slippage_pct=backtest_settings.get('slippage_pct', 0.02)
    )
    metrics = backtester.run()

    # Constraint: Bestrafe zu hohen Drawdown hart
    max_drawdown_constraint = opti_settings.get('constraints', {}).get('max_drawdown_pct', 30)
    if metrics['max_drawdown_pct'] > max_drawdown_constraint:
        return -999.0

    # Zielfunktion: PnL im Verhältnis zum Drawdown
    score = metrics['total_pnl_pct'] / (metrics['max_drawdown_pct'] + 1)
    return score if not pd.isna(score) else -999.0

def run_optimization_for_pair(symbol, timeframe, start_date, trials, jobs):
    """ Startet die komplette Optimierung für ein Handelspaar. """
    global DATA, MODEL, SCALER
    
    logging.info(f"Starte Optimierung für {symbol} ({timeframe})...")

    # Lade Daten mit dem neuen Data Handler
    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    data_raw = get_market_data(exchange, symbol, timeframe, start_date)
    if data_raw.empty:
        logging.error(f"Keine Daten für {symbol} ({timeframe}) erhalten. Überspringe Optimierung.")
        return None
    DATA = create_ann_features(data_raw[['open', 'high', 'low', 'close', 'volume']])
    
    # Lade Modell und Scaler
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)

    if MODEL is None or SCALER is None:
        logging.error(f"Modell/Scaler für {symbol} ({timeframe}) nicht gefunden. Überspringe.")
        return None

    tqdm_callback = TqdmCallback(True)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, n_jobs=jobs, callbacks=[tqdm_callback])
    
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
    global SETTINGS
    SETTINGS = load_settings()

    parser = argparse.ArgumentParser(description="L-Bot Parameter Optimizer")
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--start_date', required=True, type=str)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()

    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()
    
    for symbol in symbols:
        for timeframe in timeframes:
            run_optimization_for_pair(symbol, timeframe, args.start_date, args.trials, args.jobs)

if __name__ == "__main__":
    main()
