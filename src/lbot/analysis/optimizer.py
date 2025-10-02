# src/lbot/analysis/optimizer.py
import os
import sys
import argparse
import json
import pandas as pd
import optuna
import logging
import time 

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, load_model_and_scaler
from lbot.utils.data_handler import get_market_data
from lbot.analysis.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Eine Klasse für unseren intelligenten Callback, der den Benchmark durchführt
class BenchmarkCallback:
    def __init__(self, n_trials, n_jobs):
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.start_time = None
        self.warmup_trials = 2 # Die ersten Trials sind zum "Aufwärmen"

    def __call__(self, study, trial):
        # Beim ersten Aufruf die Startzeit speichern
        if self.start_time is None:
            self.start_time = time.time()
        
        eta_str = "berechne..."
        # Nur nach der Aufwärmphase mit der ETA-Berechnung beginnen
        if trial.number >= self.warmup_trials:
            now = time.time()
            elapsed_seconds = now - self.start_time
            avg_time_per_trial = elapsed_seconds / (trial.number + 1)
            remaining_trials = self.n_trials - (trial.number + 1)
            
            # ETA in Sekunden berechnen (geteilt durch Anzahl der Jobs)
            if self.n_jobs > 1:
                # Bei paralleler Ausführung wird die Zeit pro "Welle" an Jobs berechnet
                eta_seconds = int(avg_time_per_trial * remaining_trials / self.n_jobs)
            else:
                eta_seconds = int(avg_time_per_trial * remaining_trials)

            # ETA in Minuten und Sekunden umrechnen
            eta_minutes, eta_sec = divmod(eta_seconds, 60)
            eta_str = f"ca. {eta_minutes}m {eta_sec}s"

        best_value_str = "N/A"
        if study.best_value is not None:
            best_value_str = f"{study.best_value:.2f}"
            
        message = f"    - Optimierung läuft: Trial {trial.number + 1}/{self.n_trials} | ETA: {eta_str} | Bester Score: {best_value_str}"
        
        sys.stdout.write('\r' + message.ljust(90))
        sys.stdout.flush()
        
        if (trial.number + 1) == self.n_trials:
            sys.stdout.write('\n')
            sys.stdout.flush()


DATA = None
MODEL = None
SCALER = None
SETTINGS = None

def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r') as f:
        return json.load(f)

def objective(trial):
    params = {
        "strategy": {"prediction_threshold": trial.suggest_float("prediction_threshold", 0.51, 0.75)},
        "risk": {
            "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 5.0),
            "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.0, 5.0),
            "leverage": trial.suggest_int("leverage", 1, 20),
            "margin_mode": "isolated"
        },
        "behavior": { "use_longs": True, "use_shorts": False }
    }
    opti_settings = SETTINGS.get('optimization_settings', {})
    backtester = Backtester(data=DATA, model=MODEL, scaler=SCALER, params=params, settings=SETTINGS, start_capital=opti_settings.get('start_capital', 1000))
    metrics = backtester.run()
    max_drawdown_constraint = opti_settings.get('constraints', {}).get('max_drawdown_pct', 30)
    if metrics['max_drawdown_pct'] > max_drawdown_constraint:
        return -999.0
    score = metrics['total_pnl_pct'] / (metrics['max_drawdown_pct'] + 1)
    return score if not pd.isna(score) else -999.0

def run_optimization_for_pair(symbol, timeframe, start_date, trials, jobs):
    global DATA, MODEL, SCALER
    logging.info(f"Starte Optimierungsprozess für {symbol} ({timeframe})...")
    
    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    
    ema_period = SETTINGS['strategy_filters'].get('ema_period', 200)
    data_raw = get_market_data(exchange, symbol, timeframe, start_date)
    if data_raw.empty:
        logging.error(f"Keine Daten für {symbol}. Überspringe.")
        return None
        
    DATA = create_ann_features(data_raw[['open', 'high', 'low', 'close', 'volume']], ema_period=ema_period)
    
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)

    if MODEL is None or SCALER is None:
        logging.error(f"Modell/Scaler für {symbol} nicht gefunden. Überspringe.")
        return None

    study = optuna.create_study(direction="maximize")
    
    # Wir erstellen eine Instanz unseres neuen Benchmark-Callbacks
    benchmark_callback = BenchmarkCallback(n_trials=trials, n_jobs=jobs)
    
    # und übergeben sie an den optimize-Aufruf
    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=jobs,
        callbacks=[benchmark_callback]
    )
    
    if not study.best_trial:
        logging.warning(f"Optuna fand keine gültige Lösung für {symbol} ({timeframe}).")
        return None

    best_params_dict = study.best_trial.params
    best_score = study.best_trial.value
    logging.info(f"Beste Parameter für {symbol} ({timeframe}) gefunden. Score: {best_score:.2f}")

    final_config = {
        "market": {"symbol": symbol, "timeframe": timeframe},
        "strategy": {"prediction_threshold": best_params_dict['prediction_threshold']},
        "risk": {
            "risk_per_trade_pct": best_params_dict['risk_per_trade_pct'],
            "risk_reward_ratio": best_params_dict['risk_reward_ratio'],
            "leverage": best_params_dict['leverage'],
            "margin_mode": "isolated"
        },
        "behavior": {"use_longs": True, "use_shorts": False}
    }
    config_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_{safe_filename}.json')
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=4)
    logging.info(f"Beste Konfiguration gespeichert in: {config_path}")

    opti_settings = SETTINGS.get('optimization_settings', {})
    final_backtester = Backtester(data=DATA, model=MODEL, scaler=SCALER, params=final_config, settings=SETTINGS, start_capital=opti_settings.get('start_capital', 1000))
    final_metrics = final_backtester.run()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "score": best_score,
        "params": final_config,
        "metrics": final_metrics
    }

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
    
    total_jobs = len(symbols) * len(timeframes)
    job_count = 0
    
    all_results = []
    for symbol in symbols:
        for timeframe in timeframes:
            job_count += 1
            logging.info(f"--- Paket {job_count}/{total_jobs}: Start für {symbol} ({timeframe}) ---")
            result = run_optimization_for_pair(symbol, timeframe, args.start_date, args.trials, int(args.jobs))
            if result:
                all_results.append(result)

    if all_results:
        results_path = os.path.join(PROJECT_ROOT, 'artifacts', 'optimization_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        logging.info(f"✅ Optimierung abgeschlossen. {len(all_results)} Ergebnisse wurden in {results_path} gespeichert.")
    else:
        logging.warning("Optimierung beendet, aber keine Ergebnisse zum Speichern gefunden.")

if __name__ == "__main__":
    main()
