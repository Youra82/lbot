# src/lbot/analysis/optimizer.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
import argparse
import json
import pandas as pd
import optuna
import logging
import time 
from collections import deque

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, load_model_and_scaler
from lbot.utils.data_handler import get_market_data
from lbot.analysis.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BenchmarkCallback:
    def __init__(self, n_trials, n_jobs):
        self.n_trials = n_trials
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.trial_durations = deque(maxlen=20) 
        self.last_time = None

    def _format_seconds(self, seconds: int) -> str:
        minutes, sec = divmod(seconds, 60)
        return f"{minutes}m {sec}s"

    def __call__(self, study, trial):
        now = time.time()
        if self.last_time is not None:
            duration = now - self.last_time
            self.trial_durations.append(duration)
        self.last_time = now

        eta_str = "berechne..."
        if len(self.trial_durations) > 5:
            avg_time_per_trial = sum(self.trial_durations) / len(self.trial_durations)
            remaining_trials = self.n_trials - (trial.number + 1)
            
            eta_seconds = 0
            if self.n_jobs > 1:
                eta_seconds = int(avg_time_per_trial * remaining_trials / self.n_jobs)
            else:
                eta_seconds = int(avg_time_per_trial * remaining_trials)
            eta_str = f"ca. {self._format_seconds(eta_seconds)}"

        best_value_str = "N/A"
        if study.best_value is not None:
            best_value_str = f"{study.best_value:.2f}"
            
        total_elapsed_str = ""
        if 'start_time' in study.user_attrs:
             total_elapsed_seconds = int(now - study.user_attrs['start_time'])
             total_elapsed_str = f"Laufzeit: {self._format_seconds(total_elapsed_seconds)} | "

        message = f"    - Optimierung läuft: Trial {trial.number + 1}/{self.n_trials} | {total_elapsed_str}ETA: {eta_str} | Bester Score: {best_value_str}"
        
        sys.stdout.write('\r' + message.ljust(100))
        sys.stdout.flush()
        
        if (trial.number + 1) == self.n_trials:
            sys.stdout.write('\n')
            sys.stdout.flush()

# Globale Variablen für Rohdaten und KI-Modell
RAW_DATA = None
MODEL = None
SCALER = None
SETTINGS = None

def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r') as f:
        return json.load(f)

def objective(trial):
    try:
        # Filter-Parameter werden jetzt hier von Optuna vorgeschlagen
        ema_period = trial.suggest_int("ema_period", 50, 400, step=10)
        atr_period = trial.suggest_int("atr_period", 7, 28)

        # Features werden für jeden Trial mit den dynamischen Parametern neu erstellt
        data_with_features = create_ann_features(RAW_DATA.copy(), ema_period=ema_period, atr_period=atr_period)
        
        params = {
            "strategy": {
                "prediction_threshold": trial.suggest_float("prediction_threshold", 0.60, 0.90),
                "min_natr": trial.suggest_float("min_natr", 0.2, 1.5),
                "max_natr": trial.suggest_float("max_natr", 1.5, 8.0)
            },
            "risk": {
                "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 3.0),
                "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.5, 5.0),
                "leverage": trial.suggest_int("leverage", 1, 10),
            },
            "behavior": { "use_longs": True, "use_shorts": False },
            # Wir speichern die Filter-Perioden mit ab, um sie später zu verwenden
            "filters": {
                "ema_period": ema_period,
                "atr_period": atr_period
            }
        }
        
        if params["strategy"]["max_natr"] <= params["strategy"]["min_natr"]:
            return -999.0 # Ungültige Kombination, sofort verwerfen

        opti_settings = SETTINGS.get('optimization_settings', {})
        
        # Der Backtester erhält die pro Trial erstellten Daten
        backtester = Backtester(data=data_with_features, model=MODEL, scaler=SCALER, params=params, settings=SETTINGS, start_capital=opti_settings.get('start_capital', 1000))
        metrics = backtester.run()

        max_drawdown_constraint = opti_settings.get('constraints', {}).get('max_drawdown_pct', 30)
        if metrics['max_drawdown_pct'] > max_drawdown_constraint or metrics['num_trades'] < 10:
            return -999.0
            
        # Optional: Score-Funktion anpassen, um die Win-Rate stärker zu gewichten
        score = metrics['total_pnl_pct'] * (metrics.get('win_rate', 0) / 100) / (metrics['max_drawdown_pct'] + 1)
        return score if not pd.isna(score) else -999.0
    except Exception:
        # Fängt alle unerwarteten Fehler in einem Trial ab
        return -999.0

def run_optimization_for_pair(symbol, timeframe, start_date, trials, jobs):
    global RAW_DATA, MODEL, SCALER
    logging.info(f"Starte Optimierungsprozess für {symbol} ({timeframe})...")
    
    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    
    # Wir laden nur noch die Rohdaten einmal pro Lauf
    RAW_DATA = get_market_data(exchange, symbol, timeframe, start_date)
    if RAW_DATA.empty or len(RAW_DATA) < 400: # Mindestanzahl für lange EMAs
        logging.warning(f"Nicht genug Rohdaten für {symbol} ({len(RAW_DATA)} Kerzen). Überspringe.")
        return None
    
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)

    if MODEL is None or SCALER is None:
        logging.error(f"Modell/Scaler für {symbol} nicht gefunden. Überspringe.")
        return None

    study = optuna.create_study(direction="maximize")
    study.set_user_attr('start_time', time.time())
    benchmark_callback = BenchmarkCallback(n_trials=trials, n_jobs=jobs)
    
    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=jobs,
        callbacks=[benchmark_callback],
        catch=(Exception,)
    )
    
    if not study.best_trial or study.best_value <= 0:
        logging.warning(f"Optuna fand keine profitable Lösung für {symbol} ({timeframe}).")
        return None

    best_params_dict = study.best_trial.params
    best_score = study.best_trial.value
    logging.info(f"Beste Parameter für {symbol} ({timeframe}) gefunden. Score: {best_score:.2f}")

    final_config = {
        "market": {"symbol": symbol, "timeframe": timeframe},
        "strategy": {
            "prediction_threshold": best_params_dict['prediction_threshold'],
            "min_natr": best_params_dict['min_natr'],
            "max_natr": best_params_dict['max_natr']
        },
        "risk": {
            "risk_per_trade_pct": best_params_dict['risk_per_trade_pct'],
            "risk_reward_ratio": best_params_dict['risk_reward_ratio'],
            "leverage": best_params_dict['leverage'],
        },
        "behavior": {"use_longs": True, "use_shorts": False},
        "filters": {
            "ema_period": best_params_dict['ema_period'],
            "atr_period": best_params_dict['atr_period']
        }
    }
    
    config_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_{safe_filename}.json')
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=4)
    logging.info(f"Beste Konfiguration gespeichert in: {config_path}")

    # Finaler Backtest zur Verifizierung
    final_features = create_ann_features(RAW_DATA.copy(), ema_period=best_params_dict['ema_period'], atr_period=best_params_dict['atr_period'])
    opti_settings = SETTINGS.get('optimization_settings', {})
    final_backtester = Backtester(data=final_features, model=MODEL, scaler=SCALER, params=final_config, settings=SETTINGS, start_capital=opti_settings.get('start_capital', 1000))
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
            result = run_optimization_for_pair(symbol, timeframe, args.start_date, int(args.trials), int(args.jobs))
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
