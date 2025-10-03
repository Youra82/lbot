# src/lbot/analysis/optimizer.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys, argparse, json, pandas as pd, optuna, logging, time
from collections import deque
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features
from lbot.utils.data_handler import get_market_data
from lbot.analysis.backtester import Backtester
# KORRIGIERT: Lade auch create_sequences
from lbot.utils.lstm_model import create_sequences, load_model_and_scaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BenchmarkCallback:
    # ... (Klasse bleibt unverändert) ...
    def __init__(self, n_trials, n_jobs):
        self.n_trials=n_trials; self.n_jobs=n_jobs if n_jobs!=-1 else os.cpu_count(); self.trial_durations=deque(maxlen=20); self.last_time=None
    def _format_seconds(self, seconds: int) -> str:
        minutes, sec=divmod(seconds, 60); return f"{minutes}m {sec}s"
    def __call__(self, study, trial):
        now=time.time()
        if self.last_time is not None: self.trial_durations.append(now - self.last_time)
        self.last_time=now
        eta_str="berechne..."
        if len(self.trial_durations) > 5:
            avg_time_per_trial=sum(self.trial_durations)/len(self.trial_durations)
            remaining_trials=self.n_trials - (trial.number + 1)
            eta_seconds=0
            if self.n_jobs > 1: eta_seconds=int(avg_time_per_trial * remaining_trials / self.n_jobs)
            else: eta_seconds=int(avg_time_per_trial * remaining_trials)
            eta_str=f"ca. {self._format_seconds(eta_seconds)}"
        best_value_str="N/A"
        if study.best_value is not None: best_value_str=f"{study.best_value:.2f}"
        total_elapsed_str=""
        if 'start_time' in study.user_attrs:
             total_elapsed_seconds=int(now - study.user_attrs['start_time'])
             total_elapsed_str=f"Laufzeit: {self._format_seconds(total_elapsed_seconds)} | "
        message=f"    - Optimierung läuft: Trial {trial.number + 1}/{self.n_trials} | {total_elapsed_str}ETA: {eta_str} | Bester Score: {best_value_str}"
        sys.stdout.write('\r' + message.ljust(100)); sys.stdout.flush()
        if (trial.number + 1)==self.n_trials: sys.stdout.write('\n'); sys.stdout.flush()

DATA_WITH_FEATURES = None
MODEL = None
SCALER = None
SETTINGS = None
OPTIM_MODE = "strict" 

def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r') as f: return json.load(f)

def objective(trial):
    try:
        params = {
            "strategy": {
                "entry_threshold_pct": trial.suggest_float("entry_threshold_pct", 0.5, 3.0),
                "uncertainty_threshold": trial.suggest_float("uncertainty_threshold", 0.001, 0.02),
                "min_natr": trial.suggest_float("min_natr", 0.2, 1.5),
                "max_natr": trial.suggest_float("max_natr", 1.5, 8.0)
            },
            "risk": { "risk_per_trade_pct": trial.suggest_float("risk_per_trade_pct", 0.5, 3.0), "risk_reward_ratio": trial.suggest_float("risk_reward_ratio", 1.5, 5.0), "leverage": trial.suggest_int("leverage", 1, 10)},
            "behavior": { "use_longs": True, "use_shorts": False }
        }
        
        if params["strategy"]["max_natr"] <= params["strategy"]["min_natr"]: return -999.0

        opti_settings = SETTINGS.get('optimization_settings', {})
        backtester = Backtester(data=DATA_WITH_FEATURES.copy(), model=MODEL, scaler=SCALER, params=params, settings=SETTINGS, start_capital=opti_settings.get('start_capital', 1000))
        metrics = backtester.run()

        if OPTIM_MODE == "strict":
            constraints = opti_settings.get('constraints', {})
            min_trades = 20
            if (metrics['max_drawdown_pct'] > constraints.get('max_drawdown_pct', 99) or metrics['win_rate'] < constraints.get('min_win_rate_pct', 0) or metrics['total_pnl_pct'] < constraints.get('min_pnl_pct', -100) or metrics['num_trades'] < min_trades):
                return -999.0
        else:
            if metrics['max_drawdown_pct'] > 80 or metrics['num_trades'] < 5: return -999.0
            
        pnl = metrics['total_pnl_pct']; drawdown = metrics['max_drawdown_pct']; win_rate = metrics.get('win_rate', 0); num_trades = metrics.get('num_trades', 0)
        trade_penalty = 1.0 if num_trades > 50 else num_trades / 50.0
        if drawdown > 0: score = (pnl * (win_rate / 100)) / drawdown * trade_penalty
        else: score = pnl * (win_rate / 100) * trade_penalty
        return score if not pd.isna(score) else -999.0
    except Exception:
        return -999.0

def run_optimization_for_pair(symbol, timeframe, start_date, trials, jobs):
    global DATA_WITH_FEATURES, MODEL, SCALER
    logging.info(f"Starte Optimierungsprozess für {symbol} ({timeframe})...")
    
    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy'}
    exchange = Exchange(dummy_account)
    
    raw_data = get_market_data(exchange, symbol, timeframe, start_date)
    if raw_data.empty or len(raw_data) < 400:
        logging.warning(f"Nicht genug Rohdaten für {symbol}. Überspringe."); return None
        
    DATA_WITH_FEATURES = create_ann_features(raw_data)

    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
    MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)

    if MODEL is None or SCALER is None:
        logging.error(f"Modell/Scaler für {symbol} nicht gefunden. Überspringe."); return None

    study = optuna.create_study(direction="maximize")
    study.set_user_attr('start_time', time.time())
    benchmark_callback = BenchmarkCallback(n_trials=trials, n_jobs=jobs)
    
    study.optimize(objective, n_trials=trials, n_jobs=jobs, callbacks=[benchmark_callback], catch=(Exception,))
    
    if not study.best_trial or study.best_value <= 0:
        logging.warning(f"Optuna fand keine profitable Lösung für {symbol} ({timeframe})."); return None

    best_params_dict = study.best_trial.params
    best_score = study.best_trial.value
    logging.info(f"Beste Parameter für {symbol} ({timeframe}) gefunden. Score: {best_score:.2f}")

    final_config = {
        "market": {"symbol": symbol, "timeframe": timeframe},
        "strategy": {
            "entry_threshold_pct": best_params_dict['entry_threshold_pct'],
            "uncertainty_threshold": best_params_dict['uncertainty_threshold'],
            "min_natr": best_params_dict['min_natr'],
            "max_natr": best_params_dict['max_natr']
        },
        "risk": { "risk_per_trade_pct": best_params_dict['risk_per_trade_pct'], "risk_reward_ratio": best_params_dict['risk_reward_ratio'], "leverage": best_params_dict['leverage']},
        "behavior": {"use_longs": True, "use_shorts": False}
    }
    
    config_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f'config_{safe_filename}.json')
    with open(config_path, 'w') as f: json.dump(final_config, f, indent=4)
    logging.info(f"Beste Konfiguration gespeichert in: {config_path}")

    final_backtester = Backtester(data=DATA_WITH_FEATURES.copy(), model=MODEL, scaler=SCALER, params=final_config, settings=SETTINGS, start_capital=SETTINGS.get('optimization_settings', {}).get('start_capital', 1000))
    final_metrics = final_backtester.run()
    return {"symbol": symbol, "timeframe": timeframe, "score": best_score, "params": final_config, "metrics": final_metrics}

def main():
    global SETTINGS, OPTIM_MODE
    SETTINGS = load_settings()
    parser = argparse.ArgumentParser(description="L-Bot Parameter Optimizer")
    parser.add_argument('--mode', type=str, default='strict')
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--start_date', required=True, type=str)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()
    OPTIM_MODE = args.mode
    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()
    total_jobs = len(symbols) * len(timeframes)
    job_count = 0
    all_results = []
    for symbol in symbols:
        for timeframe in timeframes:
            job_count += 1
            logging.info(f"--- Paket {job_count}/{total_jobs}: Start für {symbol} ({timeframe}) im '{OPTIM_MODE}'-Modus ---")
            result = run_optimization_for_pair(symbol, timeframe, args.start_date, int(args.trials), int(args.jobs))
            if result: all_results.append(result)
    if all_results:
        results_path = os.path.join(PROJECT_ROOT, 'artifacts', 'optimization_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f: json.dump(all_results, f, indent=4)
        logging.info(f"✅ Optimierung abgeschlossen. {len(all_results)} Ergebnisse wurden in {results_path} gespeichert.")
    else:
        logging.warning("Optimierung beendet, aber keine Ergebnisse zum Speichern gefunden.")

if __name__ == "__main__":
    main()
