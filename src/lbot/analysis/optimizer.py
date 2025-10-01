# src/jaegerbot/analysis/optimizer.py
import os
import sys
import json
import optuna
import numpy as np
import argparse

# --- Logging-Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
# --- Ende ---

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from jaegerbot.analysis.backtester import load_data, run_ann_backtest
from jaegerbot.utils.telegram import send_message

optuna.logging.set_verbosity(optuna.logging.WARNING)
HISTORICAL_DATA = None
CURRENT_MODEL_PATHS = {}
MAX_DRAWDOWN_CONSTRAINT = 0.30
MIN_WIN_RATE_CONSTRAINT = 55.0
MIN_PNL_CONSTRAINT = 0.0
START_CAPITAL = 1000
OPTIM_MODE = "strict"

def create_safe_filename(symbol, timeframe):
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"

def objective(trial):
    params = {
        'prediction_threshold': trial.suggest_float('prediction_threshold', 0.55, 0.85),
        'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 1.0, 5.0),
        'risk_per_trade_pct': trial.suggest_float('risk_per_trade_pct', 0.5, 2.0),
        'leverage': trial.suggest_int('leverage', 5, 25),
        'trailing_stop_activation_rr': trial.suggest_float('trailing_stop_activation_rr', 1.0, 4.0),
        'trailing_stop_callback_rate_pct': trial.suggest_float('trailing_stop_callback_rate_pct', 0.5, 3.0)
    }
    result = run_ann_backtest(HISTORICAL_DATA.copy(), params, CURRENT_MODEL_PATHS, START_CAPITAL)
    
    pnl, drawdown, trades, win_rate = result.get('total_pnl_pct', -1000), result.get('max_drawdown_pct', 1.0), result.get('trades_count', 0), result.get('win_rate', 0)

    if OPTIM_MODE == "strict":
        if drawdown > MAX_DRAWDOWN_CONSTRAINT or win_rate < MIN_WIN_RATE_CONSTRAINT or pnl < MIN_PNL_CONSTRAINT or trades < 50:
            raise optuna.exceptions.TrialPruned()
    elif OPTIM_MODE == "best_profit":
        if drawdown > MAX_DRAWDOWN_CONSTRAINT or trades < 50:
            raise optuna.exceptions.TrialPruned()
        
    return pnl

def main():
    global HISTORICAL_DATA, CURRENT_MODEL_PATHS, MAX_DRAWDOWN_CONSTRAINT, MIN_WIN_RATE_CONSTRAINT, MIN_PNL_CONSTRAINT, START_CAPITAL, OPTIM_MODE
    
    parser = argparse.ArgumentParser(description="Parameter-Optimierung für JaegerBot")
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--start_date', required=True, type=str)
    parser.add_argument('--end_date', required=True, type=str)
    parser.add_argument('--jobs', required=True, type=int)
    parser.add_argument('--max_drawdown', required=True, type=float)
    parser.add_argument('--start_capital', required=True, type=float)
    parser.add_argument('--min_win_rate', required=True, type=float)
    parser.add_argument('--trials', required=True, type=int)
    parser.add_argument('--min_pnl', required=True, type=float)
    parser.add_argument('--mode', required=True, type=str)
    parser.add_argument('--top_n', type=int, default=0)
    args = parser.parse_args()

    MAX_DRAWDOWN_CONSTRAINT = args.max_drawdown / 100.0
    MIN_WIN_RATE_CONSTRAINT = args.min_win_rate
    MIN_PNL_CONSTRAINT = args.min_pnl
    START_CAPITAL = args.start_capital
    N_TRIALS = args.trials
    OPTIM_MODE = args.mode
    TOP_N_STRATEGIES = args.top_n

    symbols, timeframes = args.symbols.split(), args.timeframes.split()
    TASKS = [{'symbol': f"{s}/USDT:USDT", 'timeframe': tf} for s in symbols for tf in timeframes]
    DB_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'optuna_studies.db')
    
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    STORAGE_URL = f"sqlite:///{DB_FILE}"

    try:
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), "r") as f:
            secrets = json.load(f)
        telegram_config = secrets.get('telegram', {})
    except Exception as e:
        print(f"Warnung: Konnte Telegram-Konfiguration nicht laden: {e}")
        telegram_config = {}
    
    all_successful_results = []
    failed_tasks = []

    for task in TASKS:
        symbol, timeframe = task['symbol'], task['timeframe']
        print(f"\n===== Optimiere: {symbol} ({timeframe}) im '{OPTIM_MODE}'-Modus =====")
        
        safe_filename = create_safe_filename(symbol, timeframe)
        CURRENT_MODEL_PATHS['model'] = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
        CURRENT_MODEL_PATHS['scaler'] = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')

        HISTORICAL_DATA = load_data(symbol, timeframe, args.start_date, args.end_date)
        if HISTORICAL_DATA.empty: 
            failed_tasks.append(f"{symbol} ({timeframe}) - Datenfehler")
            continue

        study_name = f"ann_{safe_filename}_{OPTIM_MODE}"
        study = optuna.create_study(storage=STORAGE_URL, study_name=study_name, direction="maximize", load_if_exists=True)
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=args.jobs, show_progress_bar=True)
        
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not valid_trials:
            print(f"\n❌ FEHLER: Für {symbol} ({timeframe}) konnte keine Konfiguration gefunden werden.")
            failed_tasks.append(f"{symbol} ({timeframe})")
            continue
        
        best_trial = max(valid_trials, key=lambda t: t.value)
        best_params = best_trial.params
        final_result = run_ann_backtest(HISTORICAL_DATA.copy(), best_params, CURRENT_MODEL_PATHS, START_CAPITAL)
            
        result_summary = {"symbol": symbol, "timeframe": timeframe, **final_result}
        all_successful_results.append(result_summary)

        config_dir = os.path.join(PROJECT_ROOT, 'src', 'jaegerbot', 'strategy', 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_output_path = os.path.join(config_dir, f'config_{safe_filename}.json')
        config_output = { "market": {"symbol": symbol, "timeframe": timeframe}, "strategy": {"prediction_threshold": round(best_params['prediction_threshold'], 3)}, "risk": { "margin_mode": "isolated", "balance_fraction_pct": 100, "risk_per_trade_pct": round(best_params['risk_per_trade_pct'], 2), "risk_reward_ratio": round(best_params['risk_reward_ratio'], 2), "leverage": best_params['leverage'], "trailing_stop_activation_rr": round(best_params['trailing_stop_activation_rr'], 2), "trailing_stop_callback_rate_pct": round(best_params['trailing_stop_callback_rate_pct'], 2)}, "behavior": {"use_longs": True, "use_shorts": True} }
        with open(config_output_path, 'w') as f: json.dump(config_output, f, indent=4)
        print(f"\n✔ Beste Konfiguration wurde in '{config_output_path}' gespeichert.")
    
    # --- FINALE PORTFOLIO-AUSWAHL UND EINZIGER TELEGRAM-BERICHT ---
    title = f"*{'Automatische' if TOP_N_STRATEGIES > 0 else 'Manuelle'} Optimierung Abgeschlossen*"
    
    if not all_successful_results:
        message = f"❌ {title}\n\nEs wurden keine neuen, profitablen Strategien gefunden, die deine Bedingungen erfüllen."
    else:
        sorted_results = sorted(all_successful_results, key=lambda x: x['total_pnl_pct'], reverse=True)
        
        message = f"✅ {title}\n\n*Erfolgreich:* {len(all_successful_results)}\n*Fehlgeschlagen:* {len(failed_tasks)}\n\n"
        
        # Logik für den Autopiloten oder Anzeige der Top-Ergebnisse
        if TOP_N_STRATEGIES > 0:
            top_strategies = sorted_results[:TOP_N_STRATEGIES]
            autopilot_list = [{"symbol": s['symbol'], "timeframe": s['timeframe']} for s in top_strategies]
            results_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'results')
            os.makedirs(results_dir, exist_ok=True)
            results_path = os.path.join(results_dir, 'optimization_results.json')
            with open(results_path, 'w') as f:
                json.dump({"approved_strategies": autopilot_list}, f, indent=4)
            message += f"Die *Top {len(autopilot_list)}* wurden für den Autopiloten ausgewählt:\n\n"
            display_results = top_strategies
        else: # Manueller Lauf
            message += f"Hier sind die Top 5 Ergebnisse:\n\n"
            display_results = sorted_results[:5]

        for i, strat in enumerate(display_results):
            message += (
                f"*{i+1}. {strat['symbol']} ({strat['timeframe']})*\n"
                f"  - PnL: {strat['total_pnl_pct']:.2f}%\n"
                f"  - WR: {strat['win_rate']:.2f}%\n"
                f"  - DD: {strat['max_drawdown_pct']*100:.2f}%\n"
            )
    
    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)

if __name__ == "__main__":
    main()
