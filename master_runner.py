# lbot/master_runner.py
import json
import os
import sys
import subprocess
import logging
from datetime import datetime, timezone

# --- Pfad-Konfiguration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

SETTINGS_FILE = os.path.join(PROJECT_ROOT, 'settings.json')
SECRET_FILE = os.path.join(PROJECT_ROOT, 'secret.json')
TIMESTAMPS_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'db', 'last_run_timestamps.json')
BOT_RUNNER_SCRIPT = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'run.py')
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python3')

# --- Wichtige Importe aus dem Projekt ---
from lbot.utils.trade_manager import babysit_open_position
from lbot.utils.exchange import Exchange
from lbot.strategy.run import get_state, set_state

# --- Setup für einfaches Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("MasterRunner")

def get_candle_start_time(dt: datetime, timeframe: str) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == 'm': return dt.replace(minute=dt.minute - (dt.minute % value))
    elif unit == 'h': return dt.replace(minute=0, hour=dt.hour - (dt.hour % value))
    elif unit == 'd': return dt.replace(hour=0, minute=0)
    else: return dt

def load_json(file_path):
    if not os.path.exists(file_path): return {}
    try:
        with open(file_path, 'r') as f: return json.load(f)
    except json.JSONDecodeError: return {}

def save_timestamps(timestamps):
    os.makedirs(os.path.dirname(TIMESTAMPS_FILE), exist_ok=True)
    with open(TIMESTAMPS_FILE, 'w') as f: json.dump(timestamps, f, indent=4)

def load_strategy_config(symbol, timeframe):
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    config_path = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs', f'config_{safe_filename}.json')
    return load_json(config_path)

def main():
    print(f"\n--- L-Bot Master Runner gestartet um {datetime.now().isoformat()} ---")

    settings = load_json(SETTINGS_FILE)
    secrets = load_json(SECRET_FILE)
    
    active_strategies = settings.get('live_trading_settings', {}).get('active_strategies', [])
    account_config = secrets.get('lbot', [{}])[0]
    telegram_config = secrets.get('telegram', {})

    if not active_strategies or not account_config.get('apiKey'):
        print("Keine aktiven Strategien oder API-Schlüssel in settings/secret.json gefunden.")
        return

    # === DER BABYSITTER-CHECK (LÄUFT IMMER ZUERST) ===
    print("--- Starte Babysitter-Überwachung für alle aktiven Strategien ---")
    try:
        exchange = Exchange(account_config)
        for strategy in active_strategies:
            params = load_strategy_config(strategy['symbol'], strategy['timeframe'])
            if params:
                babysit_open_position(exchange, params, get_state, set_state, telegram_config, log)
    except Exception as e:
        print(f"KRITISCHER FEHLER im Babysitter-Prozess: {e}")
    print("--- Babysitter-Überwachung abgeschlossen ---")
    
    # === PRÄZISIONS-SCHEDULING ===
    timestamps = load_json(TIMESTAMPS_FILE)
    now_utc = datetime.now(timezone.utc)
    
    for strategy in active_strategies:
        symbol, timeframe = strategy.get('symbol'), strategy.get('timeframe')
        if not symbol or not timeframe: continue
            
        strategy_id = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
        last_run_str = timestamps.get(strategy_id)
        current_candle_start = get_candle_start_time(now_utc, timeframe)
        
        run_it = False
        if not last_run_str:
            print(f"[{strategy_id}] Läuft zum ersten Mal für den Intervall {current_candle_start.isoformat()}.")
            run_it = True
        else:
            last_run_time = datetime.fromisoformat(last_run_str).replace(tzinfo=timezone.utc)
            last_run_candle_start = get_candle_start_time(last_run_time, timeframe)
            if current_candle_start > last_run_candle_start:
                print(f"[{strategy_id}] Neuer Zeit-Block erkannt. Starte den Bot.")
                run_it = True
            else:
                print(f"[{strategy_id}] Läuft bereits für den aktuellen Zeit-Block. Überspringe.")

        if run_it:
            command = [VENV_PYTHON, BOT_RUNNER_SCRIPT, '--symbol', symbol, '--timeframe', timeframe]
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("--- FEHLER IM SUBPROZESS ---")
                print(result.stderr)
            timestamps[strategy_id] = now_utc.isoformat()

    save_timestamps(timestamps)
    print("--- L-Bot Master Runner beendet ---")

if __name__ == "__main__":
    main()
