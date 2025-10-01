# src/lbot/strategy/run.py
import os
import sys
import json
import logging
import sqlite3
import argparse

# --- Pfad-Konfiguration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# --- Kern-Importe für L-Bot ---
from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import load_model_and_scaler
from lbot.utils.trade_manager import full_trade_cycle
from lbot.utils.telegram import send_message
from lbot.utils.decorators import run_with_guardian_checks

# --- Hilfsfunktionen ---
def create_safe_filename(symbol, timeframe):
    return f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"

def load_config(symbol, timeframe):
    safe_filename = create_safe_filename(symbol, timeframe)
    config_dir = os.path.join(PROJECT_ROOT, 'src', 'lbot', 'strategy', 'configs')
    config_path = os.path.join(config_dir, f'config_{safe_filename}.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(symbol, timeframe):
    safe_filename = create_safe_filename(symbol, timeframe)
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'lbot_{safe_filename}.log') # Log-Name angepasst
    logger = logging.getLogger(f'lbot_{safe_filename}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s UTC %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_db_file_path(account_name, symbol, timeframe):
    db_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'db')
    os.makedirs(db_dir, exist_ok=True)
    safe_name = "".join(c for c in account_name if c.isalnum() or c in (' ', '_')).rstrip()
    safe_filename = create_safe_filename(symbol, timeframe)
    return os.path.join(db_dir, f"state_{safe_name}_{safe_filename}.db")

def setup_database(account_name, symbol, timeframe):
    db_path = get_db_file_path(account_name, symbol, timeframe)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS bot_state (key TEXT PRIMARY KEY, value TEXT)''')
    conn.commit()
    conn.close()

def get_state(account_name, symbol, timeframe, key, default='0'):
    db_path = get_db_file_path(account_name, symbol, timeframe)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM bot_state WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else default

def set_state(account_name, symbol, timeframe, key, value):
    db_path = get_db_file_path(account_name, symbol, timeframe)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO bot_state (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()

# --- Hauptlogik ---
@run_with_guardian_checks
def run_for_account(account, telegram_config, params, model, scaler, logger, model_path, scaler_path):
    account_name = account.get('name', 'Standard-Account')
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    
    logger.info(f"--- Starte L-Bot für {account_name} auf {symbol} ({timeframe}) ---")
    
    exchange = Exchange(account)
    setup_database(account_name, symbol, timeframe)
        
    current_balance = exchange.fetch_balance_usdt()
    logger.info(f"Aktueller Kontostand: {current_balance:.2f} USDT")

    if current_balance <= 0:
        logger.warning("Kein Guthaben gefunden oder Fehler beim Abruf. Überspringe den Handelszyklus.")
        return

    full_trade_cycle(
        exchange=exchange,
        model=model,
        scaler=scaler,
        params=params,
        current_balance=current_balance,
        db_set_state=set_state,
        telegram_config=telegram_config,
        logger=logger
    )

def main():
    parser = argparse.ArgumentParser(description="L-Bot LSTM Trading Skript")
    parser.add_argument('--symbol', required=True, type=str)
    parser.add_argument('--timeframe', required=True, type=str)
    args, _ = parser.parse_known_args()

    symbol, timeframe = args.symbol, args.timeframe
    logger = setup_logging(symbol, timeframe)
    
    try:
        params = load_config(symbol, timeframe)
        safe_filename = create_safe_filename(symbol, timeframe)
        model_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_predictor_{safe_filename}.h5')
        scaler_path = os.path.join(PROJECT_ROOT, 'artifacts', 'models', f'ann_scaler_{safe_filename}.joblib')
        MODEL, SCALER = load_model_and_scaler(model_path, scaler_path)
        
        if MODEL is None or SCALER is None:
            raise FileNotFoundError(f"Modell oder Scaler für {symbol} ({timeframe}) nicht gefunden.")
            
        with open(os.path.join(PROJECT_ROOT, 'secret.json'), "r") as f:
            secrets = json.load(f)
        accounts_to_run = secrets.get('lbot', [])
        telegram_config = secrets.get('telegram', {})

    except Exception as e:
        logger.critical(f"Kritischer Initialisierungs-Fehler: {e}")
        sys.exit(1)
        
    for account in accounts_to_run:
        run_for_account(account, telegram_config, params, MODEL, SCALER, logger, model_path, scaler_path)
    
    logger.info(f">>> L-Bot-Lauf für {symbol} ({timeframe}) abgeschlossen <<<\n")

if __name__ == "__main__":
    main()
