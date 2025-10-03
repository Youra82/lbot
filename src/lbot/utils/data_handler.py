# src/lbot/utils/data_handler.py
import os
import sys
import pandas as pd
import logging
import ccxt
from datetime import datetime, timezone
import time
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
HISTORY_DIR = os.path.join(PROJECT_ROOT, 'data', 'history')
os.makedirs(HISTORY_DIR, exist_ok=True)

log = logging.getLogger("DataHandler")
log.setLevel(logging.INFO)

def get_history_filepath(symbol, timeframe):
    """ Erstellt einen sicheren Dateinamen für die lokale Historiendatei. """
    safe_symbol = symbol.replace('/', '_').replace(':', '')
    return os.path.join(HISTORY_DIR, f"{safe_symbol}_{timeframe}.parquet")

def _download_binance_data(symbol, timeframe, start_date_str):
    """
    Private Funktion, die den Download-Prozess von Binance durchführt.
    Startet den Download ab dem übergebenen Startdatum.
    """
    exchange = ccxt.binance()
    since = int(datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    binance_symbol = symbol.split(':')[0]
    filepath = get_history_filepath(binance_symbol, timeframe)
    
    df_existing = None
    if os.path.exists(filepath):
        try:
            df_existing = pd.read_parquet(filepath)
            if not df_existing.empty:
                # Wenn wir schon aktuellere Daten haben, müssen wir nicht erneut laden
                if df_existing.index[-1].timestamp() * 1000 > since:
                     log.info(f"Lokale Daten für {binance_symbol} ({timeframe}) sind bereits aktuell genug.")
                     return df_existing
                # Ansonsten setzen wir den Startpunkt auf das Ende der vorhandenen Daten
                since = int(df_existing.index[-1].timestamp() * 1000) + 1
        except Exception:
            df_existing = None # Fehler beim Lesen, lade neu

    all_data = []
    pbar = tqdm(desc=f"  Download {binance_symbol} ({timeframe})", unit=" Kerzen")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, since=since, limit=1000)
            if not ohlcv: break
            
            first_ts = pd.to_datetime(ohlcv[0][0], unit='ms', utc=True)
            last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
            pbar.set_postfix_str(f"Lade Block: {first_ts.date()} -> {last_ts.date()}")
            
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            pbar.update(len(ohlcv))
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f" Fehler bei Download: {e}"); break
            
    pbar.close()

    if not all_data:
        log.info("Keine neuen Daten zum Herunterladen gefunden.")
        return df_existing if df_existing is not None else pd.DataFrame()

    df_new = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True)
    df_new.set_index('timestamp', inplace=True)

    if df_existing is not None and not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new

    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
    df_combined.to_parquet(filepath)
    
    log.info(f"Erfolgreich! {len(df_combined)} Kerzen für {binance_symbol} ({timeframe}) lokal gespeichert.")
    return df_combined

def get_market_data(exchange, symbol: str, timeframe: str, start_date_str: str):
    """
    NEUE LOGIK V2:
    1. Prüft, ob die lokale Datei die angeforderten Daten (ab start_date_str) enthält.
    2. Wenn nicht, lädt es die fehlenden Daten ab start_date_str herunter.
    3. Holt die allerletzten Live-Daten von der Trading-Börse.
    4. Gibt exakt den angeforderten Zeitraum zurück.
    """
    history_file = get_history_filepath(symbol.split(':')[0], timeframe)
    start_dt = pd.to_datetime(start_date_str, utc=True)
    df_history = None

    if os.path.exists(history_file):
        try:
            df_full_history = pd.read_parquet(history_file)
            # Prüfen, ob der Start der lokalen Daten VOR unserem gewünschten Start liegt
            if not df_full_history.empty and df_full_history.index[0] <= start_dt:
                log.info(f"Ausreichende Historiendaten für {symbol} ({timeframe}) in lokaler Datei gefunden.")
                df_history = df_full_history
            else:
                log.warning(f"Lokale Daten sind nicht alt genug. Lade fehlende Daten ab {start_date_str}...")
                df_history = _download_binance_data(symbol, timeframe, start_date_str)
        except Exception as e:
            log.warning(f"Konnte lokale Historiendatei nicht lesen: {e}. Starte Download.")
            df_history = _download_binance_data(symbol, timeframe, start_date_str)
    else:
        log.warning(f"Keine lokale Historiendatei für {symbol} ({timeframe}) gefunden.")
        print(f"Starte Download für den angeforderten Zeitraum ab {start_date_str}... Dies kann dauern.")
        df_history = _download_binance_data(symbol, timeframe, start_date_str)

    try:
        df_live = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=300)
        if not df_live.empty:
            log.info(f"{len(df_live)} neueste Kerzen von {exchange.exchange.id} geladen.")
    except Exception as e:
        log.error(f"Fehler beim Laden der Live-Daten: {e}")
        df_live = pd.DataFrame()

    if df_history is not None and not df_history.empty:
        df_combined = pd.concat([df_history, df_live])
    else:
        df_combined = df_live
        
    if df_combined.empty:
        log.error("Keine Marktdaten verfügbar.")
        return pd.DataFrame()

    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
    
    # Gib exakt den angeforderten Zeitraum zurück
    return df_combined[df_combined.index >= start_dt]
