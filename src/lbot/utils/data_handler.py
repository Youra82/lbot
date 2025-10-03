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

def _download_binance_data(symbol, timeframe, start_date_str='2020-01-01'):
    """
    Private Funktion, die den vollständigen Download-Prozess von Binance durchführt.
    """
    exchange = ccxt.binance()
    since = int(datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    binance_symbol = symbol.split(':')[0]
    filepath = get_history_filepath(binance_symbol, timeframe)
    
    df_existing = None
    if os.path.exists(filepath):
        print(f"  -> Bestehende Datei für {binance_symbol} ({timeframe}) gefunden. Aktualisiere...")
        df_existing = pd.read_parquet(filepath)
        if not df_existing.empty:
            since = int(df_existing.index[-1].timestamp() * 1000) + 1

    all_data = []
    pbar = tqdm(desc=f"  Download {binance_symbol} ({timeframe})", unit=" Kerzen")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            
            first_ts = pd.to_datetime(ohlcv[0][0], unit='ms', utc=True)
            last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
            
            pbar.set_postfix_str(f"Lade Block: {first_ts.date()} -> {last_ts.date()}")
            
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            pbar.update(len(ohlcv))
            
            time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            print(f" Netzwerkfehler: {e}. Warte 30 Sekunden...")
            time.sleep(30)
        except Exception as e:
            print(f" Ein Fehler ist aufgetreten: {e}")
            break
            
    pbar.close()

    if not all_data:
        log.info("Keine neuen Daten zum Herunterladen gefunden. Lokale Datei ist aktuell.")
        return df_existing if df_existing is not None else pd.DataFrame()

    df_new = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True)
    df_new.set_index('timestamp', inplace=True)

    if df_existing is not None and not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new

    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    df_combined.to_parquet(filepath)
    
    log.info(f"Erfolgreich! {len(df_combined)} Kerzen für {binance_symbol} ({timeframe}) lokal gespeichert.")
    return df_combined

def get_market_data(exchange, symbol: str, timeframe: str, start_date_str: str = '2020-01-01'):
    """
    NEUE LOGIK:
    1. Prüft, ob lokale Historiendaten existieren.
    2. Wenn nicht, lädt es sie automatisch von Binance herunter.
    3. Wenn doch, aktualisiert es sie nur mit den neuesten Daten von Binance.
    4. Holt die allerletzten Live-Daten von der Trading-Börse (Bitget).
    5. Kombiniert alles zu einem vollständigen, aktuellen Datensatz.
    """
    history_file = get_history_filepath(symbol.split(':')[0], timeframe)
    
    # Schritt 1 & 2: Lokale Daten prüfen und ggf. herunterladen/aktualisieren
    if not os.path.exists(history_file):
        log.warning(f"Keine lokale Historiendatei für {symbol} ({timeframe}) gefunden.")
        print("Starte vollständigen Download von Binance... Dies kann einige Minuten dauern.")
        df_history = _download_binance_data(symbol, timeframe, '2020-01-01')
    else:
        # Datei existiert, wir rufen die Download-Funktion auf, die nur aktualisiert
        df_history = _download_binance_data(symbol, timeframe, '2020-01-01')

    # Schritt 3: Lade die neuesten Kerzen von der Live-Börse (Bitget)
    try:
        df_live = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=300) 
        if not df_live.empty:
            log.info(f"{len(df_live)} neueste Kerzen von {exchange.exchange.id} geladen.")
    except Exception as e:
        log.error(f"Fehler beim Laden der Live-Daten: {e}")
        df_live = pd.DataFrame()

    # Schritt 4: Kombiniere die Datensätze
    if df_history is not None and not df_history.empty:
        df_combined = pd.concat([df_history, df_live])
    else:
        df_combined = df_live
        
    if df_combined.empty:
        log.error("Keine Marktdaten verfügbar (weder historisch noch live).")
        return pd.DataFrame()

    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    
    # Filtere auf das Startdatum des Backtests (relevant für Trainer/Optimizer)
    start_dt = pd.to_datetime(start_date_str, utc=True)
    return df_combined[df_combined.index >= start_dt]
