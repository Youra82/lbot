# src/lbot/utils/data_handler.py
import os
import pandas as pd
from datetime import datetime, timezone
import logging

# --- Pfad-Konfiguration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

log = logging.getLogger("DataHandler")

def get_cache_filepath(symbol, timeframe):
    """ Erstellt einen sicheren Dateinamen für die Cache-Datei. """
    safe_symbol = symbol.replace('/', '_').replace(':', '')
    return os.path.join(CACHE_DIR, f"{safe_symbol}_{timeframe}.parquet")

def get_market_data(exchange, symbol: str, timeframe: str, start_date_str: str = '2020-01-01'):
    """
    Lädt Marktdaten, nutzt Caching, um API-Anfragen zu minimieren.
    Holt neue Daten und fügt sie dem Cache hinzu.
    """
    cache_file = get_cache_filepath(symbol, timeframe)
    df_cached = None
    
    # 1. Versuche, Daten aus dem Cache zu laden
    if os.path.exists(cache_file):
        try:
            df_cached = pd.read_parquet(cache_file)
            if not df_cached.empty and isinstance(df_cached.index, pd.DatetimeIndex):
                log.info(f"Daten für {symbol} ({timeframe}) aus Cache geladen. {len(df_cached)} Kerzen gefunden.")
            else:
                df_cached = None # Invalider Cache
        except Exception as e:
            log.warning(f"Konnte Cache-Datei {cache_file} nicht lesen: {e}. Lade alle Daten neu.")
            df_cached = None

    # 2. Bestimme, ab wann neue Daten benötigt werden
    since_timestamp = None
    if df_cached is not None and not df_cached.empty:
        last_timestamp = df_cached.index[-1]
        # +1 Millisekunde, um die letzte Kerze nicht doppelt zu laden
        since_timestamp = int(last_timestamp.timestamp() * 1000) + 1 
    else:
        # Wenn kein Cache da ist, lade ab dem konfigurierten Startdatum
        start_dt = pd.to_datetime(start_date_str, utc=True)
        since_timestamp = int(start_dt.timestamp() * 1000)

    # 3. Lade neue Daten von der Börse
    log.info(f"Lade neue Daten für {symbol} ({timeframe}) seit {pd.to_datetime(since_timestamp, unit='ms', utc=True)}...")
    df_new = exchange.fetch_ohlcv_since(symbol, timeframe, since=since_timestamp, limit=5000)
    
    if df_new.empty:
        log.info("Keine neuen Daten gefunden.")
        return df_cached if df_cached is not None else pd.DataFrame()

    # 4. Kombiniere gecachte und neue Daten
    if df_cached is not None:
        df_combined = pd.concat([df_cached, df_new])
    else:
        df_combined = df_new
        
    # 5. Bereinige Duplikate und speichere zurück in den Cache
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    
    try:
        df_combined.to_parquet(cache_file)
        log.info(f"{len(df_new)} neue Kerzen zu Cache hinzugefügt. Gesamt: {len(df_combined)}.")
    except Exception as e:
        log.error(f"Fehler beim Speichern der Cache-Datei {cache_file}: {e}")
        
    return df_combined
