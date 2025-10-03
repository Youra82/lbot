# src/lbot/utils/data_handler.py
import os
import pandas as pd
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# NEU: Der primäre Speicherort für die tiefen historischen Daten
HISTORY_DIR = os.path.join(PROJECT_ROOT, 'data', 'history')
os.makedirs(HISTORY_DIR, exist_ok=True)

log = logging.getLogger("DataHandler")
log.setLevel(logging.INFO)

def get_cache_filepath(symbol, timeframe):
    """ Erstellt einen sicheren Dateinamen für die Cache-Datei. """
    safe_symbol = symbol.replace('/', '_').replace(':', '')
    return os.path.join(HISTORY_DIR, f"{safe_symbol}_{timeframe}.parquet")

def get_market_data(exchange, symbol: str, timeframe: str, start_date_str: str = '2020-01-01'):
    """
    NEUE LOGIK:
    1. Versucht, tiefe historische Daten aus dem lokalen Speicher zu laden.
    2. Holt nur die allerletzten, fehlenden Kerzen von der Live-Börse (Bitget).
    """
    history_file = get_cache_filepath(symbol.split(':')[0], timeframe)
    df_history = None
    
    # 1. Lade tiefe Historiendaten, falls vorhanden
    if os.path.exists(history_file):
        try:
            df_history = pd.read_parquet(history_file)
            log.info(f"Tiefe Historiendaten für {symbol} ({timeframe}) aus lokaler Datei geladen: {len(df_history)} Kerzen.")
        except Exception as e:
            log.warning(f"Konnte lokale Historiendatei nicht lesen: {e}")
    else:
        log.warning(f"Keine lokale Historiendatei für {symbol} ({timeframe}) gefunden. Performance wird schlecht sein.")
        log.warning("Bitte führe 'bash ./download_data.sh' aus, um die Daten herunterzuladen.")

    # 2. Lade die neuesten Kerzen von der Live-Börse (Bitget)
    try:
        # Lade etwas mehr als die Sequenzlänge, um eine gute Überlappung zu gewährleisten
        df_live = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=300) 
        if not df_live.empty:
            log.info(f"{len(df_live)} neueste Kerzen von {exchange.exchange.id} geladen.")
        else:
            log.info("Keine neuen Live-Daten von der Börse erhalten.")
    except Exception as e:
        log.error(f"Fehler beim Laden der Live-Daten: {e}")
        df_live = pd.DataFrame()


    # 3. Kombiniere die Datensätze
    if df_history is not None and not df_history.empty:
        df_combined = pd.concat([df_history, df_live])
    else:
        df_combined = df_live
        
    if df_combined.empty:
        log.error("Keine Marktdaten verfügbar (weder historisch noch live).")
        return pd.DataFrame()

    # Duplikate entfernen und sicherstellen, dass alles sortiert ist
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    
    # Filtere auf das Startdatum des Backtests (relevant für Trainer/Optimizer)
    start_dt = pd.to_datetime(start_date_str, utc=True)
    return df_combined[df_combined.index >= start_dt]
