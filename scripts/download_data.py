# scripts/download_data.py
import os
import sys
import argparse
import pandas as pd
import ccxt
from datetime import datetime, timezone
from tqdm import tqdm

# Füge das Hauptverzeichnis zum Pfad hinzu, um lbot-Module zu finden
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.lbot.utils.data_handler import get_cache_filepath # Wir nutzen die gleiche Logik für Dateipfade

def download_all_data(symbol, timeframe, start_date_str='2020-01-01'):
    """
    Lädt historische OHLCV-Daten von Binance in Blöcken und speichert sie.
    Aktualisiert vorhandene Dateien, anstatt alles neu zu laden.
    """
    # Wir verwenden Binance für den Download, da sie die beste Historie haben
    exchange = ccxt.binance()
    
    # Konvertiere das Startdatum in einen Millisekunden-Timestamp
    since = int(datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    # Passe den Symbolnamen für Binance an (z.B. BTC/USDT:USDT -> BTC/USDT)
    binance_symbol = symbol.split(':')[0]
    
    # Pfad zur Zieldatei
    filepath = get_cache_filepath(binance_symbol, timeframe)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prüfe, ob bereits Daten vorhanden sind und setze den Startpunkt
    if os.path.exists(filepath):
        print(f"Bestehende Datei gefunden für {binance_symbol} ({timeframe}). Lade nur neue Daten...")
        df_existing = pd.read_parquet(filepath)
        if not df_existing.empty:
            since = int(df_existing.index[-1].timestamp() * 1000) + 1

    all_data = []
    
    # TQDM für den Fortschrittsbalken initialisieren
    pbar = tqdm(desc=f"Download {binance_symbol} ({timeframe})")
    
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
            
            # Respektiere die Rate Limits von Binance
            time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            print(f"Netzwerkfehler: {e}. Warte 30 Sekunden...")
            time.sleep(30)
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")
            break
            
    pbar.close()

    if not all_data:
        print("Keine neuen Daten zum Herunterladen gefunden.")
        return

    df_new = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms', utc=True)
    df_new.set_index('timestamp', inplace=True)

    # Kombiniere alte und neue Daten
    if 'df_existing' in locals() and not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new])
    else:
        df_combined = df_new

    # Duplikate entfernen und speichern
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)
    df_combined.to_parquet(filepath)
    
    print(f"Erfolgreich! {len(df_combined)} Kerzen für {binance_symbol} ({timeframe}) gespeichert in {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L-Bot Daten-Downloader")
    parser.add_argument('--symbols', required=True, type=str, help="Symbole, getrennt durch Leerzeichen (z.B. 'BTC ETH')")
    parser.add_argument('--timeframes', required=True, type=str, help="Timeframes, getrennt durch Leerzeichen (z.B. '1h 4h')")
    parser.add_argument('--start_date', type=str, default='2020-01-01', help="Startdatum im Format JJJJ-MM-TT")
    args = parser.parse_args()

    symbols_to_download = [s.upper() + "/USDT" for s in args.symbols.split()]
    timeframes_to_download = args.timeframes.split()

    for symbol in symbols_to_download:
        for timeframe in timeframes_to_download:
            download_all_data(symbol, timeframe, args.start_date)
