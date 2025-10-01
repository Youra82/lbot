# src/lbot/analysis/trainer.py
import os
import sys
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump as joblib_dump
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, create_sequences, create_lstm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_for_symbol(symbol, timeframe, start_date, end_date):
    logging.info(f"Starte LSTM-Trainingsprozess für {symbol} auf {timeframe}...")

    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    data = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=5000)
    
    if data.empty or len(data) < 100:
        logging.warning(f"Nicht genug Daten für {symbol} ({timeframe}). Überspringe.")
        return

    data_with_features = create_ann_features(data[['open', 'high', 'low', 'close', 'volume']])
    logging.info(f"Features für {symbol} ({timeframe}) erstellt. Shape: {data_with_features.shape}")

    feature_columns_to_scale = data_with_features.columns.drop('close')
    scaler = StandardScaler()
    data_with_features[feature_columns_to_scale] = scaler.fit_transform(data_with_features[feature_columns_to_scale])
    logging.info("Daten erfolgreich skaliert.")

    SEQUENCE_LENGTH = 24
    X, y = create_sequences(data_with_features, sequence_length=SEQUENCE_LENGTH)
    
    if len(X) == 0:
        logging.warning(f"Nicht genug Daten für Sequenzen für {symbol} ({timeframe}). Überspringe.")
        return
    logging.info(f"{len(X)} Trainings-Sequenzen erstellt.")

    num_features = X.shape[2]
    model = create_lstm_model(SEQUENCE_LENGTH, num_features)
    
    logging.info(f"Trainiere LSTM-Modell...")
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    models_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(models_dir, f'ann_scaler_{safe_filename}.joblib')
    
    model.save(model_path)
    joblib_dump(scaler, scaler_path)
    
    logging.info(f"Modell und Scaler für {symbol} ({timeframe}) erfolgreich gespeichert.")

def main():
    parser = argparse.ArgumentParser(description="L-Bot LSTM Model Trainer")
    parser.add_argument('--symbols', required=True, type=str, help="Symbole (z.B. 'BTC ETH')")
    parser.add_argument('--timeframes', required=True, type=str, help="Timeframes (z.B. '1h 4h')")
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default=pd.Timestamp.now().strftime('%Y-%m-%d'))
    args = parser.parse_args()

    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                train_for_symbol(symbol, timeframe, args.start_date, args.end_date)
            except Exception as e:
                logging.error(f"FATALER FEHLER beim Training von {symbol} ({timeframe}): {e}", exc_info=True)

if __name__ == "__main__":
    main()
