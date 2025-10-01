# src/lbot/analysis/trainer.py
import os
import sys
import argparse
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump as joblib_dump
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from lbot.utils.exchange import Exchange
from lbot.utils.lstm_model import create_ann_features, create_sequences, create_lstm_model
from lbot.utils.data_handler import get_market_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_settings():
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r') as f:
        return json.load(f)

def train_for_symbol(symbol, timeframe, start_date, settings):
    logging.info(f"Starte LSTM-Trainingsprozess für {symbol} auf {timeframe}...")
    
    model_conf = settings['model_settings']

    dummy_account = {'apiKey': 'dummy', 'secret': 'dummy', 'password': 'dummy'}
    exchange = Exchange(dummy_account)
    
    # Lade Daten über den neuen Data Handler
    data = get_market_data(exchange, symbol, timeframe, start_date)
    
    if data.empty or len(data) < 100:
        logging.warning(f"Nicht genug Daten für {symbol} ({timeframe}). Überspringe.")
        return

    data_with_features = create_ann_features(data[['open', 'high', 'low', 'close', 'volume']])
    logging.info(f"Features für {symbol} ({timeframe}) erstellt. Shape: {data_with_features.shape}")

    feature_columns_to_scale = data_with_features.columns.drop('close')
    scaler = StandardScaler()
    data_with_features[feature_columns_to_scale] = scaler.fit_transform(data_with_features[feature_columns_to_scale])
    logging.info("Daten erfolgreich skaliert.")

    # Verwende Parameter aus der Konfigurationsdatei
    X, y = create_sequences(
        data=data_with_features, 
        sequence_length=model_conf['sequence_length'],
        future_steps=model_conf['future_steps'],
        threshold=model_conf['prediction_threshold_classify']
    )
    
    if len(X) == 0:
        logging.warning(f"Nicht genug Daten für Sequenzen für {symbol} ({timeframe}). Überspringe.")
        return
    logging.info(f"{len(X)} Trainings-Sequenzen erstellt.")

    num_features = X.shape[2]
    model = create_lstm_model(model_conf['sequence_length'], num_features)
    
    logging.info(f"Trainiere LSTM-Modell...")
    model.fit(
        X, y, 
        epochs=model_conf['epochs'], 
        batch_size=model_conf['batch_size'], 
        validation_split=model_conf['validation_split'], 
        verbose=1
    )

    # Speichere Modell und Scaler
    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    models_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(models_dir, f'ann_scaler_{safe_filename}.joblib')
    
    model.save(model_path)
    joblib_dump(scaler, scaler_path)
    
    logging.info(f"Modell und Scaler für {symbol} ({timeframe}) erfolgreich gespeichert.")

def main():
    settings = load_settings()
    
    parser = argparse.ArgumentParser(description="L-Bot LSTM Model Trainer")
    parser.add_argument('--symbols', required=True, type=str, help="Symbole (z.B. 'BTC ETH')")
    parser.add_argument('--timeframes', required=True, type=str, help="Timeframes (z.B. '1h 4h')")
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    args = parser.parse_args()

    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()

    for symbol in symbols:
        for timeframe in timeframes:
            try:
                train_for_symbol(symbol, timeframe, args.start_date, settings)
            except Exception as e:
                logging.error(f"FATALER FEHLER beim Training von {symbol} ({timeframe}): {e}", exc_info=True)

if __name__ == "__main__":
    main()
