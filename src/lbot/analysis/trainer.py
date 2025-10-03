# src/lbot/analysis/trainer.py
import os
import sys
import argparse
import json
from sklearn.preprocessing import StandardScaler
from joblib import dump as joblib_dump
import logging
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

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
    
    model_conf = settings.get('model_settings', {})
    exchange = Exchange({'apiKey': 'dummy', 'secret': 'dummy'})
    data = get_market_data(exchange, symbol, timeframe, start_date)
    
    if data.empty or len(data) < 400:
        logging.warning(f"Nicht genug Daten für {symbol} ({timeframe}). Überspringe.")
        return

    data_with_features = create_ann_features(data)
    
    # KORREKTE REIHENFOLGE:
    # 1. Definiere die Spalten, die das Modell lernen soll (ohne Filter-Indikatoren)
    model_feature_columns = ['rsi', 'adx', 'stoch_k', 'price_vs_ema_short', 'price_vs_ema_medium', 'rsi_vs_ema_rsi']
    features_to_scale = data_with_features[model_feature_columns]

    # 2. Trainiere den Scaler auf dem 2D-DataFrame
    scaler = StandardScaler()
    scaled_feature_values = scaler.fit_transform(features_to_scale)
    
    # Erstelle einen DataFrame mit den skalierten Werten und den Original-Spaltennamen
    scaled_features_df = pd.DataFrame(scaled_feature_values, index=features_to_scale.index, columns=features_to_scale.columns)
    
    # Füge die unskalierten Spalten (für Filter und Ziel) wieder hinzu
    full_df_for_sequences = pd.concat([scaled_features_df, data_with_features.drop(columns=model_feature_columns)], axis=1)

    # 3. Erstelle die Sequenzen aus den jetzt skalierten Daten
    X, y = create_sequences(
        data=full_df_for_sequences, 
        sequence_length=model_conf.get('sequence_length', 24),
        future_steps=model_conf.get('future_steps', 5)
    )
    
    if len(X) == 0:
        logging.warning(f"Nicht genug Daten für Sequenzen. Überspringe.")
        return
    logging.info(f"{len(X)} Trainings-Sequenzen erstellt.")

    num_features = X.shape[2]
    model = create_lstm_model(model_conf.get('sequence_length', 24), num_features)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, mode='min')

    logging.info(f"Trainiere LSTM-Regressions-Modell mit Early Stopping...")
    model.fit(
        X, y, 
        epochs=model_conf.get('epochs', 50), 
        batch_size=model_conf.get('batch_size', 32), 
        validation_split=model_conf.get('validation_split', 0.1), 
        callbacks=[early_stopping],
        verbose=1
    )

    safe_filename = f"{symbol.replace('/', '').replace(':', '')}_{timeframe}"
    models_dir = os.path.join(PROJECT_ROOT, 'artifacts', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f'ann_predictor_{safe_filename}.h5')
    scaler_path = os.path.join(models_dir, f'ann_scaler_{safe_filename}.joblib')
    model.save(model_path)
    joblib_dump(scaler, scaler_path)
    logging.info(f"Modell und Scaler erfolgreich gespeichert.")

def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description="L-Bot LSTM Model Trainer")
    parser.add_argument('--symbols', required=True, type=str)
    parser.add_argument('--timeframes', required=True, type=str)
    parser.add_argument('--start_date', type=str, default='2020-01-01')
    args = parser.parse_args()
    symbols = [s.upper() + "/USDT:USDT" for s in args.symbols.split()]
    timeframes = args.timeframes.split()
    total_jobs = len(symbols) * len(timeframes)
    job_count = 0
    for symbol in symbols:
        for timeframe in timeframes:
            job_count += 1
            logging.info(f"--- Paket {job_count}/{total_jobs}: Start für {symbol} ({timeframe}) ---")
            try:
                train_for_symbol(symbol, timeframe, args.start_date, settings)
            except Exception as e:
                logging.error(f"FATALER FEHLER bei {symbol} ({timeframe}): {e}", exc_info=True)

if __name__ == "__main__":
    main()
