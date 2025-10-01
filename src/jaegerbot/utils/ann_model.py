# src/jaegerbot/utils/ann_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import logging
import ta
import os

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def create_ann_features(df):
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_width'] = bollinger.bollinger_wband()
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['obv'] = ta.volume.on_balance_volume(close=df['close'], volume=df['volume'])
    else:
        df['obv'] = 0
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_diff'] = macd.macd_diff()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['returns_lag1'] = df['close'].pct_change().shift(1)
    df['returns_lag2'] = df['close'].pct_change().shift(2)
    return df

def prepare_data_for_ann(df, lookahead=5, threshold=0.005):
    df = create_ann_features(df)
    future_returns = df['close'].pct_change(periods=lookahead).shift(-lookahead)
    df['target'] = 0 
    df.loc[future_returns > threshold, 'target'] = 1
    df.loc[future_returns < -threshold, 'target'] = -1
    df = df[df['target'] != 0].copy()
    df['target'] = df['target'].replace(-1, 0)
    feature_cols = ['bb_width', 'obv', 'rsi', 'macd_diff', 'hour', 'day_of_week', 'returns_lag1', 'returns_lag2']
    df.dropna(inplace=True)
    if df.empty: return pd.DataFrame(), pd.Series()
    X = df[feature_cols]
    y = df['target']
    return X, y

def build_and_train_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)
    return model

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Modell & Scaler gespeichert.")

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"Modell & Scaler geladen.")
        return model, scaler
    except Exception as e:
        logger.error(f"Fehler beim Laden von Modell/Scaler: {e}")
        return None, None
