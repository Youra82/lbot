# src/lbot/utils/lstm_model.py
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import load as joblib_load

# HINWEIS: Die Perioden werden jetzt hier fest definiert.
EMA_SHORT_PERIOD = 20
EMA_MEDIUM_PERIOD = 50
EMA_LONG_PERIOD = 200
ATR_PERIOD = 14
RSI_EMA_PERIOD = 21

def create_ann_features(df_in):
    """ Erstellt ein festes Set von technischen Indikatoren und relativen Features. """
    df = df_in.copy()
    
    # --- Basis-Indikatoren ---
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()

    # --- Filter-Indikatoren ---
    df[f'ema_{EMA_LONG_PERIOD}'] = ta.trend.EMAIndicator(close=df['close'], window=EMA_LONG_PERIOD).ema_indicator()
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_PERIOD)
    df[f'atr_{ATR_PERIOD}'] = atr_indicator.average_true_range()
    df[f'natr_{ATR_PERIOD}'] = (df[f'atr_{ATR_PERIOD}'] / df['close']) * 100
    
    # --- Relative Features ---
    df[f'ema_{EMA_SHORT_PERIOD}'] = ta.trend.EMAIndicator(close=df['close'], window=EMA_SHORT_PERIOD).ema_indicator()
    df[f'ema_{EMA_MEDIUM_PERIOD}'] = ta.trend.EMAIndicator(close=df['close'], window=EMA_MEDIUM_PERIOD).ema_indicator()
    
    df['price_vs_ema_short'] = (df['close'] / df[f'ema_{EMA_SHORT_PERIOD}'] - 1) * 100
    df['price_vs_ema_medium'] = (df['close'] / df[f'ema_{EMA_MEDIUM_PERIOD}'] - 1) * 100
    
    df['ema_rsi'] = ta.trend.EMAIndicator(close=df['rsi'], window=RSI_EMA_PERIOD).ema_indicator()
    df['rsi_vs_ema_rsi'] = df['rsi'] - df['ema_rsi']

    # Lösche die reinen EMA-Hilfsspalten
    df.drop(columns=[f'ema_{EMA_SHORT_PERIOD}', f'ema_{EMA_MEDIUM_PERIOD}', 'ema_rsi'], inplace=True)

    df.dropna(inplace=True)
    return df

# ... (Rest der Datei bleibt unverändert) ...
def create_sequences(data, sequence_length, future_steps):
    data['future_price'] = data['close'].shift(-future_steps)
    data['target'] = (data['future_price'] / data['close']) - 1
    data.dropna(inplace=True)
    feature_columns = data.columns.drop(['close', 'future_price', 'target'], errors='ignore')
    X, y = [], []
    if len(data) > sequence_length:
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:(i + sequence_length)][feature_columns].values)
            y.append(data.iloc[i + sequence_length]['target'])
    return np.array(X), np.array(y)

def create_lstm_model(sequence_length, num_features):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def load_model_and_scaler(model_path, scaler_path):
    try:
        model = load_model(model_path)
        scaler = joblib_load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Fehler beim Laden von Modell/Scaler: {e}")
        return None, None
