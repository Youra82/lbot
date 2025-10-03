# src/lbot/utils/lstm_model.py
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import load as joblib_load

def create_ann_features(df_in, ema_period=200, atr_period=14):
    """ Erstellt technische Indikatoren, inkl. Trend-, Vola- und Momentum-Features. """
    df = df_in.copy()
    
    # Standard-Indikatoren
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # Trend-Filter EMA (Periode wird dynamisch von Optuna gesetzt)
    df[f'ema_{ema_period}'] = ta.trend.EMAIndicator(close=df['close'], window=ema_period).ema_indicator()
    
    # Volatilitäts-Filter ATR (Periode wird dynamisch von Optuna gesetzt)
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df[f'atr_{atr_period}'] = atr_indicator.average_true_range()
    # Normalisierter ATR (ATR geteilt durch den Schlusskurs) für bessere Vergleichbarkeit
    df[f'natr_{atr_period}'] = (df[f'atr_{atr_period}'] / df['close']) * 100

    # NEU: Zusätzliche Features für das Modell
    # Trendstärke
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    # Momentum / Oszillator
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df.dropna(inplace=True)
    return df

def create_sequences(data, sequence_length, future_steps, threshold):
    """
    Wandelt einen DataFrame von Features in Sequenzen (X) und ein binäres Ziel (y) um.
    """
    data['future_price'] = data['close'].shift(-future_steps)
    data['target'] = (data['future_price'] / data['close'] - 1) > threshold
    data.dropna(inplace=True)
    
    # Wir entfernen Features, die das Modell nicht sehen soll (inkl. der neuen Filter-Indikatoren)
    feature_columns = data.columns.drop(['close', 'future_price', 'target'], errors='ignore')
    
    X, y = [], []
    if len(data) > sequence_length:
        for i in range(len(data) - sequence_length):
            X.append(data.iloc[i:(i + sequence_length)][feature_columns].values)
            y.append(data.iloc[i + sequence_length]['target'])
        
    return np.array(X), np.array(y)

def create_lstm_model(sequence_length, num_features):
    """ Erstellt ein LSTM-Modell. """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_model_and_scaler(model_path, scaler_path):
    """ Lädt ein gespeichertes Modell und einen Scaler. """
    try:
        model = load_model(model_path)
        scaler = joblib_load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Fehler beim Laden von Modell/Scaler: {e}")
        return None, None
