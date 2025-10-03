# src/lbot/utils/lstm_model.py
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import load as joblib_load

def create_ann_features(df_in, ema_period=200, atr_period=14):
    # ... (Diese Funktion bleibt unverändert) ...
    df = df_in.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df[f'ema_{ema_period}'] = ta.trend.EMAIndicator(close=df['close'], window=ema_period).ema_indicator()
    atr_indicator = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period)
    df[f'atr_{atr_period}'] = atr_indicator.average_true_range()
    df[f'natr_{atr_period}'] = (df[f'atr_{atr_period}'] / df['close']) * 100
    df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df.dropna(inplace=True)
    return df

def create_sequences(data, sequence_length, future_steps):
    """
    MODIFIZIERT FÜR REGRESSION:
    Wandelt einen DataFrame in Sequenzen (X) und ein numerisches Ziel (y) um.
    Das Ziel 'y' ist jetzt die tatsächliche prozentuale Preisänderung.
    """
    data['future_price'] = data['close'].shift(-future_steps)
    
    # MODIFIZIERT: Ziel ist jetzt die prozentuale Änderung, nicht mehr True/False
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
    """
    MODIFIZIERT FÜR REGRESSION:
    - Letzte Aktivierungsfunktion ist 'linear' (kann jeden Wert ausgeben).
    - Verlustfunktion ist 'mean_squared_error' (Standard für Regression).
    - Metrik ist 'mae' (Mean Absolute Error) statt 'accuracy'.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        # MODIFIZIERT: 'linear' für die Vorhersage eines kontinuierlichen Wertes
        Dense(1, activation='linear')
    ])
    
    # MODIFIZIERT: Verlustfunktion und Metriken für Regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def load_model_and_scaler(model_path, scaler_path):
    # ... (Diese Funktion bleibt unverändert) ...
    try:
        model = load_model(model_path)
        scaler = joblib_load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Fehler beim Laden von Modell/Scaler: {e}")
        return None, None
