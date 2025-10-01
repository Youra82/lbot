# src/lbot/utils/lstm_model.py
import pandas as pd
import numpy as np
import ta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from joblib import load as joblib_load

def create_ann_features(df_in, ema_period=200):
    """ Erstellt technische Indikatoren, zeitbasierte Features und den Trend-Filter. """
    df = df_in.copy()
    
    # Technische Indikatoren
    indicator_bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
    df['bb_width'] = indicator_bb.bollinger_wband()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    indicator_macd = ta.trend.MACD(close=df['close'])
    df['macd_diff'] = indicator_macd.macd_diff()
    
    # Zeitbasierte Features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['returns_lag1'] = df['close'].pct_change(1)
    
    # NEU: Trend-Filter EMA
    df[f'ema_{ema_period}'] = ta.trend.EMAIndicator(close=df['close'], window=ema_period).ema_indicator()
    
    df.dropna(inplace=True)
    return df

def create_sequences(data, sequence_length, future_steps, threshold):
    """
    Wandelt einen DataFrame von Features in Sequenzen (X) und ein binäres Ziel (y) um.
    """
    data['future_price'] = data['close'].shift(-future_steps)
    data['target'] = (data['future_price'] / data['close'] - 1) > threshold
    data.dropna(inplace=True)
    
    # 'close' wird für die Vorhersage nicht benötigt, der EMA aber schon
    feature_columns = data.columns.drop(['future_price', 'target'], errors='ignore')
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:(i + sequence_length)][feature_columns].values)
        y.append(data.iloc[i + sequence_length]['target'])
        
    return np.array(X), np.array(y)

def create_lstm_model(sequence_length, num_features):
    """ Erstellt ein LSTM-Modell. """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
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
