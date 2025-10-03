# src/lbot/utils/mc_dropout_predictor.py
import numpy as np
import tensorflow as tf

def make_mc_prediction(model, data, n_samples=30):
    """
    Führt Monte-Carlo-Dropout-Vorhersagen durch, um eine robustere Schätzung
    und ein Maß für die Unsicherheit zu erhalten.
    """
    # Stelle sicher, dass die Daten ein Tensor sind
    tf_data = tf.convert_to_tensor(data, dtype=tf.float32)
    
    predictions = []
    for _ in range(n_samples):
        # Rufe das Modell mit training=True auf, um Dropout zu aktivieren
        prediction = model(tf_data, training=True)
        predictions.append(prediction.numpy())
    
    predictions = np.array(predictions).flatten()
    
    # Berechne Mittelwert und Standardabweichung
    mean_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)
    
    return mean_prediction, std_prediction
