import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import tensorflow as tf

# 1. Load and Clean Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')
df.ffill(inplace=True)
# Define Features and Target
feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = 'prectotcorr'

X = df[feature_cols].values
y = df[[target_col]].values

# 2. Scaling
# Scaling is critical for LSTM convergence
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

SEQUENCE_LENGTHS = [1, 7, 14]


def create_sequences(data_x, data_y, seq_len):
    x_seq, y_seq = [], []
    for i in range(len(data_x) - seq_len):
        x_seq.append(data_x[i : i + seq_len])
        y_seq.append(data_y[i + seq_len])
    return np.array(x_seq), np.array(y_seq)


def prepare_sequences(x_scaled, y_scaled, seq_len):
    if seq_len == 1:
        x_seq = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        y_seq = y_scaled
    else:
        x_seq, y_seq = create_sequences(x_scaled, y_scaled, seq_len)
    return x_seq, y_seq


history_dir = Path(__file__).resolve().parent / "training_history"
history_dir.mkdir(parents=True, exist_ok=True)
models_dir = Path(__file__).resolve().parent / "saved_models"
models_dir.mkdir(parents=True, exist_ok=True)
metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
metrics_payload = {}
if metrics_path.exists():
    try:
        metrics_payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        metrics_payload = {}

model_registry = {}


def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


for seq_len in SEQUENCE_LENGTHS:
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, seq_len)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = Sequential([
        Bidirectional(LSTM(64, activation='relu', return_sequences=False), input_shape=(seq_len, len(feature_cols))),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')

    print(f"Starting Bi-LSTM training ({seq_len} day window)...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

    history_path = history_dir / f"bilstm_{seq_len}d.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    nse = calculate_nse(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    bias = np.mean(y_pred - y_actual)

    print("\n" + "=" * 40)
    print(f"       MODEL EVALUATION METRICS ({seq_len}d)")
    print("=" * 40)
    print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
    print(f"R-Squared (R2) Score:           {r2:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
    print(f"Model Bias:                     {bias:.2f} mm")
    print("=" * 40)

    model_path = models_dir / f"bilstm_{seq_len}d.keras"
    model.save(model_path)

    metrics_payload[f"bilstm_{seq_len}d"] = {
        "nse": float(nse),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
    }

    model_registry[seq_len] = model

metrics_path.write_text(json.dumps(metrics_payload, indent=2))