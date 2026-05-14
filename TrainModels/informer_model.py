import json
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Load and Prepare Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')
df.ffill(inplace=True)

feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = ['prectotcorr']

# Scaling
f_scaler = MinMaxScaler()
t_scaler = MinMaxScaler()

X_scaled = f_scaler.fit_transform(df[feature_cols])
y_scaled = t_scaler.fit_transform(df[target_col])

SEQUENCE_LENGTHS = [1, 7, 14]


def create_sequences(data_x, data_y, seq_len):
    X, y = [], []
    for i in range(len(data_x) - seq_len):
        X.append(data_x[i : i + seq_len])
        y.append(data_y[i + seq_len])
    return np.array(X), np.array(y)


def prepare_sequences(x_scaled, y_scaled, seq_len):
    if seq_len == 1:
        x_seq = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        y_seq = y_scaled
    else:
        x_seq, y_seq = create_sequences(x_scaled, y_scaled, seq_len)
    return x_seq, y_seq

# 3. Build Transformer Model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # 2 Transformer Blocks
    for _ in range(2):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x) # Target: prectotcorr
    return models.Model(inputs, outputs)

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


def calculate_nse(true, pred):
    return 1 - (np.sum((true - pred)**2) / np.sum((true - np.mean(true))**2))


for seq_len in SEQUENCE_LENGTHS:
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, seq_len)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = build_model((seq_len, len(feature_cols)))
    model.compile(optimizer='adam', loss='mse')

    print(f"Training Transformer Model ({seq_len} day window)...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

    history_path = history_dir / f"transformer_{seq_len}d.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    y_pred_scaled = model.predict(X_test)
    y_pred = t_scaler.inverse_transform(y_pred_scaled)
    y_true = t_scaler.inverse_transform(y_test)

    nse = calculate_nse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)

    print("\n" + "=" * 40)
    print(f"  TRANSFORMER MODEL PERFORMANCE ({seq_len}d)")
    print("=" * 40)
    print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
    print(f"R-Squared (R2) Score:           {r2:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
    print(f"Model Bias:                     {bias:.2f} mm")
    print("=" * 40)

    model_path = models_dir / f"transformer_{seq_len}d.keras"
    model.save(model_path)

    metrics_payload[f"transformer_{seq_len}d"] = {
        "nse": float(nse),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
    }

    model_registry[seq_len] = model

metrics_path.write_text(json.dumps(metrics_payload, indent=2))