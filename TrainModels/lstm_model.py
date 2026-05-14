import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# 1. Load the Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')

# Handle any potential missing values by forward-filling
df.ffill(inplace=True)

# 2. Define Features and Target
feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = ['prectotcorr']

X = df[feature_cols].values
y = df[target_col].values

# 3. Scale the Data
# Neural networks (especially LSTMs) require scaled data to converge properly.
# We use separate scalers so we can easily reverse the scaling on our final prediction.
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = feature_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y)

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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)


for seq_len in SEQUENCE_LENGTHS:
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, seq_len)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(seq_len, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f"Training the LSTM model ({seq_len} day window)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    history_path = history_dir / f"lstm_{seq_len}d.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    y_pred_scaled = model.predict(X_test)
    y_pred_rescaled = target_scaler.inverse_transform(y_pred_scaled)
    y_test_rescaled = target_scaler.inverse_transform(y_test)

    nse = calculate_nse(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    bias = np.mean(y_pred_rescaled - y_test_rescaled)

    print("\n" + "=" * 30)
    print(f"  LSTM PERFORMANCE REPORT ({seq_len}d)")
    print("=" * 30)
    print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
    print(f"R-Squared (R2) Score:           {r2:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
    print(f"Model Bias:                     {bias:.2f} mm")
    print("=" * 30)

    model_path = models_dir / f"lstm_{seq_len}d.keras"
    model.save(model_path)

    metrics_payload[f"lstm_{seq_len}d"] = {
        "nse": float(nse),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
    }

    model_registry[seq_len] = model

metrics_path.write_text(json.dumps(metrics_payload, indent=2))