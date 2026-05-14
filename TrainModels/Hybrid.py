import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# 1. Load and Clean Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')
df.ffill(inplace=True)
# Define columns
feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = 'prectotcorr'

# 2. Data Preparation & Scaling
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

scaled_x = scaler_x.fit_transform(df[feature_cols])
scaled_y = scaler_y.fit_transform(df[[target_col]])

SEQUENCE_LENGTHS = [1, 7, 14]


def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def prepare_sequences(x_scaled, y_scaled, seq_len):
    if seq_len == 1:
        x_seq = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        y_seq = y_scaled
    else:
        x_seq, y_seq = create_sequences(x_scaled, y_scaled, seq_len)
    return x_seq, y_seq


def build_hybrid(seq_len, feature_count):
    if seq_len < 3:
        return Sequential([
            LSTM(50, return_sequences=False, activation='relu', input_shape=(seq_len, feature_count)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1),
        ])

    pool_size = 2 if seq_len >= 2 else 1
    return Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_len, feature_count)),
        MaxPooling1D(pool_size=pool_size),
        Dropout(0.2),
        LSTM(50, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1),
    ])


def calculate_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))


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

for seq_len in SEQUENCE_LENGTHS:
    X_seq, y_seq = prepare_sequences(scaled_x, scaled_y, seq_len)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = build_hybrid(seq_len, len(feature_cols))
    model.compile(optimizer='adam', loss='mse')

    print(f"Training TCNN-LSTM Hybrid Model ({seq_len} day window)...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

    history_path = history_dir / f"hybrid_{seq_len}d.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)

    nse = calculate_nse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)

    print("\n" + "=" * 40)
    print(f"      HYBRID MODEL PERFORMANCE ({seq_len}d)")
    print("=" * 40)
    print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
    print(f"R-Squared (R2) Score:           {r2:.4f}")
    print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
    print(f"Model Bias:                     {bias:.4f} mm")
    print("=" * 40)

    model_path = models_dir / f"hybrid_{seq_len}d.keras"
    model.save(model_path)

    metrics_payload[f"hybrid_{seq_len}d"] = {
        "nse": float(nse),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
    }

    model_registry[seq_len] = model

metrics_path.write_text(json.dumps(metrics_payload, indent=2))

# 7. Prediction Function
def predict_rainfall_hybrid(input_data_list):
    """
    input_data_list: A list of 7 lists, each containing the 11 features 
    representing the last 7 days of weather.
    """
    # Scaler transformation
    input_scaled = scaler_x.transform(input_data_list)
    input_reshaped = input_scaled.reshape(1, window_size, len(feature_cols))
    
    # Predict and Inverse Scale
    pred_scaled = model.predict(input_reshaped, verbose=0)
    prediction = scaler_y.inverse_transform(pred_scaled)
    
    return max(0, prediction[0][0])

# Example Call:
# You must provide data for the last 7 days to get the prediction for the target day.
# last_7_days_data = [ [year, month, day, ps...], [...], ... (7 total) ]