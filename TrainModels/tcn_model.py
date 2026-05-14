import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN  # Importing the Temporal Convolutional Network layer

# 1. Load the Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')
df.ffill(inplace=True)

# 2. Define Features and Target
feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = ['prectotcorr']

X = df[feature_cols].values
y = df[target_col].values

# 3. Scaling
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y)

# 4. Reshape for TCN (Samples, Timesteps, Features)
# We use a window size of 1 as per your request to pass a single day's data
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

split = int(len(X_reshaped) * 0.8)
X_train, X_test = X_reshaped[:split], X_reshaped[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# 5. Build TCN Model
model = Sequential([
    TCN(input_shape=(1, len(feature_cols)),
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=[1, 2, 4],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.1,
        return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mae')

# 6. Train
print("Training TCN Model...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

history_dir = Path(__file__).resolve().parent / "training_history"
history_dir.mkdir(parents=True, exist_ok=True)
history_path = history_dir / "tcn.json"
history_path.write_text(json.dumps(history.history, indent=2))

# 7. Evaluation
y_pred_scaled = model.predict(X_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_true = target_scaler.inverse_transform(y_test)

# Metric Calculations
def calculate_nse(y_true, y_pred):
    return 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

nse = calculate_nse(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
bias = np.mean(y_pred - y_true)

print("\n" + "="*40)
print("       TCN PERFORMANCE METRICS")
print("="*40)
print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
print(f"R-Squared (R2) Score:           {r2:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Model Bias:                     {bias:.2f} mm")
print("="*40)

# 8. Save model + metrics for frontend consumption
models_dir = Path(__file__).resolve().parent / "saved_models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "tcn.keras"
model.save(model_path)

metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
metrics_payload = {}
if metrics_path.exists():
    try:
        metrics_payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        metrics_payload = {}

metrics_payload["tcn"] = {
    "nse": float(nse),
    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "bias": float(bias),
}
metrics_path.write_text(json.dumps(metrics_payload, indent=2))

# 8. Prediction Function
def predict_rainfall(year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn):
    data = np.array([[year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn]])
    scaled_data = feature_scaler.transform(data).reshape(1, 1, len(feature_cols))
    pred = model.predict(scaled_data, verbose=0)
    return max(0, target_scaler.inverse_transform(pred)[0][0])

# Example Call:
rain = predict_rainfall(2025, 7, 20, 100.2, 27.5, 31.0, 24.0, 88.0, 3.5, 210.0, 18.5)
print(f"Predicted Rain: {rain:.2f} mm")