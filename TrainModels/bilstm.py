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

# 3. Reshape for Bi-LSTM [samples, time_steps, features]
# We use 1 time step to allow for single-day input prediction
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split into Training (80%) and Testing (20%)
split = int(len(X_reshaped) * 0.8)
X_train, X_test = X_reshaped[:split], X_reshaped[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# 4. Build Bidirectional LSTM Model
model = Sequential([
    Bidirectional(LSTM(64, activation='relu', return_sequences=False), input_shape=(1, len(feature_cols))),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) # Final output for rainfall
])

model.compile(optimizer='adam', loss='mse')

# 5. Training
print("Starting Bi-LSTM training...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

history_dir = Path(__file__).resolve().parent / "training_history"
history_dir.mkdir(parents=True, exist_ok=True)
history_path = history_dir / "bilstm.json"
history_path.write_text(json.dumps(history.history, indent=2))

# 6. Evaluation Logic
def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# Predict on test set
y_pred_scaled = model.predict(X_test)

# Inverse transform to original scale (mm)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

# Calculate Metrics
nse = calculate_nse(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
bias = np.mean(y_pred - y_actual)

# 7. Print Results
print("\n" + "="*40)
print("       MODEL EVALUATION METRICS")
print("="*40)
print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
print(f"R-Squared (R2) Score:           {r2:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Model Bias:                     {bias:.2f} mm")
print("="*40)

# 7. Save model + metrics for frontend consumption
models_dir = Path(__file__).resolve().parent / "saved_models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "bilstm.keras"
model.save(model_path)

metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
metrics_payload = {}
if metrics_path.exists():
    try:
        metrics_payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        metrics_payload = {}

metrics_payload["bilstm"] = {
    "nse": float(nse),
    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "bias": float(bias),
}
metrics_path.write_text(json.dumps(metrics_payload, indent=2))

# 8. Prediction Function for Custom Input
def predict_prectotcorr(year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn):
    input_data = np.array([[year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn]])
    scaled_input = scaler_X.transform(input_data).reshape((1, 1, len(feature_cols)))
    pred_scaled = model.predict(scaled_input, verbose=0)
    prediction = scaler_y.inverse_transform(pred_scaled)
    return max(0, prediction[0][0])