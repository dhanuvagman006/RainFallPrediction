import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# 1. Load and Prepare Data
df = pd.read_csv('TrainModels/dakshina_kannada_rainfall_daily_2000_2024.csv')
df.ffill(inplace=True)

feature_cols = ['year', 'month', 'day', 'ps', 't2m', 't2m_max', 't2m_min', 'rh2m', 'ws2m', 'wd2m', 'allsky_sfc_sw_dwn']
target_col = ['prectotcorr']

X = df[feature_cols].values
y = df[target_col].values

# 2. Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshape for GRU: [samples, time_steps, features]
# Using 1 time_step to match your input-output requirement
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split: 80% Train, 20% Test
split = int(len(X_reshaped) * 0.8)
X_train, X_test = X_reshaped[:split], X_reshaped[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# 3. Build the GRU Model
model = Sequential([
    GRU(64, activation='relu', input_shape=(1, X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) # Final output for rainfall
])

model.compile(optimizer='adam', loss='mse')

# 4. Training
print("Starting GRU Training...")
history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.1, verbose=1)

history_dir = Path(__file__).resolve().parent / "training_history"
history_dir.mkdir(parents=True, exist_ok=True)
history_path = history_dir / "gru.json"
history_path.write_text(json.dumps(history.history, indent=2))

# 5. Model Evaluation Metrics
y_pred_scaled = model.predict(X_test)

# Inverse transform to original units (mm)
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

def calculate_metrics(y_true, y_pred):
    # Nash-Sutcliffe Efficiency (NSE)
    nse = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    # R2 Score
    r2 = r2_score(y_true, y_pred)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Model Bias
    bias = np.mean(y_pred - y_true)
    
    return nse, r2, mae, rmse, bias

nse, r2, mae, rmse, bias = calculate_metrics(y_test_inv, y_pred_inv)

print("\n" + "="*40)
print("       GRU PERFORMANCE METRICS")
print("="*40)
print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
print(f"R-Squared (R2) Score:           {r2:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Model Bias:                     {bias:.2f} mm")
print("="*40)

# 6. Save model + metrics for frontend consumption
models_dir = Path(__file__).resolve().parent / "saved_models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "gru.keras"
model.save(model_path)

metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
metrics_payload = {}
if metrics_path.exists():
    try:
        metrics_payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        metrics_payload = {}

metrics_payload["gru"] = {
    "nse": float(nse),
    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "bias": float(bias),
}
metrics_path.write_text(json.dumps(metrics_payload, indent=2))

# 6. Prediction Function
def predict_rainfall(year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn):
    inputs = np.array([[year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn]])
    inputs_scaled = scaler_X.transform(inputs).reshape((1, 1, 11))
    prediction_scaled = model.predict(inputs_scaled, verbose=0)
    prediction_final = scaler_y.inverse_transform(prediction_scaled)
    return max(0, prediction_final[0][0])