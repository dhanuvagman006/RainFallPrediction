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

# 4. Reshape for LSTM
# LSTMs expect a 3D input shape: (number_of_samples, time_steps, number_of_features)
# Since you want to predict using a single day's input, time_steps = 1
time_steps = 1
X_reshaped = X_scaled.reshape((X_scaled.shape[0], time_steps, X_scaled.shape[1]))

# Split into training and testing sets (80% train, 20% test)
split_idx = int(len(X_reshaped) * 0.8)
X_train, X_test = X_reshaped[:split_idx], X_reshaped[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

# 5. Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2)) # Helps prevent overfitting
model.add(Dense(units=1)) # Single output node for 'prectotcorr'

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the Model
print("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
print("Training complete!")

history_dir = Path(__file__).resolve().parent / "training_history"
history_dir.mkdir(parents=True, exist_ok=True)
history_path = history_dir / "lstm.json"
history_path.write_text(json.dumps(history.history, indent=2))

# 7. Define the Prediction Function
def predict_rainfall(year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn):
    """
    Takes daily features as input and returns the predicted corrected total precipitation.
    """
    # Create a NumPy array from the inputs
    input_data = np.array([[year, month, day, ps, t2m, t2m_max, t2m_min, rh2m, ws2m, wd2m, allsky_sfc_sw_dwn]])
    
    # Scale the input features using the fitted feature_scaler
    input_scaled = feature_scaler.transform(input_data)
    
    # Reshape to (1 sample, 1 time_step, 11 features)
    input_reshaped = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    
    # Generate the prediction
    predicted_scaled = model.predict(input_reshaped, verbose=0)
    
    # Inverse the scaling to get the real rainfall value
    predicted_rainfall = target_scaler.inverse_transform(predicted_scaled)
    
    # Ensure the rainfall doesn't output a negative number
    final_prediction = max(0, predicted_rainfall[0][0])
    
    return final_prediction

# ==========================================
# Example Usage:
# ==========================================
# Assuming these are the weather conditions for a specific day:
predicted_precip = predict_rainfall(
    year=2025, 
    month=6, 
    day=15, 
    ps=100.5, 
    t2m=28.5, 
    t2m_max=32.1, 
    t2m_min=25.0, 
    rh2m=85.0, 
    ws2m=4.5, 
    wd2m=180.0, 
    allsky_sfc_sw_dwn=20.5
)

print(f"\nPredicted Rainfall (prectotcorr): {predicted_precip:.2f} mm")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. Generate predictions on the test set
y_pred_scaled = model.predict(X_test)

# 2. Inverse transform back to original scale (mm)
y_pred_rescaled = target_scaler.inverse_transform(y_pred_scaled)
y_test_rescaled = target_scaler.inverse_transform(y_test)

# 3. Define NSE (Nash-Sutcliffe Efficiency)
def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# 4. Calculate Metrics
nse = calculate_nse(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
bias = np.mean(y_pred_rescaled - y_test_rescaled)

# 5. Print Performance Report
print("\n" + "="*30)
print("  MODEL PERFORMANCE REPORT")
print("="*30)
print(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
print(f"R-Squared (R2) Score:           {r2:.4f}")
print(f"Mean Absolute Error (MAE):      {mae:.2f} mm")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mm")
print(f"Model Bias:                     {bias:.2f} mm")
print("="*30)

# 6. Save model + metrics for frontend consumption
models_dir = Path(__file__).resolve().parent / "saved_models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "lstm.keras"
model.save(model_path)

metrics_path = Path(__file__).resolve().parent / "model_metrics.json"
metrics_payload = {}
if metrics_path.exists():
    try:
        metrics_payload = json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        metrics_payload = {}

metrics_payload["lstm"] = {
    "nse": float(nse),
    "r2": float(r2),
    "mae": float(mae),
    "rmse": float(rmse),
    "bias": float(bias),
}
metrics_path.write_text(json.dumps(metrics_payload, indent=2))