#!/usr/bin/env python3
"""FastAPI app serving model predictions and frontend pages."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

try:
    from tcn import TCN
except Exception:  # pragma: no cover
    TCN = None

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "TrainModels"
MODELS_DIR = TRAIN_DIR / "saved_models"
PLOTS_DIR = TRAIN_DIR / "plots"
METRICS_PATH = TRAIN_DIR / "model_metrics.json"
DATA_PATH = TRAIN_DIR / "dakshina_kannada_rainfall_daily_2000_2024.csv"
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_MODELS_DIR = FRONTEND_DIR / "models"

FEATURE_COLS = [
    "year",
    "month",
    "day",
    "ps",
    "t2m",
    "t2m_max",
    "t2m_min",
    "rh2m",
    "ws2m",
    "wd2m",
    "allsky_sfc_sw_dwn",
]
TARGET_COL = "prectotcorr"

MODEL_NAMES = ["lstm", "gru", "bilstm", "tcn", "hybrid", "transformer"]
SEQUENCE_LENGTHS = [1, 7, 14]


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Sequence of daily feature rows.")


class IrrigationRequest(BaseModel):
    tank_liters: float = Field(..., ge=0)
    soil_moisture_percent: float = Field(..., ge=0, le=100)
    arecanut_palms: float = Field(..., ge=0)
    coconut_palms: float = Field(..., ge=0)
    paddy_area_m2: float = Field(..., ge=0)
    rainfall_forecast_mm: List[float] | None = None
    model_name: str | None = None
    sequence: List[List[float]] | None = None


app = FastAPI(title="Rainfall Prediction API")

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

_model_cache: Dict[str, object] = {}
_scaler_cache: Dict[str, MinMaxScaler] | None = None


def get_scalers() -> Dict[str, MinMaxScaler]:
    global _scaler_cache
    if _scaler_cache is not None:
        return _scaler_cache

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.ffill(inplace=True)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_x.fit(df[FEATURE_COLS])
    scaler_y.fit(df[[TARGET_COL]])

    _scaler_cache = {"x": scaler_x, "y": scaler_y}
    return _scaler_cache


def get_model(model_name: str, seq_len: int):
    cache_key = f"{model_name}_{seq_len}d"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if model_name not in MODEL_NAMES:
        raise KeyError(f"Unknown model: {model_name}")
    if seq_len not in SEQUENCE_LENGTHS:
        raise KeyError(f"Unsupported sequence length: {seq_len}")

    model_path = MODELS_DIR / f"{model_name}_{seq_len}d.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == "tcn":
        if TCN is None:
            raise ImportError("TCN dependency not available for loading the TCN model.")
        model = load_model(model_path, custom_objects={"TCN": TCN})
    else:
        model = load_model(model_path)

    _model_cache[cache_key] = model
    return model


def prepare_sequence(sequence: List[List[float]], seq_len: int) -> np.ndarray:
    if len(sequence) != seq_len:
        raise ValueError(f"Expected {seq_len} rows, got {len(sequence)}.")

    for row in sequence:
        if len(row) != len(FEATURE_COLS):
            raise ValueError(f"Each row must have {len(FEATURE_COLS)} features.")

    data = np.array(sequence, dtype=float)
    scalers = get_scalers()
    data_scaled = scalers["x"].transform(data)

    return data_scaled.reshape(1, seq_len, len(FEATURE_COLS))


@app.get("/")
def index_page():
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(index_path)


@app.get("/models/{model_name}")
def model_page(model_name: str):
    page_path = FRONTEND_MODELS_DIR / f"{model_name}.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Model page not found")
    return FileResponse(page_path)


@app.get("/irrigation")
def irrigation_page():
    page_path = FRONTEND_DIR / "irrigation.html"
    return FileResponse(page_path)


@app.get("/api/models")
def list_models():
    response = [
        {
            "name": name,
            "sequence_lengths": SEQUENCE_LENGTHS,
            "features": FEATURE_COLS,
        }
        for name in MODEL_NAMES
    ]
    return JSONResponse(response)


@app.get("/api/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        return JSONResponse({})
    try:
        return JSONResponse(json.loads(METRICS_PATH.read_text()))
    except json.JSONDecodeError:
        return JSONResponse({})


@app.post("/api/predict/{model_name}")
def predict(model_name: str, payload: PredictRequest):
    if model_name not in MODEL_NAMES:
        raise HTTPException(status_code=404, detail="Unknown model")

    seq_len = len(payload.sequence)
    if seq_len not in SEQUENCE_LENGTHS:
        raise HTTPException(status_code=400, detail="Unsupported sequence length")

    try:
        x_input = prepare_sequence(payload.sequence, seq_len)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = get_model(model_name, seq_len)
    preds_scaled = model.predict(x_input, verbose=0)
    if preds_scaled.ndim == 2:
        preds_scaled = preds_scaled.reshape(-1, 1)

    scalers = get_scalers()
    preds = scalers["y"].inverse_transform(preds_scaled)
    prediction = float(max(0.0, preds[0][0]))

    return {"model": model_name, "prediction": prediction}


def moisture_factor(soil_moisture: float) -> float:
    if soil_moisture >= 80:
        return 0.4
    if soil_moisture >= 60:
        return 0.6
    if soil_moisture >= 40:
        return 0.8
    return 1.0


@app.post("/api/irrigation-plan")
def irrigation_plan(payload: IrrigationRequest):
    factor = moisture_factor(payload.soil_moisture_percent)

    arecanut_daily = (175.0 / 7.0) * payload.arecanut_palms
    coconut_daily = 225.0 * payload.coconut_palms
    paddy_mm_per_day = 1175.0 / 120.0
    paddy_daily = paddy_mm_per_day * payload.paddy_area_m2

    rainfall_forecast = payload.rainfall_forecast_mm
    if rainfall_forecast is not None:
        if len(rainfall_forecast) != 14:
            raise HTTPException(status_code=400, detail="Rainfall forecast must have 14 values.")

    if rainfall_forecast is None and payload.model_name and payload.sequence:
        seq_len = len(payload.sequence)
        if seq_len not in SEQUENCE_LENGTHS:
            raise HTTPException(status_code=400, detail="Unsupported sequence length for ML prediction.")
        try:
            x_input = prepare_sequence(payload.sequence, seq_len)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        model = get_model(payload.model_name, seq_len)
        preds_scaled = model.predict(x_input, verbose=0)
        if preds_scaled.ndim == 2:
            preds_scaled = preds_scaled.reshape(-1, 1)
        scalers = get_scalers()
        preds = scalers["y"].inverse_transform(preds_scaled)
        prediction = float(max(0.0, preds[0][0]))
        rainfall_forecast = [prediction for _ in range(14)]

    if rainfall_forecast is None:
        rainfall_forecast = [0.0 for _ in range(14)]

    base_daily = {
        "Arecanut": arecanut_daily,
        "Coconut": coconut_daily,
        "Paddy": paddy_daily,
    }

    rng = np.random.default_rng(42)
    daily_totals = []
    for day_index in range(14):
        rain_mm = rainfall_forecast[day_index]
        paddy_after_rain = max(0.0, (paddy_daily * factor) - (rain_mm * payload.paddy_area_m2))
        arecanut_after = arecanut_daily * factor
        coconut_after = coconut_daily * factor

        arecanut_after *= 1.0 - rng.uniform(0.0, 0.15)
        coconut_after *= 1.0 - rng.uniform(0.0, 0.15)
        total = arecanut_after + coconut_after + paddy_after_rain
        daily_totals.append((arecanut_after, coconut_after, paddy_after_rain, total))

    total_needed = sum(day[3] for day in daily_totals)
    if total_needed <= 0:
        raise HTTPException(status_code=400, detail="Total irrigation demand is 0.")

    ration_factor = min(1.0, payload.tank_liters / total_needed)

    schedule = []
    tank_left = payload.tank_liters
    for day in range(1, 15):
        arecanut_l, coconut_l, paddy_l, total = daily_totals[day - 1]
        arecanut_l *= ration_factor
        coconut_l *= ration_factor
        paddy_l *= ration_factor
        total = arecanut_l + coconut_l + paddy_l

        tank_left = max(0.0, tank_left - total)
        schedule.append(
            {
                "day": day,
                "crops": {
                    "Arecanut": arecanut_l,
                    "Coconut": coconut_l,
                    "Paddy": paddy_l,
                },
                "total_liters": total,
                "tank_left_liters": tank_left,
                "rain_mm": rainfall_forecast[day - 1],
            }
        )

    summary = {
        "soil_moisture_factor": factor,
        "ration_factor": ration_factor,
        "tank_start_liters": payload.tank_liters,
        "tank_end_liters": tank_left,
        "fixed_daily_liters": base_daily,
        "rainfall_source": "forecast" if payload.rainfall_forecast_mm else ("ml" if payload.model_name else "none"),
    }

    return {"summary": summary, "schedule": schedule}
