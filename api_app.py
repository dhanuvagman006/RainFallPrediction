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

MODEL_CONFIGS = {
    "lstm": {"sequence_length": 1, "model_file": "lstm.keras"},
    "gru": {"sequence_length": 1, "model_file": "gru.keras"},
    "bilstm": {"sequence_length": 1, "model_file": "bilstm.keras"},
    "tcn": {"sequence_length": 1, "model_file": "tcn.keras"},
    "hybrid": {"sequence_length": 7, "model_file": "hybrid.keras"},
    "transformer": {"sequence_length": 14, "model_file": "transformer.keras"},
}


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Sequence of daily feature rows.")


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


def get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise KeyError(f"Unknown model: {model_name}")

    model_path = MODELS_DIR / config["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == "tcn":
        if TCN is None:
            raise ImportError("TCN dependency not available for loading the TCN model.")
        model = load_model(model_path, custom_objects={"TCN": TCN})
    else:
        model = load_model(model_path)

    _model_cache[model_name] = model
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


@app.get("/api/models")
def list_models():
    response = [
        {
            "name": name,
            "sequence_length": config["sequence_length"],
            "features": FEATURE_COLS,
        }
        for name, config in MODEL_CONFIGS.items()
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
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise HTTPException(status_code=404, detail="Unknown model")

    try:
        x_input = prepare_sequence(payload.sequence, config["sequence_length"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    model = get_model(model_name)
    preds_scaled = model.predict(x_input, verbose=0)
    if preds_scaled.ndim == 2:
        preds_scaled = preds_scaled.reshape(-1, 1)

    scalers = get_scalers()
    preds = scalers["y"].inverse_transform(preds_scaled)
    prediction = float(max(0.0, preds[0][0]))

    return {"model": model_name, "prediction": prediction}
