#!/usr/bin/env python3
"""Generate evaluation plots for all rainfall models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from scipy import stats

try:
    from tcn import TCN
except Exception:  # pragma: no cover - optional dependency for loading TCN
    TCN = None


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


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.ffill(inplace=True)
    return df


def create_sequences(data_x: np.ndarray, data_y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x_seq, y_seq = [], []
    for i in range(len(data_x) - seq_len):
        x_seq.append(data_x[i : i + seq_len])
        y_seq.append(data_y[i + seq_len])
    return np.array(x_seq), np.array(y_seq)


def prepare_test_data(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    x = df[FEATURE_COLS].values
    y = df[[TARGET_COL]].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    if sequence_length == 1:
        x_seq = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        y_seq = y_scaled
    else:
        x_seq, y_seq = create_sequences(x_scaled, y_scaled, sequence_length)

    split = int(len(x_seq) * 0.8)
    x_test = x_seq[split:]
    y_test = y_seq[split:]

    y_test_inv = scaler_y.inverse_transform(y_test)
    return x_test, y_test_inv, scaler_y


def load_trained_model(models_dir: Path, model_name: str, seq_len: int):
    model_file = f"{model_name}_{seq_len}d.keras"
    model_path = models_dir / model_file
    if not model_path.exists():
        print(f"Skipping missing model: {model_path}")
        return None
    if model_name == "tcn" and TCN is not None:
        return load_model(model_path, custom_objects={"TCN": TCN})
    return load_model(model_path)


def plot_training_loss(history: dict, output_dir: Path) -> None:
    if not history or "loss" not in history:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()


def plot_pred_vs_actual_line(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", linewidth=1.5)
    plt.plot(y_pred, label="Predicted", linewidth=1.5)
    plt.title("Predicted vs Actual (Line)")
    plt.xlabel("Test Sample")
    plt.ylabel("Rainfall (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_actual_line.png", dpi=160)
    plt.close()


def plot_pred_vs_actual_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.title("Predicted vs Actual (Scatter)")
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Predicted Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_actual_scatter.png", dpi=160)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    residuals = y_pred - y_true

    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, residuals, alpha=0.5, s=12)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title("Residuals vs Actual")
    plt.xlabel("Actual Rainfall (mm)")
    plt.ylabel("Residual (Predicted - Actual)")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=40, alpha=0.8)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    stats.probplot(residuals.flatten(), dist="norm", plot=plt)
    plt.title("Residual Q-Q Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_qq.png", dpi=160)
    plt.close()


def plot_cumulative_rainfall(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    actual_cum = np.cumsum(y_true)
    pred_cum = np.cumsum(y_pred)

    plt.figure(figsize=(10, 5))
    plt.plot(actual_cum, label="Actual Cumulative", linewidth=1.5)
    plt.plot(pred_cum, label="Predicted Cumulative", linewidth=1.5)
    plt.title("Cumulative Rainfall")
    plt.xlabel("Test Sample")
    plt.ylabel("Cumulative Rainfall (mm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_rainfall.png", dpi=160)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, output_dir: Path) -> None:
    actual_binary = (y_true >= threshold).astype(int)
    pred_binary = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(actual_binary, pred_binary)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Rain/No-Rain Confusion Matrix (>= {threshold} mm)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        plt.text(j, i, str(value), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["No-Rain", "Rain"])
    plt.yticks([0, 1], ["No-Rain", "Rain"])
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def plot_model_comparison(metrics: dict, output_dir: Path, seq_len: int) -> None:
    if not metrics:
        return

    model_names = [name for name in MODEL_NAMES if f"{name}_{seq_len}d" in metrics]
    if not model_names:
        return

    rmse_vals = [metrics[f"{name}_{seq_len}d"]["rmse"] for name in model_names]
    mae_vals = [metrics[f"{name}_{seq_len}d"]["mae"] for name in model_names]
    r2_vals = [metrics[f"{name}_{seq_len}d"]["r2"] for name in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(model_names, rmse_vals)
    axes[0].set_title(f"RMSE Comparison ({seq_len}d)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].bar(model_names, mae_vals)
    axes[1].set_title(f"MAE Comparison ({seq_len}d)")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=35)

    axes[2].bar(model_names, r2_vals)
    axes[2].set_title(f"R2 Comparison ({seq_len}d)")
    axes[2].set_ylabel("R2")
    axes[2].tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(output_dir / f"model_comparison_{seq_len}d.png", dpi=160)
    plt.close()


def plot_taylor_diagram(stats_map: dict[str, dict[str, float]], output_dir: Path, seq_len: int) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, polar=True)

    max_std = max(v["std"] for v in stats_map.values())
    ax.set_ylim(0, max_std * 1.1)

    for name, values in stats_map.items():
        corr = values["corr"]
        std = values["std"]
        angle = np.arccos(np.clip(corr, -1, 1))
        ax.plot(angle, std, "o", label=name)

    ax.set_title("Taylor Diagram")
    ax.set_thetalim(0, np.pi / 2)
    ax.set_thetagrids([0, 30, 60, 90], labels=["1.0", "0.87", "0.5", "0.0"])
    ax.set_rlabel_position(135)
    ax.set_ylabel("Standard Deviation")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / f"taylor_diagram_{seq_len}d.png", dpi=160)
    plt.close()


def load_history(history_dir: Path, model_key: str) -> dict | None:
    history_path = history_dir / f"{model_key}.json"
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text())
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate evaluation plots for rainfall models.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Rain/no-rain threshold in mm.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "dakshina_kannada_rainfall_daily_2000_2024.csv"
    models_dir = base_dir / "saved_models"
    history_dir = base_dir / "training_history"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)

    metrics_path = base_dir / "model_metrics.json"
    metrics_payload = {}
    if metrics_path.exists():
        try:
            metrics_payload = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics_payload = {}

    for seq_len in SEQUENCE_LENGTHS:
        stats_map: dict[str, dict[str, float]] = {}

        for model_name in MODEL_NAMES:
            model_key = f"{model_name}_{seq_len}d"
            model_output_dir = plots_dir / model_name / f"{seq_len}d"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            history = load_history(history_dir, model_key)
            if history:
                plot_training_loss(history, model_output_dir)

            x_test, y_true, scaler_y = prepare_test_data(df, seq_len)
            model = load_trained_model(models_dir, model_name, seq_len)
            if model is None:
                continue
            y_pred_scaled = model.predict(x_test, verbose=0)

            if y_pred_scaled.ndim == 2:
                y_pred = y_pred_scaled
            else:
                y_pred = y_pred_scaled.reshape(-1, 1)

            y_pred = scaler_y.inverse_transform(y_pred).astype(float)

            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()

            plot_pred_vs_actual_line(y_true_flat, y_pred_flat, model_output_dir)
            plot_pred_vs_actual_scatter(y_true_flat, y_pred_flat, model_output_dir)
            plot_residuals(y_true_flat, y_pred_flat, model_output_dir)
            plot_cumulative_rainfall(y_true_flat, y_pred_flat, model_output_dir)
            plot_confusion_matrix(y_true_flat, y_pred_flat, args.threshold, model_output_dir)

            corr = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
            stats_map[model_name] = {
                "corr": float(corr),
                "std": float(np.std(y_pred_flat)),
            }

        comparison_dir = plots_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        plot_model_comparison(metrics_payload, comparison_dir, seq_len)

        if stats_map:
            plot_taylor_diagram(stats_map, comparison_dir, seq_len)

    print(f"Plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
