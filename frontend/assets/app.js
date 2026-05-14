const FEATURES = [
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
];

const MODEL_CONFIGS = {
  lstm: { sequenceLength: 1, title: "LSTM" },
  gru: { sequenceLength: 1, title: "GRU" },
  bilstm: { sequenceLength: 1, title: "Bi-LSTM" },
  tcn: { sequenceLength: 1, title: "TCN" },
  hybrid: { sequenceLength: 7, title: "Hybrid" },
  transformer: { sequenceLength: 14, title: "Transformer" },
};

function createInputRow(index, sequenceLength) {
  const row = document.createElement("div");
  row.className = "input-row";

  FEATURES.forEach((feature) => {
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.placeholder = feature;
    input.dataset.feature = feature;
    input.dataset.rowIndex = String(index);
    row.appendChild(input);
  });

  return row;
}

function renderInputGrid(container, sequenceLength) {
  container.innerHTML = "";
  for (let i = 0; i < sequenceLength; i += 1) {
    const row = createInputRow(i, sequenceLength);
    container.appendChild(row);
  }
}

function collectSequence(container, sequenceLength) {
  const rows = Array.from(container.querySelectorAll(".input-row"));
  if (rows.length !== sequenceLength) {
    throw new Error("Input rows mismatch.");
  }

  return rows.map((row) => {
    const inputs = Array.from(row.querySelectorAll("input"));
    return inputs.map((input) => {
      const value = parseFloat(input.value);
      if (Number.isNaN(value)) {
        throw new Error("Please fill in all fields with numeric values.");
      }
      return value;
    });
  });
}

async function loadMetrics(modelName, metricsContainer) {
  try {
    const response = await fetch("/api/metrics");
    const metrics = await response.json();
    const data = metrics[modelName];

    if (!data) {
      metricsContainer.innerHTML = "<p class=\"helper\">Metrics not available yet.</p>";
      return;
    }

    metricsContainer.innerHTML = "";
    const items = [
      ["NSE", data.nse],
      ["R2", data.r2],
      ["MAE", data.mae],
      ["RMSE", data.rmse],
      ["Bias", data.bias],
    ];

    items.forEach(([label, value]) => {
      const card = document.createElement("div");
      card.className = "metric";
      card.innerHTML = `<strong>${label}</strong><div>${Number(value).toFixed(4)}</div>`;
      metricsContainer.appendChild(card);
    });
  } catch (err) {
    metricsContainer.innerHTML = "<p class=\"helper\">Metrics unavailable.</p>";
  }
}

async function runPrediction(modelName, sequence, outputEl) {
  outputEl.textContent = "Running...";
  const response = await fetch(`/api/predict/${modelName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });

  if (!response.ok) {
    const errorPayload = await response.json();
    throw new Error(errorPayload.detail || "Prediction failed.");
  }

  const result = await response.json();
  outputEl.textContent = `${result.prediction.toFixed(3)} mm`;
  return result.prediction;
}

function setupPage() {
  const root = document.body;
  const modelName = root.dataset.model;
  if (!modelName) {
    return;
  }

  const config = MODEL_CONFIGS[modelName];
  const inputGrid = document.getElementById("input-grid");
  const submitBtn = document.getElementById("predict-btn");
  const outputEl = document.getElementById("prediction-output");
  const waterEl = document.getElementById("water-output");
  const roofInput = document.getElementById("roof-area");
  const errorEl = document.getElementById("prediction-error");
  const metricsContainer = document.getElementById("metrics");

  renderInputGrid(inputGrid, config.sequenceLength);
  loadMetrics(modelName, metricsContainer);

  submitBtn.addEventListener("click", async () => {
    errorEl.textContent = "";
    try {
      const sequence = collectSequence(inputGrid, config.sequenceLength);
      const predictionMm = await runPrediction(modelName, sequence, outputEl);
      const roofArea = parseFloat(roofInput.value);
      if (Number.isNaN(roofArea) || roofArea <= 0) {
        waterEl.textContent = "";
        errorEl.textContent = "Enter a valid roof area to estimate water collected.";
        return;
      }
      const liters = predictionMm * roofArea;
      waterEl.textContent = `${liters.toFixed(2)} liters`;
    } catch (err) {
      outputEl.textContent = "";
      waterEl.textContent = "";
      errorEl.textContent = err.message;
    }
  });
}

document.addEventListener("DOMContentLoaded", setupPage);
