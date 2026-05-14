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

const SEQUENCE_OPTIONS = [1, 7, 14];

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

async function loadMetrics(modelName, seqLen, metricsContainer) {
  try {
    const response = await fetch("/api/metrics");
    const metrics = await response.json();
    const data = metrics[`${modelName}_${seqLen}d`];

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

function updatePlots(modelName, seqLen) {
  const plotImages = document.querySelectorAll("[data-plot]");
  plotImages.forEach((img) => {
    const filename = img.dataset.plot;
    img.src = `/plots/${modelName}/${seqLen}d/${filename}`;
  });
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

  const inputGrid = document.getElementById("input-grid");
  const submitBtn = document.getElementById("predict-btn");
  const outputEl = document.getElementById("prediction-output");
  const waterEl = document.getElementById("water-output");
  const roofInput = document.getElementById("roof-area");
  const errorEl = document.getElementById("prediction-error");
  const metricsContainer = document.getElementById("metrics");
  const horizonSelect = document.getElementById("horizon-select");
  const sequenceBadge = document.getElementById("sequence-badge");

  const defaultSeq = SEQUENCE_OPTIONS[0];
  horizonSelect.value = String(defaultSeq);

  const updateForSeq = (seqLen) => {
    renderInputGrid(inputGrid, seqLen);
    loadMetrics(modelName, seqLen, metricsContainer);
    updatePlots(modelName, seqLen);
    sequenceBadge.textContent = `Sequence length: ${seqLen} day${seqLen === 1 ? "" : "s"}`;
    outputEl.textContent = "";
    waterEl.textContent = "";
    errorEl.textContent = "";
  };

  updateForSeq(defaultSeq);

  horizonSelect.addEventListener("change", () => {
    const seqLen = parseInt(horizonSelect.value, 10);
    updateForSeq(seqLen);
  });

  submitBtn.addEventListener("click", async () => {
    errorEl.textContent = "";
    try {
      const seqLen = parseInt(horizonSelect.value, 10);
      const sequence = collectSequence(inputGrid, seqLen);
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

function setupIrrigation() {
  const root = document.body;
  if (root.dataset.page !== "irrigation") {
    return;
  }

  const tankInput = document.getElementById("tank-liters");
  const moistureInput = document.getElementById("soil-moisture");
  const arecanutInput = document.getElementById("arecanut-palms");
  const coconutInput = document.getElementById("coconut-palms");
  const paddyInput = document.getElementById("paddy-area");
  const rainInput = document.getElementById("rain-forecast");
  const useMl = document.getElementById("use-ml");
  const mlSection = document.getElementById("ml-section");
  const mlModel = document.getElementById("ml-model");
  const mlHorizon = document.getElementById("ml-horizon");
  const mlInputGrid = document.getElementById("ml-input-grid");
  const planBtn = document.getElementById("plan-btn");
  const planBody = document.getElementById("plan-body");
  const planSummary = document.getElementById("plan-summary");
  const planError = document.getElementById("plan-error");

  const updateMlGrid = () => {
    const seqLen = parseInt(mlHorizon.value, 10);
    renderInputGrid(mlInputGrid, seqLen);
  };

  useMl.addEventListener("change", () => {
    mlSection.style.display = useMl.checked ? "block" : "none";
    if (useMl.checked) {
      updateMlGrid();
    }
  });

  mlHorizon.addEventListener("change", updateMlGrid);

  planBtn.addEventListener("click", async () => {
    planError.textContent = "";
    planBody.innerHTML = "";
    planSummary.textContent = "";

    const tankLiters = parseFloat(tankInput.value);
    const soilMoisture = parseFloat(moistureInput.value);
    const arecanutPalms = parseFloat(arecanutInput.value);
    const coconutPalms = parseFloat(coconutInput.value);
    const paddyArea = parseFloat(paddyInput.value);

    if (Number.isNaN(tankLiters) || tankLiters < 0) {
      planError.textContent = "Enter a valid tank water value.";
      return;
    }
    if (Number.isNaN(soilMoisture) || soilMoisture < 0 || soilMoisture > 100) {
      planError.textContent = "Soil moisture must be between 0 and 100%.";
      return;
    }
    if (Number.isNaN(arecanutPalms) || arecanutPalms < 0) {
      planError.textContent = "Enter a valid arecanut palms count.";
      return;
    }
    if (Number.isNaN(coconutPalms) || coconutPalms < 0) {
      planError.textContent = "Enter a valid coconut palms count.";
      return;
    }
    if (Number.isNaN(paddyArea) || paddyArea < 0) {
      planError.textContent = "Enter a valid paddy area.";
      return;
    }

    let rainfallForecast = null;
    if (rainInput.value.trim().length > 0) {
      const parts = rainInput.value.split(",").map((value) => parseFloat(value.trim()));
      if (parts.length !== 14 || parts.some((value) => Number.isNaN(value))) {
        planError.textContent = "Rainfall forecast must have 14 numeric values.";
        return;
      }
      rainfallForecast = parts;
    }

    let mlPayload = {};
    if (useMl.checked) {
      const seqLen = parseInt(mlHorizon.value, 10);
      try {
        const sequence = collectSequence(mlInputGrid, seqLen);
        mlPayload = { model_name: mlModel.value, sequence };
      } catch (err) {
        planError.textContent = err.message;
        return;
      }
    }
    const payload = {
      tank_liters: tankLiters,
      soil_moisture_percent: soilMoisture,
      arecanut_palms: arecanutPalms,
      coconut_palms: coconutPalms,
      paddy_area_m2: paddyArea,
      rainfall_forecast_mm: rainfallForecast,
      ...mlPayload,
    };

    try {
      const response = await fetch("/api/irrigation-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errPayload = await response.json();
        throw new Error(errPayload.detail || "Failed to generate plan.");
      }

      const result = await response.json();
      const summary = result.summary;
      planSummary.innerHTML = `
        <div><strong>Soil moisture factor:</strong> ${summary.soil_moisture_factor.toFixed(2)}</div>
        <div><strong>Ration factor:</strong> ${summary.ration_factor.toFixed(2)}</div>
        <div><strong>Tank start (L):</strong> ${summary.tank_start_liters.toFixed(2)}</div>
        <div><strong>Tank end (L):</strong> ${summary.tank_end_liters.toFixed(2)}</div>
        <div><strong>Rainfall source:</strong> ${summary.rainfall_source}</div>
      `;

      result.schedule.forEach((day) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${day.day}</td>
          <td>${day.crops.Arecanut.toFixed(2)}</td>
          <td>${day.crops.Coconut.toFixed(2)}</td>
          <td>${day.crops.Paddy.toFixed(2)}</td>
          <td>${day.total_liters.toFixed(2)}</td>
          <td>${day.tank_left_liters.toFixed(2)}</td>
        `;
        planBody.appendChild(row);
      });
    } catch (err) {
      planError.textContent = err.message;
    }
  });
}

document.addEventListener("DOMContentLoaded", setupIrrigation);
