import { useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useData } from "../context/DataContext";

const REQUIRED_COLS = [
  "Container_ID",
  "Declaration_Date",
  "Declaration_Time",
  "Trade_Regime",
  "Origin_Country",
  "Destination_Country",
  "Destination_Port",
  "HS_Code",
  "Importer_ID",
  "Exporter_ID",
  "Declared_Value",
  "Declared_Weight",
  "Measured_Weight",
  "Shipping_Line",
  "Dwell_Time_Hours",
];

const OPTIONAL_COLS = ["Clearance_Status", "is_risky"];

const normalizeHeader = (header) =>
  String(header || "")
    .replace(/^\uFEFF/, "")
    .replace(/\s*\([^)]*\)\s*/g, "")
    .replace(/-/g, "_")
    .replace(/\s+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();

const parseCsvRows = (text) => {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("CSV must contain a header and at least one data row.");
  }

  const rawHeaders = lines[0]
    .split(",")
    .map((h) => h.trim().replace(/^"|"$/g, ""));

  const requiredLookup = Object.fromEntries(
    REQUIRED_COLS.map((col) => [normalizeHeader(col), col])
  );

  const headers = rawHeaders.map((header) => {
    const normalized = normalizeHeader(header);
    return requiredLookup[normalized] || header.replace(/^\uFEFF/, "");
  });

  return lines.slice(1).map((line) => {
    const values = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx] ?? "";
    });
    return row;
  });
};

const toNumber = (value, fallback = 0) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const buildPreview = (rows) => {
  const origins = new Set(rows.map((r) => r.Origin_Country).filter(Boolean));
  const avgValue =
    rows.length > 0
      ? rows.reduce((acc, r) => acc + toNumber(r.Declared_Value), 0) / rows.length
      : 0;
  const avgDwell =
    rows.length > 0
      ? rows.reduce((acc, r) => acc + toNumber(r.Dwell_Time_Hours), 0) / rows.length
      : 0;

  return {
    total: rows.length,
    uniqueOrigins: origins.size,
    avgValue,
    avgDwell,
    sample: rows.slice(0, 3),
  };
};

export default function UploadPage() {
  const { predictFromFile, retrainModel, resetToSample, isPredicting, lastError, isTraining, trainingStatus } = useData();
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | parsing | ready | uploading | error
  const [errorMsg, setErrorMsg] = useState("");
  const [preview, setPreview] = useState(null);
  const [pendingRows, setPendingRows] = useState(null);
  const [pendingName, setPendingName] = useState("");
  const [pendingFile, setPendingFile] = useState(null);
  const [mode, setMode] = useState("predict"); // predict | retrain

  const processFile = useCallback((file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setErrorMsg("Only CSV files are supported.");
      setStatus("error");
      return;
    }

    setStatus("parsing");
    setErrorMsg("");
    setPreview(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const rows = parseCsvRows(String(event.target?.result || ""));
        const firstRow = rows[0] || {};
        const missing = REQUIRED_COLS.filter((col) => !(col in firstRow));
        if (missing.length > 0) {
          throw new Error(
            `Missing required columns: ${missing.join(", ")}`
          );
        }

        setPendingRows(rows);
        setPendingName(file.name);
        setPendingFile(file);
        setPreview(buildPreview(rows));
        setStatus("ready");
      } catch (error) {
        setErrorMsg(error.message || "Failed to parse CSV.");
        setStatus("error");
      }
    };

    reader.onerror = () => {
      setErrorMsg("Failed to read the selected file.");
      setStatus("error");
    };

    reader.readAsText(file);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragOver(false);
      processFile(event.dataTransfer.files[0]);
    },
    [processFile]
  );

  const handleFileInput = useCallback(
    (event) => {
      processFile(event.target.files[0]);
      event.target.value = "";
    },
    [processFile]
  );

  const handleConfirm = async () => {
    if (!pendingFile || !pendingRows) return;

    setStatus("uploading");
    setErrorMsg("");
    try {
      if (mode === "predict") {
        await predictFromFile(pendingFile, pendingRows);
        navigate("/dashboard");
      } else if (mode === "retrain") {
        await retrainModel(pendingFile);
        setStatus("completed");
        setErrorMsg("Model retrained successfully!");
      }
    } catch (error) {
      setErrorMsg(error.message || (mode === "predict" ? "Prediction failed." : "Retraining failed."));
      setStatus("error");
    }
  };

  const handleUseSample = () => {
    resetToSample();
    navigate("/dashboard");
  };

  const LEVEL_COLORS = {
    Critical: "#ef4444",
    High: "#f97316",
    Medium: "#eab308",
    Low: "#22c55e",
  };

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 flex flex-col">
      <div className="border-b border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        <div className="max-w-3xl mx-auto px-6 pt-12 pb-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-900/40 border border-blue-700/50 text-blue-300 text-xs font-semibold mb-6 tracking-widest uppercase">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            AI-Powered Risk Assessment
          </div>
          <h1 className="text-4xl sm:text-5xl font-black text-white tracking-tight mb-4 leading-none">
            Upload Shipment Data
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
            Upload raw shipment CSV, run the model, and see predicted risk
            levels instantly.
          </p>
        </div>
      </div>

      <div className="max-w-3xl mx-auto px-6 py-10 w-full space-y-6">
        {/* Mode Toggle */}
        <div className="flex gap-2 rounded-lg bg-slate-900 p-1 border border-slate-800">
          <button
            onClick={() => { setMode("predict"); setStatus("idle"); setErrorMsg(""); }}
            className={`flex-1 px-4 py-2 rounded-md font-semibold text-sm transition-all ${
              mode === "predict"
                ? "bg-blue-600 text-white"
                : "text-slate-400 hover:text-slate-300"
            }`}
          >
            📊 Run Prediction
          </button>
          <button
            onClick={() => { setMode("retrain"); setStatus("idle"); setErrorMsg(""); }}
            className={`flex-1 px-4 py-2 rounded-md font-semibold text-sm transition-all ${
              mode === "retrain"
                ? "bg-purple-600 text-white"
                : "text-slate-400 hover:text-slate-300"
            }`}
          >
            🔄 Retrain Model
          </button>
        </div>

        {/* Training Status Progress */}
        {isTraining && trainingStatus && (
          <div className="rounded-xl bg-purple-950/50 border border-purple-800 p-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-purple-300 font-semibold text-sm">Model Training in Progress</p>
              <span className="text-purple-400 text-xs font-mono">{trainingStatus.progress || 0}%</span>
            </div>
            <div className="w-full bg-purple-900/30 rounded-full h-2 border border-purple-700/50 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-purple-400 transition-all duration-300"
                style={{ width: `${trainingStatus.progress || 0}%` }}
              />
            </div>
            <p className="text-purple-400/80 text-xs mt-2">{trainingStatus.message}</p>
            {trainingStatus.rows_valid && (
              <p className="text-purple-400/60 text-xs mt-1">Valid rows: {trainingStatus.rows_valid} / {trainingStatus.rows_loaded}</p>
            )}
          </div>
        )}

        {status === "completed" && mode === "retrain" && (
          <div className="rounded-xl bg-green-950/50 border border-green-800 p-4">
            <p className="text-green-300 font-semibold text-sm">✓ Model Retrained Successfully!</p>
            <p className="text-green-400/80 text-xs mt-1">The model has been updated with your new training data.</p>
          </div>
        )}
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-200 p-12 text-center group ${
            dragOver
              ? "border-blue-500 bg-blue-950/30 scale-[1.01]"
              : status === "ready"
                ? "border-green-600 bg-green-950/20"
                : status === "error"
                  ? "border-red-700 bg-red-950/20"
                  : "border-slate-700 bg-slate-900 hover:border-slate-500 hover:bg-slate-900/80"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleFileInput}
          />

          {status === "parsing" || status === "uploading" || isPredicting ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
              <p className="text-slate-300 font-semibold">
                {status === "uploading" || isPredicting
                  ? "Running model prediction..."
                  : "Parsing CSV..."}
              </p>
            </div>
          ) : status === "ready" && preview ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-full bg-green-900/50 border border-green-700 flex items-center justify-center text-2xl">
                OK
              </div>
              <p className="text-green-400 font-bold text-lg">{pendingName}</p>
              <p className="text-slate-400 text-sm">
                File validated - {preview.total} rows ready for prediction
              </p>
              <p className="text-slate-500 text-xs">Click to replace file</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4">
              <div
                className={`w-16 h-16 rounded-2xl border flex items-center justify-center text-3xl transition-colors ${
                  dragOver
                    ? "border-blue-500 bg-blue-900/40 text-blue-300"
                    : "border-slate-700 bg-slate-800 text-slate-400 group-hover:border-slate-500 group-hover:text-slate-300"
                }`}
              >
                ^
              </div>
              <div>
                <p className="text-white font-bold text-lg">
                  {dragOver ? "Drop CSV here" : "Drag and drop your CSV"}
                </p>
                <p className="text-slate-500 text-sm mt-1">
                  or{" "}
                  <span className="text-blue-400 underline underline-offset-2">
                    browse files
                  </span>
                </p>
              </div>
              <p className="text-slate-600 text-xs">
                Accepts raw shipment CSV with required columns
              </p>
            </div>
          )}
        </div>

        {(status === "error" || errorMsg || lastError) && (
          <div className="rounded-xl bg-red-950/50 border border-red-800 p-4">
            <p className="text-red-300 font-semibold text-sm mb-1">
              Upload or prediction failed
            </p>
            <p className="text-red-400/80 text-sm">{errorMsg || lastError}</p>
          </div>
        )}

        {status === "ready" && preview && (
          <div className="rounded-2xl border border-slate-700 bg-slate-900 overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-800 flex flex-wrap items-center gap-4 justify-between">
              <p className="text-slate-200 font-semibold text-sm">
                Preview - {preview.total} rows
              </p>
              <div className="flex gap-3 text-xs text-slate-400">
                <span>Origins: {preview.uniqueOrigins}</span>
                <span>Avg Value: ${preview.avgValue.toFixed(2)}</span>
                <span>Avg Dwell: {preview.avgDwell.toFixed(2)}h</span>
              </div>
            </div>
            <div className="divide-y divide-slate-800">
              {preview.sample.map((row, idx) => (
                <div
                  key={`${row.Container_ID || "row"}-${idx}`}
                  className="px-5 py-3 flex items-center gap-4 text-sm"
                >
                  <span className="font-mono text-slate-300 font-semibold text-xs w-28 flex-shrink-0">
                    {row.Container_ID || `ROW_${idx + 1}`}
                  </span>
                  <span className="text-xs text-slate-400">
                    {row.Origin_Country || "N/A"} to{" "}
                    {row.Destination_Country || "N/A"}
                  </span>
                  <span className="text-xs text-slate-500 truncate">
                    HS: {row.HS_Code || "N/A"} - Dwell:{" "}
                    {toNumber(row.Dwell_Time_Hours, 0).toFixed(1)}h
                  </span>
                </div>
              ))}
              {preview.total > 3 && (
                <div className="px-5 py-2 text-xs text-slate-600 text-center">
                  + {preview.total - 3} more rows
                </div>
              )}
            </div>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-3">
          {status === "ready" && pendingRows && (
            <button
              onClick={handleConfirm}
              disabled={isPredicting || isTraining}
              className="flex-1 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold text-sm transition-all shadow-lg shadow-blue-900/40"
            >
              {isPredicting || isTraining
                ? mode === "retrain"
                  ? "Retraining Model..."
                  : "Running Prediction..."
                : mode === "retrain"
                ? `Retrain with ${pendingRows.length} rows`
                : `Run Model Prediction (${pendingRows.length} rows)`}
            </button>
          )}
          <button
            onClick={handleUseSample}
            className={`py-3 rounded-xl font-semibold text-sm transition-all ${
              status === "ready"
                ? "px-5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                : "flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
            }`}
          >
            Use Sample Data
          </button>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <p className="text-slate-300 font-semibold text-sm mb-3">
            Expected CSV Format
          </p>
          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-2">
                Required columns
              </p>
              <div className="flex flex-wrap gap-1.5">
                {REQUIRED_COLS.map((col) => (
                  <span
                    key={col}
                    className="px-2 py-0.5 rounded bg-blue-900/40 border border-blue-700/50 text-blue-300 text-xs font-mono"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider font-semibold mb-2">
                Optional columns
              </p>
              <div className="flex flex-wrap gap-1.5">
                {OPTIONAL_COLS.map((col) => (
                  <span
                    key={col}
                    className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400 text-xs font-mono"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
