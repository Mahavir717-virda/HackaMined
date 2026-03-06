import { createContext, useContext, useState } from "react";
import ContainerRiskAPI from "../services/containerRiskAPI";

// ── Sample data embedded so dashboard works before any upload ──────────────
export const SAMPLE_DATA = [
  {"id":"CONT749837","score":100.0,"level":"Critical","origin":"SD","value":1893178,"dw":2210.6,"mw":5174.4,"dwell":0.7,"hs":"9013","line":"PIL","expl":"Risk indicators: measured weight is 134% higher than declared; unusually high value-per-kg ($856/kg); medium-risk origin country.","date":"2024-03-12"},
  {"id":"CONT443912","score":99.5,"level":"Critical","origin":"LY","value":1517284,"dw":3505.3,"mw":12668.3,"dwell":0.8,"hs":"7108","line":"HMM","expl":"Risk indicators: measured weight is 261% higher than declared; unusually high value-per-kg ($433/kg); high-risk origin country.","date":"2024-03-11"},
  {"id":"CONT256570","score":99.0,"level":"Critical","origin":"KP","value":1474064,"dw":10501.7,"mw":1943.5,"dwell":1.1,"hs":"9013","line":"PIL","expl":"Risk indicators: measured weight is 82% lower than declared; unusually high value-per-kg ($140/kg); high-risk origin country.","date":"2024-03-10"},
  {"id":"CONT594829","score":98.5,"level":"Critical","origin":"IR","value":884064,"dw":5427.7,"mw":520.3,"dwell":0.5,"hs":"7108","line":"MSC","expl":"Risk indicators: measured weight is 90% lower than declared; unusually high value-per-kg ($163/kg); high-risk origin country.","date":"2024-03-09"},
  {"id":"CONT912744","score":98.0,"level":"Critical","origin":"SY","value":1262983,"dw":2079.9,"mw":7064.2,"dwell":1.3,"hs":"8703","line":"ONE","expl":"Risk indicators: measured weight is 240% higher than declared; unusually high value-per-kg ($607/kg); high-risk origin country.","date":"2024-03-08"},
  {"id":"CONT532941","score":97.5,"level":"Critical","origin":"PK","value":1920488,"dw":3958.8,"mw":9794.7,"dwell":0.7,"hs":"8542","line":"COSCO","expl":"Risk indicators: measured weight is 147% higher than declared; unusually high value-per-kg ($485/kg); medium-risk origin country.","date":"2024-03-07"},
  {"id":"CONT895326","score":97.0,"level":"Critical","origin":"MM","value":1098238,"dw":8254.9,"mw":5096.9,"dwell":0.9,"hs":"8517","line":"YML","expl":"Risk indicators: measured weight is 38% lower than declared; unusually high value-per-kg ($133/kg); high-risk origin country.","date":"2024-03-06"},
  {"id":"CONT922827","score":96.5,"level":"Critical","origin":"KP","value":877499,"dw":6862.5,"mw":347.9,"dwell":0.4,"hs":"2710","line":"HAPAG","expl":"Risk indicators: measured weight is 95% lower than declared; unusually high value-per-kg ($128/kg); high-risk origin country.","date":"2024-03-05"},
  {"id":"CONT461447","score":96.0,"level":"Critical","origin":"NG","value":614438,"dw":11823.2,"mw":14843.7,"dwell":140.2,"hs":"8471","line":"MAERSK","expl":"Risk indicators: measured weight is 26% higher than declared; medium-risk origin country; excessive dwell time (140 hrs).","date":"2024-03-04"},
  {"id":"CONT317314","score":95.5,"level":"Critical","origin":"SD","value":428191,"dw":9543.1,"mw":188.5,"dwell":74.3,"hs":"8542","line":"CMA CGM","expl":"Risk indicators: measured weight is 98% lower than declared; medium-risk origin country; excessive dwell time (74 hrs).","date":"2024-03-03"},
  {"id":"CONT771023","score":95.0,"level":"Critical","origin":"AF","value":521834,"dw":4823.7,"mw":12089.5,"dwell":0.3,"hs":"6110","line":"EVERGREEN","expl":"Risk indicators: measured weight is 151% higher than declared; medium-risk origin country; unusually short dwell time.","date":"2024-03-02"},
  {"id":"CONT384612","score":94.5,"level":"Critical","origin":"IR","value":1342765,"dw":7234.8,"mw":1823.2,"dwell":88.5,"hs":"9013","line":"MSC","expl":"Risk indicators: measured weight is 75% lower than declared; high-risk origin country; excessive dwell time (89 hrs).","date":"2024-03-01"},
  {"id":"CONT628451","score":94.0,"level":"Critical","origin":"YE","value":987432,"dw":3421.6,"mw":8934.2,"dwell":1.2,"hs":"2933","line":"PIL","expl":"Risk indicators: measured weight is 161% higher than declared; unusually high value-per-kg ($289/kg); high-risk origin country.","date":"2024-02-28"},
  {"id":"CONT195837","score":93.5,"level":"Critical","origin":"LY","value":2145632,"dw":5678.3,"mw":1234.7,"dwell":0.6,"hs":"8471","line":"HMM","expl":"Risk indicators: measured weight is 78% lower than declared; unusually high value-per-kg ($378/kg); high-risk origin country.","date":"2024-02-27"},
  {"id":"CONT847291","score":93.0,"level":"Critical","origin":"VE","value":1567823,"dw":8912.4,"mw":18234.6,"dwell":112.3,"hs":"8703","line":"COSCO","expl":"Risk indicators: measured weight is 105% higher than declared; high-risk origin country; excessive dwell time (112 hrs).","date":"2024-02-26"},
  {"id":"CONT563018","score":85.5,"level":"Critical","origin":"MM","value":734521,"dw":2345.6,"mw":6789.1,"dwell":0.4,"hs":"7108","line":"ONE","expl":"Risk indicators: measured weight is 190% higher than declared; high-risk origin country; unusually short dwell time.","date":"2024-02-25"},
  {"id":"CONT291743","score":82.0,"level":"Critical","origin":"KP","value":1893456,"dw":12456.7,"mw":2134.5,"dwell":0.8,"hs":"2710","line":"MAERSK","expl":"Risk indicators: measured weight is 83% lower than declared; unusually high value-per-kg ($152/kg); high-risk origin country.","date":"2024-02-24"},
  {"id":"CONT742198","score":80.0,"level":"Critical","origin":"CF","value":456789,"dw":3456.8,"mw":9123.4,"dwell":95.7,"hs":"8517","line":"YML","expl":"Risk indicators: measured weight is 164% higher than declared; medium-risk origin country; excessive dwell time (96 hrs).","date":"2024-02-23"},
  {"id":"CONT234567","score":74.8,"level":"High","origin":"FR","value":87432,"dw":3421.0,"mw":4123.0,"dwell":18.2,"hs":"8471","line":"MAERSK","expl":"Minor flags noted (measured weight 21% higher than declared; weekend submission), but overall low risk profile.","date":"2024-02-22"},
  {"id":"CONT891237","score":73.9,"level":"High","origin":"US","value":124560,"dw":5678.0,"mw":4901.0,"dwell":22.1,"hs":"8517","line":"MSC","expl":"Minor flags noted (weekend submission; declared during off-hours), but overall low risk profile.","date":"2024-02-21"},
  {"id":"CONT456793","score":72.4,"level":"High","origin":"DE","value":98234,"dw":2345.0,"mw":3012.0,"dwell":31.5,"hs":"8703","line":"CMA CGM","expl":"Minor flags noted (measured weight 28% higher than declared), but overall low risk profile.","date":"2024-02-20"},
  {"id":"CONT678916","score":71.2,"level":"High","origin":"GB","value":67891,"dw":4567.0,"mw":3890.0,"dwell":14.3,"hs":"6110","line":"COSCO","expl":"Minor flags noted (weekend submission), but overall low risk profile.","date":"2024-02-19"},
  {"id":"CONT345683","score":70.1,"level":"High","origin":"JP","value":187654,"dw":6789.0,"mw":8234.0,"dwell":28.7,"hs":"2710","line":"EVERGREEN","expl":"Minor flags noted (measured weight 21% higher than declared; declared during off-hours), but overall low risk profile.","date":"2024-02-18"},
  {"id":"CONT901238","score":68.7,"level":"High","origin":"CA","value":54321,"dw":3456.0,"mw":2789.0,"dwell":19.8,"hs":"9013","line":"HAPAG","expl":"Minor flags noted (measured weight 19% lower than declared), but overall low risk profile.","date":"2024-02-17"},
  {"id":"CONT123462","score":67.5,"level":"High","origin":"AU","value":76543,"dw":8901.0,"mw":10234.0,"dwell":36.2,"hs":"7108","line":"ONE","expl":"Minor flags noted (measured weight 15% higher than declared; weekend submission), but overall low risk profile.","date":"2024-02-16"},
  {"id":"CONT567239","score":65.4,"level":"High","origin":"SG","value":234561,"dw":2678.0,"mw":3145.0,"dwell":12.4,"hs":"8542","line":"YML","expl":"Minor flags noted (declared during off-hours), but overall low risk profile.","date":"2024-02-15"},
  {"id":"CONT789350","score":63.8,"level":"High","origin":"NL","value":145678,"dw":5123.0,"mw":4456.0,"dwell":24.6,"hs":"2933","line":"MAERSK","expl":"Minor flags noted (measured weight 13% lower than declared), but overall low risk profile.","date":"2024-02-14"},
  {"id":"CONT234568","score":61.2,"level":"High","origin":"CH","value":98765,"dw":3789.0,"mw":4567.0,"dwell":42.3,"hs":"8471","line":"PIL","expl":"Minor flags noted (measured weight 21% higher than declared; excessive dwell time), but overall low risk profile.","date":"2024-02-13"},
  {"id":"CONT891241","score":58.9,"level":"High","origin":"US","value":56789,"dw":7234.0,"mw":6123.0,"dwell":17.8,"hs":"8517","line":"CMA CGM","expl":"Minor flags noted (weekend submission), but overall low risk profile.","date":"2024-02-12"},
  {"id":"CONT456796","score":55.3,"level":"High","origin":"DE","value":167890,"dw":4568.0,"mw":5234.0,"dwell":33.2,"hs":"8703","line":"COSCO","expl":"Minor flags noted (measured weight 15% higher than declared), but overall low risk profile.","date":"2024-02-11"},
  {"id":"CONT567245","score":49.2,"level":"Medium","origin":"CA","value":34567,"dw":3789.0,"mw":4234.0,"dwell":22.8,"hs":"8542","line":"MSC","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-10"},
  {"id":"CONT789356","score":47.7,"level":"Medium","origin":"AU","value":89012,"dw":7234.0,"mw":6789.0,"dwell":18.4,"hs":"2933","line":"CMA CGM","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-09"},
  {"id":"CONT234574","score":45.5,"level":"Medium","origin":"SG","value":156789,"dw":4567.0,"mw":5123.0,"dwell":29.1,"hs":"8471","line":"COSCO","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-08"},
  {"id":"CONT891247","score":43.3,"level":"Medium","origin":"NL","value":67890,"dw":2901.0,"mw":2567.0,"dwell":15.7,"hs":"8517","line":"EVERGREEN","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-07"},
  {"id":"CONT456802","score":41.8,"level":"Medium","origin":"CH","value":98765,"dw":8901.0,"mw":9678.0,"dwell":33.5,"hs":"8703","line":"HAPAG","expl":"Minor flags noted (measured weight 9% higher than declared), but overall low risk profile.","date":"2024-02-06"},
  {"id":"CONT678925","score":39.6,"level":"Medium","origin":"US","value":45678,"dw":3456.0,"mw":3123.0,"dwell":21.2,"hs":"6110","line":"ONE","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-05"},
  {"id":"CONT345692","score":37.4,"level":"Medium","origin":"DE","value":178901,"dw":6789.0,"mw":7345.0,"dwell":27.8,"hs":"2710","line":"YML","expl":"Minor flags noted (measured weight 8% higher than declared), but overall low risk profile.","date":"2024-02-04"},
  {"id":"CONT901247","score":34.9,"level":"Medium","origin":"FR","value":89012,"dw":2345.0,"mw":2089.0,"dwell":18.9,"hs":"9013","line":"MAERSK","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-03"},
  {"id":"CONT123471","score":31.7,"level":"Medium","origin":"GB","value":34567,"dw":5678.0,"mw":6123.0,"dwell":34.2,"hs":"7108","line":"MSC","expl":"Minor flags noted (measured weight 8% higher than declared), but overall low risk profile.","date":"2024-02-02"},
  {"id":"CONT567248","score":28.5,"level":"Medium","origin":"JP","value":156789,"dw":8901.0,"mw":8234.0,"dwell":23.6,"hs":"8542","line":"CMA CGM","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-02-01"},
  {"id":"CONT789359","score":22.3,"level":"Low","origin":"CA","value":23456,"dw":3456.0,"mw":3534.0,"dwell":24.1,"hs":"8471","line":"COSCO","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-31"},
  {"id":"CONT234577","score":19.8,"level":"Low","origin":"AU","value":67890,"dw":7890.0,"mw":7990.0,"dwell":18.7,"hs":"8517","line":"EVERGREEN","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-30"},
  {"id":"CONT891250","score":17.9,"level":"Low","origin":"SG","value":123456,"dw":4567.0,"mw":4623.0,"dwell":22.4,"hs":"8703","line":"HAPAG","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-29"},
  {"id":"CONT456805","score":15.4,"level":"Low","origin":"NL","value":56789,"dw":2901.0,"mw":2934.0,"dwell":19.8,"hs":"6110","line":"ONE","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-28"},
  {"id":"CONT678928","score":12.7,"level":"Low","origin":"US","value":89012,"dw":6789.0,"mw":6812.0,"dwell":28.3,"hs":"2710","line":"YML","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-27"},
  {"id":"CONT345695","score":10.2,"level":"Low","origin":"DE","value":34567,"dw":3456.0,"mw":3501.0,"dwell":16.9,"hs":"9013","line":"MAERSK","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-26"},
  {"id":"CONT901250","score":7.8,"level":"Low","origin":"CH","value":145678,"dw":8901.0,"mw":8956.0,"dwell":31.2,"hs":"7108","line":"MSC","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-25"},
  {"id":"CONT123474","score":5.3,"level":"Low","origin":"GB","value":67890,"dw":5678.0,"mw":5712.0,"dwell":25.7,"hs":"8542","line":"CMA CGM","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-24"},
  {"id":"CONT567251","score":4.1,"level":"Low","origin":"FR","value":23456,"dw":2345.0,"mw":2367.0,"dwell":21.3,"hs":"2933","line":"COSCO","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-23"},
  {"id":"CONT789362","score":2.8,"level":"Low","origin":"JP","value":98765,"dw":4567.0,"mw":4589.0,"dwell":27.6,"hs":"8471","line":"EVERGREEN","expl":"All shipment parameters within normal ranges. Standard clearance applicable.","date":"2024-01-22"},
];

const DataContext = createContext(null);

const toNumber = (value, fallback = 0) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
};

const REQUIRED_UPLOAD_COLUMNS = [
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

const OPTIONAL_UPLOAD_COLUMNS = ["Clearance_Status", "is_risky"];

const escapeCsvValue = (value) => {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
};

const buildCanonicalCsvFile = (rows = [], originalFileName = "upload.csv") => {
  const includeColumns = [
    ...REQUIRED_UPLOAD_COLUMNS,
    ...OPTIONAL_UPLOAD_COLUMNS.filter((col) =>
      rows.some((row) => Object.prototype.hasOwnProperty.call(row || {}, col))
    ),
  ];

  const headerLine = includeColumns.join(",");
  const dataLines = rows.map((row) =>
    includeColumns.map((col) => escapeCsvValue(row?.[col] ?? "")).join(",")
  );
  const csvText = [headerLine, ...dataLines].join("\n");
  const blob = new Blob([csvText], { type: "text/csv" });
  return new File([blob], originalFileName, { type: "text/csv" });
};

const buildRawRowLookup = (rows = []) => {
  const lookup = new Map();
  rows.forEach((row) => {
    const key = String(
      row?.Container_ID ??
      row?.container_id ??
      row?.id ??
      ""
    ).trim();
    if (key) lookup.set(key, row);
  });
  return lookup;
};

const mapPredictionRowsToShipments = (predictions = [], sourceRows = []) => {
  const rawLookup = buildRawRowLookup(sourceRows);
  return predictions.map((prediction, idx) => {
    const containerId = String(
      prediction?.container_id ??
      prediction?.id ??
      `ROW_${idx + 1}`
    );
    const raw = rawLookup.get(containerId) || {};

    return {
      id: containerId,
      score: toNumber(prediction?.risk_score, 0),
      level: prediction?.risk_level || "Low",
      origin: raw?.Origin_Country || "N/A",
      value: toNumber(raw?.Declared_Value, 0),
      dw: toNumber(raw?.Declared_Weight, 0),
      mw: toNumber(raw?.Measured_Weight, 0),
      dwell: toNumber(raw?.Dwell_Time_Hours, 0),
      hs: raw?.HS_Code || "N/A",
      line: raw?.Shipping_Line || "N/A",
      expl: prediction?.explanation_summary || "No explanation available.",
      date: raw?.Declaration_Date || "",
      confidence: toNumber(prediction?.confidence, 0),
      classifierScore: toNumber(prediction?.classifier_score, 0),
      anomalyScore: toNumber(prediction?.anomaly_score, 0),
    };
  });
};

const formatApiError = (error) => {
  const detail =
    error?.response?.data?.detail ||
    error?.response?.data?.message ||
    error?.message;
  return detail || "Prediction request failed.";
};

const isMissingRequiredFieldError = (message) =>
  typeof message === "string" &&
  message.toLowerCase().includes("missing required field");

export function DataProvider({ children }) {
  const [shipments, setShipments] = useState(SAMPLE_DATA);
  const [uploadMeta, setUploadMeta] = useState({
    filename: "sample_data.csv",
    uploadedAt: null,
    isSample: true,
    fileId: null,
    rowsReceived: SAMPLE_DATA.length,
    rowsValid: SAMPLE_DATA.length,
    summary: null,
  });
  const [isPredicting, setIsPredicting] = useState(false);
  const [lastError, setLastError] = useState("");  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const loadData = (rows, filename) => {
    setShipments(rows);
    setUploadMeta({
      filename,
      uploadedAt: new Date(),
      isSample: false,
      fileId: null,
      rowsReceived: rows.length,
      rowsValid: rows.length,
      summary: null,
    });
    setLastError("");
  };

  const resetToSample = () => {
    setShipments(SAMPLE_DATA);
    setUploadMeta({
      filename: "sample_data.csv",
      uploadedAt: null,
      isSample: true,
      fileId: null,
      rowsReceived: SAMPLE_DATA.length,
      rowsValid: SAMPLE_DATA.length,
      summary: null,
    });
    setLastError("");
  };

  const predictFromFile = async (file, sourceRows = []) => {
    if (!file) {
      throw new Error("Please select a CSV file first.");
    }

    setIsPredicting(true);
    setLastError("");

    try {
      const executeUploadAndPredict = async (uploadFile) => {
        const uploadResponse = await ContainerRiskAPI.uploadDataset(uploadFile);
        const uploadData = uploadResponse?.data;
        const fileId = uploadData?.file_id;

        if (!fileId) {
          throw new Error("Upload succeeded but no file ID was returned.");
        }

        if (uploadData?.status === "partial_success") {
          const message = uploadData?.errors?.length
            ? uploadData.errors.join("; ")
            : (uploadData?.message || "Uploaded file failed schema validation.");
          throw new Error(message);
        }

        const predictResponse = await ContainerRiskAPI.predict(fileId, true);
        const predictData = predictResponse?.data;
        return { fileId, uploadData, predictData };
      };

      let flowResult;
      try {
        flowResult = await executeUploadAndPredict(file);
      } catch (firstError) {
        const firstMessage = formatApiError(firstError);
        const canRetryWithCanonicalCsv =
          isMissingRequiredFieldError(firstMessage) &&
          Array.isArray(sourceRows) &&
          sourceRows.length > 0;

        if (!canRetryWithCanonicalCsv) {
          throw firstError;
        }

        const canonicalFile = buildCanonicalCsvFile(sourceRows, file.name);
        flowResult = await executeUploadAndPredict(canonicalFile);
      }

      const { fileId, uploadData, predictData } = flowResult;
      const predictions = predictData?.predictions || [];
      const mappedShipments = mapPredictionRowsToShipments(predictions, sourceRows);

      setShipments(mappedShipments);
      setUploadMeta({
        filename: file.name,
        uploadedAt: new Date(),
        isSample: false,
        fileId,
        rowsReceived: uploadData?.rows_received ?? sourceRows.length,
        rowsValid: uploadData?.rows_valid ?? mappedShipments.length,
        summary: predictData?.summary || null,
      });

      return {
        fileId,
        uploadData,
        predictData,
        shipments: mappedShipments,
      };
    } catch (error) {
      const message = formatApiError(error);
      setLastError(message);
      throw new Error(message);
    } finally {
      setIsPredicting(false);
    }
  };

  const retrainModel = async (file) => {
    if (!file) {
      throw new Error("Please select a CSV file first.");
    }

    setIsTraining(true);
    setLastError("");
    setTrainingStatus({ status: "queued", progress: 0, message: "Queueing training job..." });

    try {
      // Start retraining
      const retrainResponse = await ContainerRiskAPI.retrain(file);
      const retrainData = retrainResponse?.data;
      const jobId = retrainData?.job_id;

      if (!jobId) {
        throw new Error("Retraining start failed - no job ID returned");
      }

      setTrainingStatus({ status: "queued", progress: 5, message: "Training job started", jobId });

      // Poll training status
      return new Promise((resolve, reject) => {
        let attempts = 0;
        const maxAttempts = 300; // 5 minutes max polling
        
        const pollStatus = async () => {
          try {
            attempts++;
            if (attempts > maxAttempts) {
              throw new Error("Training timeout - exceeded max polling attempts");
            }

            const statusResponse = await ContainerRiskAPI.getTrainingStatus(jobId);
            const statusData = statusResponse?.data;

            setTrainingStatus({
              status: statusData?.status,
              progress: statusData?.progress || 0,
              message: statusData?.message,
              jobId: statusData?.job_id,
              metrics: statusData?.metrics,
              error: statusData?.error
            });

            if (statusData?.status === "completed") {
              // Training completed - reload model
              await ContainerRiskAPI.reloadModel();
              setIsTraining(false);
              resolve({ jobId, status: "completed", metrics: statusData?.metrics });
              return;
            }

            if (statusData?.status === "failed") {
              throw new Error(statusData?.error || "Training failed");
            }

            // Continue polling
            setTimeout(pollStatus, 500); // Poll every 500ms
          } catch (error) {
            setIsTraining(false);
            const message = formatApiError(error);
            setLastError(message);
            setTrainingStatus({ status: "failed", progress: 0, message, error: message });
            reject(error);
          }
        };

        pollStatus();
      });
    } catch (error) {
      setIsTraining(false);
      const message = formatApiError(error);
      setLastError(message);
      setTrainingStatus({ status: "failed", progress: 0, message, error: message });
      throw new Error(message);
    }
  };

  return (
    <DataContext.Provider
      value={{
        shipments,
        uploadMeta,
        isPredicting,
        lastError,
        trainingStatus,
        isTraining,
        loadData,
        resetToSample,
        predictFromFile,
        retrainModel,
      }}
    >
      {children}
    </DataContext.Provider>
  );
}

export const useData = () => useContext(DataContext);
