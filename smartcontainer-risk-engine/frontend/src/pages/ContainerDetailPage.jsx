import { useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from "recharts";
import { useData } from "../context/DataContext";
import { LEVEL_CONFIG, LEVEL_ORDER, ScoreBadge } from "../components/shared";

function InfoRow({ label, value, highlight }) {
  return (
    <div className="flex items-start justify-between py-3 border-b border-slate-800 last:border-0">
      <span className="text-slate-500 text-sm">{label}</span>
      <span className={`text-sm font-semibold text-right max-w-xs ${highlight ? "text-orange-400" : "text-slate-200"}`}>
        {value}
      </span>
    </div>
  );
}

export default function ContainerDetailPage() {
  const { id } = useParams();
  const { shipments } = useData();
  const navigate = useNavigate();

  const container = shipments.find(s => s.id === id);

  const allSorted = useMemo(
    () => [...shipments].sort((a, b) => b.score - a.score),
    [shipments]
  );
  const rank = allSorted.findIndex(s => s.id === id) + 1;

  if (!container) {
    return (
      <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <p className="text-5xl mb-4">🔍</p>
          <p className="text-white font-bold text-xl mb-2">Container not found</p>
          <p className="text-slate-400 mb-6">ID: {id}</p>
          <button
            onClick={() => navigate("/dashboard")}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm"
          >
            ← Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const cfg = LEVEL_CONFIG[container.level] || LEVEL_CONFIG.Low;

  const weightDiff = container.dw > 0 ? ((container.mw - container.dw) / container.dw * 100) : 0;
  const valuePkg   = container.dw > 0 ? (container.value / container.dw) : 0;

  // Radar chart — normalized risk factors 0–100
  const radarData = [
    {
      factor: "Weight Δ",
      value: Math.min(100, Math.abs(weightDiff)),
    },
    {
      factor: "Value/kg",
      value: Math.min(100, Math.log1p(valuePkg) * 8),
    },
    {
      factor: "Risk Score",
      value: container.score,
    },
    {
      factor: "Dwell Time",
      value: Math.min(100, (container.dwell / 200) * 100),
    },
    {
      factor: "Origin Risk",
      value: (() => {
        const HIGH = new Set(["KP","IR","SY","LY","YE","VE","MM","CU","BY"]);
        const MED  = new Set(["NG","PK","AF","SD","IQ","SO","ML","CD","CF","SS"]);
        if (HIGH.has(container.origin)) return 90;
        if (MED.has(container.origin))  return 55;
        return 15;
      })(),
    },
  ];

  // Previous / Next navigation
  const currentIdx = allSorted.findIndex(s => s.id === id);
  const prevContainer = currentIdx > 0 ? allSorted[currentIdx - 1] : null;
  const nextContainer = currentIdx < allSorted.length - 1 ? allSorted[currentIdx + 1] : null;

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 text-slate-200">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* Breadcrumb + nav */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm">
            <button onClick={() => navigate("/dashboard")} className="text-slate-500 hover:text-slate-300 transition-colors">
              Dashboard
            </button>
            <span className="text-slate-700">/</span>
            <span className="text-slate-300 font-mono font-semibold">{id}</span>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => prevContainer && navigate(`/container/${prevContainer.id}`)}
              disabled={!prevContainer}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300 text-xs transition-colors"
            >
              ← Prev
            </button>
            <button
              onClick={() => nextContainer && navigate(`/container/${nextContainer.id}`)}
              disabled={!nextContainer}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300 text-xs transition-colors"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Hero card */}
        <div className={`rounded-2xl border ${cfg.border} ${cfg.bg} p-6`}>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold mb-1">Container ID</p>
              <p className="text-3xl font-black font-mono text-white">{container.id}</p>
              <p className="text-slate-500 text-sm mt-1">
                Rank #{rank} of {allSorted.length} · {container.date || "No date"}
              </p>
            </div>
            <div className="flex flex-col items-start sm:items-end gap-2">
              <ScoreBadge level={container.level} score={container.score} size="lg" />
              <div className="flex items-center gap-2">
                <div className="w-32 bg-slate-800/60 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${container.score}%`, background: cfg.color }}
                  />
                </div>
                <span className="text-2xl font-black" style={{ color: cfg.color }}>
                  {container.score}
                </span>
              </div>
            </div>
          </div>

          {/* Explanation banner */}
          <div className={`mt-5 rounded-xl p-4 bg-slate-950/40 border ${cfg.border}`}>
            <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold mb-1">AI Explanation</p>
            <p className={`text-sm leading-relaxed ${cfg.text}`}>{container.expl}</p>
          </div>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Shipment details */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-300 font-semibold text-sm mb-2">Shipment Details</p>
            <InfoRow label="Origin Country"   value={container.origin} />
            <InfoRow label="Shipping Line"    value={container.line || "N/A"} />
            <InfoRow label="HS Code"          value={container.hs || "N/A"} />
            <InfoRow label="Declared Value"   value={`$${(container.value || 0).toLocaleString()}`} />
            <InfoRow label="Value per kg"     value={`$${valuePkg.toFixed(2)}/kg`} highlight={valuePkg > 500} />
            <InfoRow label="Dwell Time"       value={`${container.dwell} hours`} highlight={container.dwell > 72 || container.dwell < 2} />
          </div>

          {/* Weight analysis */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-300 font-semibold text-sm mb-4">Weight Analysis</p>
            <div className="grid grid-cols-2 gap-3 mb-4">
              {[
                { label: "Declared Weight", val: `${(container.dw || 0).toLocaleString()} kg`, hi: false },
                { label: "Measured Weight", val: `${(container.mw || 0).toLocaleString()} kg`, hi: false },
                { label: "Difference",      val: `${(container.mw - container.dw).toLocaleString()} kg`, hi: Math.abs(container.mw - container.dw) > 1000 },
                { label: "Δ Percentage",    val: `${weightDiff > 0 ? "+" : ""}${weightDiff.toFixed(1)}%`, hi: Math.abs(weightDiff) > 20 },
              ].map(({ label, val, hi }) => (
                <div key={label} className="bg-slate-800/60 rounded-xl p-3">
                  <p className="text-slate-500 text-xs mb-1">{label}</p>
                  <p className={`font-bold font-mono ${hi ? "text-orange-400" : "text-slate-200"}`}>{val}</p>
                </div>
              ))}
            </div>
            {/* Weight bar comparison */}
            <div className="space-y-2">
              {[
                { label: "Declared", val: container.dw, color: "#60a5fa" },
                { label: "Measured", val: container.mw, color: Math.abs(weightDiff) > 20 ? "#f97316" : "#22c55e" },
              ].map(({ label, val, color }) => {
                const max = Math.max(container.dw, container.mw, 1);
                return (
                  <div key={label} className="flex items-center gap-3">
                    <span className="text-xs text-slate-500 w-16 flex-shrink-0">{label}</span>
                    <div className="flex-1 bg-slate-800 rounded-full h-3 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${(val / max) * 100}%`, background: color }}
                      />
                    </div>
                    <span className="text-xs font-mono text-slate-300 w-24 text-right">{val.toLocaleString()} kg</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Radar chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
          <p className="text-slate-300 font-semibold text-sm mb-4">Risk Factor Profile</p>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#334155" />
              <PolarAngleAxis dataKey="factor" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Radar
                name="Risk"
                dataKey="value"
                stroke={cfg.color}
                fill={cfg.color}
                fillOpacity={0.2}
              />
              <Tooltip
                content={({ active, payload }) =>
                  active && payload?.length ? (
                    <div className="bg-slate-800 border border-slate-600 rounded-lg p-2 text-xs">
                      <p className="text-slate-300 font-semibold">{payload[0]?.payload?.factor}</p>
                      <p style={{ color: cfg.color }}>Score: {payload[0]?.value?.toFixed(1)}</p>
                    </div>
                  ) : null
                }
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Similar risky containers */}
        {container.level !== "Low" && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-300 font-semibold text-sm mb-4">
              Other {container.level} Containers from {container.origin}
            </p>
            <div className="space-y-2">
              {shipments
                .filter(s => s.id !== id && s.origin === container.origin && s.level === container.level)
                .slice(0, 5)
                .map(s => (
                  <div
                    key={s.id}
                    className="flex items-center justify-between py-2 px-3 rounded-lg bg-slate-800/60 hover:bg-slate-800 cursor-pointer transition-colors"
                    onClick={() => navigate(`/container/${s.id}`)}
                  >
                    <span className="font-mono text-xs text-slate-300 font-semibold">{s.id}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-slate-500 hidden sm:block truncate max-w-48">{s.expl.slice(0, 60)}…</span>
                      <ScoreBadge level={s.level} score={s.score} />
                    </div>
                  </div>
                ))}
              {shipments.filter(s => s.id !== id && s.origin === container.origin && s.level === container.level).length === 0 && (
                <p className="text-slate-600 text-sm">No other {container.level} containers from {container.origin}.</p>
              )}
            </div>
          </div>
        )}

        {/* Back button */}
        <div className="pb-4">
          <button
            onClick={() => navigate("/dashboard")}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-colors"
          >
            ← Back to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
