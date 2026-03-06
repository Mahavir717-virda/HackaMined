import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { useData } from "../context/DataContext";
import { LEVEL_CONFIG, LEVEL_ORDER, CustomTooltip } from "../components/shared";

function Section({ title, children }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800">
        <p className="text-slate-200 font-semibold text-sm">{title}</p>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

export default function ReportsPage() {
  const { shipments, uploadMeta } = useData();
  const navigate = useNavigate();

  const summary = useMemo(() => {
    const total    = shipments.length;
    const critical = shipments.filter(d => d.level === "Critical").length;
    const high     = shipments.filter(d => d.level === "High").length;
    const medium   = shipments.filter(d => d.level === "Medium").length;
    const low      = shipments.filter(d => d.level === "Low").length;
    const scores   = shipments.map(d => d.score);
    const avgScore = total ? (scores.reduce((a, b) => a + b, 0) / total).toFixed(1) : 0;
    const maxScore = total ? Math.max(...scores).toFixed(1) : 0;
    const minScore = total ? Math.min(...scores).toFixed(1) : 0;
    const flagged  = critical + high;
    const flagRate = total ? ((flagged / total) * 100).toFixed(1) : 0;
    return { total, critical, high, medium, low, avgScore, maxScore, minScore, flagged, flagRate };
  }, [shipments]);

  // Shipping line breakdown
  const lineData = useMemo(() => {
    const counts = {};
    shipments.forEach(d => {
      if (!d.line || d.line === "N/A") return;
      if (!counts[d.line]) counts[d.line] = { total: 0, flagged: 0 };
      counts[d.line].total++;
      if (d.level === "Critical" || d.level === "High") counts[d.line].flagged++;
    });
    return Object.entries(counts)
      .map(([name, v]) => ({ name, ...v, rate: ((v.flagged / v.total) * 100).toFixed(0) }))
      .sort((a, b) => b.flagged - a.flagged)
      .slice(0, 8);
  }, [shipments]);

  // HS Code breakdown
  const hsData = useMemo(() => {
    const counts = {};
    shipments.filter(d => d.level === "Critical" || d.level === "High").forEach(d => {
      if (!d.hs || d.hs === "N/A") return;
      counts[d.hs] = (counts[d.hs] || 0) + 1;
    });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, value]) => ({ name, value }));
  }, [shipments]);

  // Value at risk
  const totalValueAtRisk = useMemo(() =>
    shipments
      .filter(d => d.level === "Critical" || d.level === "High")
      .reduce((acc, d) => acc + (d.value || 0), 0),
  [shipments]);

  // Export CSV
  const exportCSV = () => {
    const headers = [
      "Container_ID", "Risk_Score", "Risk_Level", "Origin_Country",
      "Declared_Value", "Declared_Weight", "Measured_Weight",
      "Dwell_Time_Hours", "HS_Code", "Shipping_Line", "Explanation_Summary",
    ];
    const rows = shipments.map(d => [
      d.id, d.score, d.level, d.origin,
      d.value, d.dw, d.mw, d.dwell, d.hs, d.line,
      `"${(d.expl || "").replace(/"/g, "'")}"`,
    ]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `customs_risk_report_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (shipments.length === 0) {
    return (
      <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <p className="text-5xl mb-4">📊</p>
          <p className="text-white font-bold text-xl mb-6">No data to report</p>
          <button onClick={() => navigate("/")} className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm">
            ← Upload Data
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 text-slate-200">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">Reports & Summary</h1>
            <p className="text-slate-500 text-sm mt-0.5">
              {uploadMeta.filename} · {shipments.length} containers ·{" "}
              {uploadMeta.uploadedAt
                ? `uploaded ${uploadMeta.uploadedAt.toLocaleString()}`
                : "Sample dataset"}
            </p>
          </div>
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-green-700 hover:bg-green-600 text-white font-semibold text-sm transition-colors shadow-lg shadow-green-900/30"
          >
            ⬇ Export CSV
          </button>
        </div>

        {/* Headline KPIs */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { label: "Total Containers", val: summary.total.toLocaleString(), color: "#94a3b8", icon: "⛶" },
            { label: "Flagged (Crit+High)", val: summary.flagged, color: "#ef4444", icon: "⛔", sub: `${summary.flagRate}% flag rate` },
            { label: "Avg Risk Score", val: summary.avgScore, color: "#60a5fa", icon: "◎", sub: `Max: ${summary.maxScore}` },
            { label: "Value at Risk", val: `$${(totalValueAtRisk / 1e6).toFixed(1)}M`, color: "#f97316", icon: "💰", sub: "Critical + High" },
          ].map(({ label, val, color, icon, sub }) => (
            <div key={label} className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <div className="flex items-center justify-between mb-1">
                <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold">{label}</p>
                <span className="text-base opacity-50">{icon}</span>
              </div>
              <p className="text-3xl font-black" style={{ color }}>{val}</p>
              {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
            </div>
          ))}
        </div>

        {/* Risk level breakdown table */}
        <Section title="Risk Level Breakdown">
          <div className="space-y-3">
            {LEVEL_ORDER.map(level => {
              const count   = summary[level.toLowerCase()];
              const pct     = summary.total ? (count / summary.total) * 100 : 0;
              const cfg     = LEVEL_CONFIG[level];
              const valSum  = shipments.filter(d => d.level === level).reduce((a, d) => a + (d.value || 0), 0);
              return (
                <div key={level} className={`rounded-xl border ${cfg.border} ${cfg.bg} p-4`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`w-2.5 h-2.5 rounded-full ${cfg.dot}`} />
                      <span className={`font-bold text-sm ${cfg.text}`}>{level}</span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-slate-400">
                      <span>{count} containers</span>
                      <span>${(valSum / 1e6).toFixed(2)}M declared value</span>
                      <span className="font-bold text-slate-200">{pct.toFixed(1)}%</span>
                    </div>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ width: `${pct}%`, background: cfg.color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Section>

        {/* Two column charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Shipping line breakdown */}
          <Section title="Flagged Containers by Shipping Line">
            {lineData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={lineData} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 9, fill: "#64748b" }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: "#94a3b8" }} width={60} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="flagged" name="Flagged" fill="#ef4444" radius={[0, 3, 3, 0]} />
                  <Bar dataKey="total"   name="Total"   fill="#334155" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-slate-600 text-sm text-center py-8">No shipping line data</p>
            )}
          </Section>

          {/* HS Code breakdown */}
          <Section title="Flagged Containers by HS Code (Critical + High)">
            {hsData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={hsData} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 9, fill: "#64748b" }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 600 }} width={40} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Flagged" radius={[0, 3, 3, 0]}>
                    {hsData.map((_, i) => (
                      <Cell key={i} fill={`hsl(${200 + i * 15}, 70%, 55%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-slate-600 text-sm text-center py-8">No flagged containers</p>
            )}
          </Section>
        </div>

        {/* Top 10 most risky */}
        <Section title="Top 10 Highest Risk Containers">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-slate-800">
                  <th className="pb-3 text-left font-semibold">Rank</th>
                  <th className="pb-3 text-left font-semibold">Container ID</th>
                  <th className="pb-3 text-left font-semibold">Risk</th>
                  <th className="pb-3 text-left font-semibold">Origin</th>
                  <th className="pb-3 text-left font-semibold hidden sm:table-cell">Value</th>
                  <th className="pb-3 text-left font-semibold hidden md:table-cell">Dwell</th>
                  <th className="pb-3 text-left font-semibold"></th>
                </tr>
              </thead>
              <tbody>
                {[...shipments]
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 10)
                  .map((row, i) => {
                    const cfg = LEVEL_CONFIG[row.level] || LEVEL_CONFIG.Low;
                    return (
                      <tr key={row.id} className="border-b border-slate-800/50 hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 pr-4">
                          <span className={`text-xs font-black ${i === 0 ? "text-red-400" : i < 3 ? "text-orange-400" : "text-slate-500"}`}>
                            #{i + 1}
                          </span>
                        </td>
                        <td className="py-3 font-mono text-xs text-slate-300 font-bold">{row.id}</td>
                        <td className="py-3">
                          <span className={`text-xs font-bold px-2 py-0.5 rounded border font-mono ${cfg.badge}`}>
                            {row.level} · {row.score}
                          </span>
                        </td>
                        <td className="py-3">
                          <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-300 text-xs font-mono font-semibold">
                            {row.origin}
                          </span>
                        </td>
                        <td className="py-3 text-slate-400 text-xs hidden sm:table-cell">
                          ${(row.value || 0).toLocaleString()}
                        </td>
                        <td className={`py-3 text-xs font-mono hidden md:table-cell ${row.dwell > 72 ? "text-orange-400" : "text-slate-400"}`}>
                          {row.dwell} hrs
                        </td>
                        <td className="py-3">
                          <button
                            onClick={() => navigate(`/container/${row.id}`)}
                            className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                          >
                            View →
                          </button>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </Section>

        <div className="pb-4 text-center text-xs text-slate-600">
          Customs Risk Intel · AI-Powered Container Risk Assessment
        </div>
      </div>
    </div>
  );
}
