// ── Shared constants & reusable components ──────────────────────────────────

export const LEVEL_CONFIG = {
  Critical: {
    color: "#ef4444",
    bg: "bg-red-950",
    border: "border-red-800",
    text: "text-red-400",
    badge: "bg-red-900 text-red-300 border-red-700",
    dot: "bg-red-500",
    ring: "ring-red-500/30",
  },
  High: {
    color: "#f97316",
    bg: "bg-orange-950",
    border: "border-orange-800",
    text: "text-orange-400",
    badge: "bg-orange-900 text-orange-300 border-orange-700",
    dot: "bg-orange-500",
    ring: "ring-orange-500/30",
  },
  Medium: {
    color: "#eab308",
    bg: "bg-yellow-950",
    border: "border-yellow-800",
    text: "text-yellow-400",
    badge: "bg-yellow-900 text-yellow-300 border-yellow-700",
    dot: "bg-yellow-500",
    ring: "ring-yellow-500/30",
  },
  Low: {
    color: "#22c55e",
    bg: "bg-green-950",
    border: "border-green-800",
    text: "text-green-400",
    badge: "bg-green-900 text-green-300 border-green-700",
    dot: "bg-green-500",
    ring: "ring-green-500/30",
  },
};

export const LEVEL_ORDER = ["Critical", "High", "Medium", "Low"];

// ── ScoreBadge ───────────────────────────────────────────────────────────────
export function ScoreBadge({ level, score, size = "sm" }) {
  const cfg = LEVEL_CONFIG[level] || LEVEL_CONFIG.Low;
  const sizeClass = size === "lg"
    ? "px-3 py-1 text-sm gap-2"
    : "px-2 py-0.5 text-xs gap-1.5";
  return (
    <span className={`inline-flex items-center rounded font-mono font-bold border ${cfg.badge} ${sizeClass}`}>
      <span className={`rounded-full flex-shrink-0 ${cfg.dot} ${size === "lg" ? "w-2 h-2" : "w-1.5 h-1.5"}`} />
      {level} · {score}
    </span>
  );
}

// ── StatCard ─────────────────────────────────────────────────────────────────
export function StatCard({ label, value, color, sub, icon }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 flex flex-col gap-1 hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold">{label}</p>
        {icon && <span className="text-lg opacity-60">{icon}</span>}
      </div>
      <p className="text-3xl font-black tracking-tight" style={{ color }}>{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

// ── ScoreBar ─────────────────────────────────────────────────────────────────
export function ScoreBar({ score, level }) {
  const cfg = LEVEL_CONFIG[level] || LEVEL_CONFIG.Low;
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden flex-shrink-0">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${score}%`, background: cfg.color }}
        />
      </div>
      <ScoreBadge level={level} score={score} />
    </div>
  );
}

// ── Custom recharts tooltip ───────────────────────────────────────────────────
export const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload?.length) {
    return (
      <div className="bg-slate-800 border border-slate-600 rounded-lg p-3 text-xs shadow-xl">
        {label && <p className="text-slate-400 mb-1">{label}</p>}
        {payload.map((p, i) => (
          <p key={i} className="font-semibold" style={{ color: p.color || "#e2e8f0" }}>
            {p.name}: {p.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ── WeightDeltaBadge ─────────────────────────────────────────────────────────
export function WeightDelta({ dw, mw }) {
  if (!dw || dw === 0) return <span className="text-slate-500 text-xs">—</span>;
  const pct = ((mw - dw) / dw * 100).toFixed(1);
  const isHigh = Math.abs(pct) > 20;
  return (
    <span className={`font-mono text-xs font-semibold ${isHigh ? "text-orange-400" : "text-slate-400"}`}>
      {pct > 0 ? `+${pct}` : pct}%
    </span>
  );
}

// ── parse uploaded CSV ────────────────────────────────────────────────────────
export function parseCSV(text) {
  const lines = text.split("\n").filter(Boolean);
  if (lines.length < 2) throw new Error("CSV must have at least a header row and one data row.");
  const headers = lines[0].split(",").map(h => h.trim().replace(/"/g, ""));

  return lines.slice(1).map((line, idx) => {
    const vals = line.split(",").map(v => v.trim().replace(/"/g, ""));
    const obj = {};
    headers.forEach((h, i) => (obj[h] = vals[i] || ""));

    return {
      id:    obj["Container_ID"]  || obj["id"]    || `ROW_${idx + 1}`,
      score: parseFloat(obj["Risk_Score"]   || obj["score"] || 0),
      level: obj["Risk_Level"]    || obj["level"] || "Low",
      origin:obj["Origin_Country"]|| obj["origin"]|| "N/A",
      value: parseFloat(obj["Declared_Value"]  || obj["value"]|| 0),
      dw:    parseFloat(obj["Declared_Weight"] || obj["dw"]  || 0),
      mw:    parseFloat(obj["Measured_Weight"] || obj["mw"]  || 0),
      dwell: parseFloat(obj["Dwell_Time_Hours"]|| obj["dwell"]||0),
      hs:    obj["HS_Code"]       || obj["hs"]    || "N/A",
      line:  obj["Shipping_Line"] || obj["line"]  || "N/A",
      expl:  obj["Explanation_Summary"] || obj["expl"] || "No explanation available.",
      date:  obj["date"] || obj["Declaration_Date"] || "",
    };
  });
}
