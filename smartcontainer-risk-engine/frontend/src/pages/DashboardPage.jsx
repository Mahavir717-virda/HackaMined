import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { useData } from "../context/DataContext";
import {
  LEVEL_CONFIG, LEVEL_ORDER,
  StatCard, ScoreBar, WeightDelta, CustomTooltip,
} from "../components/shared";

export default function DashboardPage() {
  const { shipments, uploadMeta } = useData();
  const navigate = useNavigate();

  const [filterLevel, setFilterLevel] = useState("All");
  const [search, setSearch]           = useState("");
  const [sortBy, setSortBy]           = useState("score");
  const [page, setPage]               = useState(1);
  const PER_PAGE = 20;

  // ── Summary ─────────────────────────────────────────────────────────────
  const summary = useMemo(() => ({
    total:    shipments.length,
    critical: shipments.filter(d => d.level === "Critical").length,
    high:     shipments.filter(d => d.level === "High").length,
    medium:   shipments.filter(d => d.level === "Medium").length,
    low:      shipments.filter(d => d.level === "Low").length,
    avgScore: shipments.length
      ? (shipments.reduce((a, b) => a + b.score, 0) / shipments.length).toFixed(1)
      : 0,
  }), [shipments]);

  // ── Chart data ────────────────────────────────────────────────────────────
  const pieData = useMemo(() =>
    LEVEL_ORDER
      .map(l => ({ name: l, value: shipments.filter(d => d.level === l).length, color: LEVEL_CONFIG[l].color }))
      .filter(d => d.value > 0),
  [shipments]);

  const scoreDistData = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({ range: `${i * 10}–${i * 10 + 9}`, count: 0 }));
    shipments.forEach(d => {
      const idx = Math.min(9, Math.floor(d.score / 10));
      buckets[idx].count++;
    });
    return buckets;
  }, [shipments]);

  const originData = useMemo(() => {
    const counts = {};
    shipments
      .filter(d => d.level === "Critical" || d.level === "High")
      .forEach(d => { counts[d.origin] = (counts[d.origin] || 0) + 1; });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, value]) => ({ name, value }));
  }, [shipments]);

  // ── Filtered / sorted rows ────────────────────────────────────────────────
  const filtered = useMemo(() => {
    let d = shipments;
    if (filterLevel !== "All") d = d.filter(x => x.level === filterLevel);
    if (search) {
      const q = search.toLowerCase();
      d = d.filter(x =>
        x.id.toLowerCase().includes(q) ||
        x.origin.toLowerCase().includes(q) ||
        (x.line || "").toLowerCase().includes(q) ||
        (x.hs || "").includes(q)
      );
    }
    return [...d].sort((a, b) => {
      if (sortBy === "score") return b.score - a.score;
      if (sortBy === "value") return b.value - a.value;
      if (sortBy === "dwell") return b.dwell - a.dwell;
      return 0;
    });
  }, [shipments, filterLevel, search, sortBy]);

  const totalPages = Math.ceil(filtered.length / PER_PAGE);
  const paginated  = filtered.slice((page - 1) * PER_PAGE, page * PER_PAGE);

  const handleFilter = (lvl) => { setFilterLevel(lvl); setPage(1); };
  const handleSearch = (e) => { setSearch(e.target.value); setPage(1); };

  if (shipments.length === 0) {
    return (
      <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 flex items-center justify-center">
        <div className="text-center">
          <p className="text-6xl mb-4">📭</p>
          <p className="text-white font-bold text-xl mb-2">No data loaded</p>
          <p className="text-slate-400 mb-6">Upload a predictions CSV to get started.</p>
          <button
            onClick={() => navigate("/")}
            className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm transition-colors"
          >
            ← Go to Upload
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-3.5rem)] bg-slate-950 text-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-6">

        {/* Page header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-black text-white tracking-tight">Risk Dashboard</h1>
            <p className="text-slate-500 text-sm mt-0.5">
              {uploadMeta.isSample ? "Sample dataset" : uploadMeta.filename}
              {uploadMeta.uploadedAt && ` · uploaded ${uploadMeta.uploadedAt.toLocaleTimeString()}`}
            </p>
          </div>
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm transition-colors border border-slate-700"
          >
            <span>⬆</span> <span className="hidden sm:inline">Upload New</span>
          </button>
        </div>

        {/* KPI strip */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard label="Total"    value={summary.total.toLocaleString()} color="#94a3b8" icon="⛶" />
          <StatCard label="Avg Score" value={summary.avgScore}              color="#60a5fa" icon="◎" sub="out of 100" />
          <StatCard label="Critical" value={summary.critical}               color="#ef4444" icon="⛔" sub={`${((summary.critical / summary.total) * 100).toFixed(0)}% of total`} />
          <StatCard label="High"     value={summary.high}                   color="#f97316" icon="⚠" sub={`${((summary.high / summary.total) * 100).toFixed(0)}% of total`} />
          <StatCard label="Medium"   value={summary.medium}                 color="#eab308" icon="◐" sub={`${((summary.medium / summary.total) * 100).toFixed(0)}% of total`} />
          <StatCard label="Low / Clear" value={summary.low}                 color="#22c55e" icon="✓" sub={`${((summary.low / summary.total) * 100).toFixed(0)}% of total`} />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Pie */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-sm font-semibold text-slate-300 mb-4">Risk Level Distribution</p>
            <ResponsiveContainer width="100%" height={170}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={48} outerRadius={76} paddingAngle={3} dataKey="value">
                  {pieData.map(e => <Cell key={e.name} fill={e.color} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-3 mt-2">
              {pieData.map(d => (
                <button
                  key={d.name}
                  onClick={() => handleFilter(d.name)}
                  className="text-xs flex items-center gap-1.5 text-slate-400 hover:text-slate-200 transition-colors"
                >
                  <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                  {d.name} ({d.value})
                </button>
              ))}
            </div>
          </div>

          {/* Score dist */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-sm font-semibold text-slate-300 mb-4">Score Distribution</p>
            <ResponsiveContainer width="100%" height={170}>
              <BarChart data={scoreDistData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                <XAxis dataKey="range" tick={{ fontSize: 8, fill: "#64748b" }} />
                <YAxis tick={{ fontSize: 9, fill: "#64748b" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" name="Containers" radius={[3, 3, 0, 0]}>
                  {scoreDistData.map((_, i) => {
                    const s = i * 10;
                    const c = s >= 75 ? "#ef4444" : s >= 50 ? "#f97316" : s >= 25 ? "#eab308" : "#22c55e";
                    return <Cell key={i} fill={c} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top origins */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-sm font-semibold text-slate-300 mb-4">High-Risk Origins (Critical + High)</p>
            {originData.length > 0 ? (
              <ResponsiveContainer width="100%" height={170}>
                <BarChart data={originData} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 9, fill: "#64748b" }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8", fontWeight: 600 }} width={28} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Flagged" fill="#ef4444" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[170px] flex items-center justify-center text-slate-600 text-sm">
                No critical/high containers
              </div>
            )}
          </div>
        </div>

        {/* Table section */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden">
          {/* Toolbar */}
          <div className="p-4 border-b border-slate-800 flex flex-wrap gap-3 items-center">
            <input
              type="text"
              placeholder="Search ID, origin, HS code, shipping line…"
              value={search}
              onChange={handleSearch}
              className="flex-1 min-w-48 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
            />
            <div className="flex gap-1.5 flex-wrap">
              {["All", ...LEVEL_ORDER].map(l => (
                <button
                  key={l}
                  onClick={() => handleFilter(l)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors border ${
                    filterLevel === l
                      ? l === "All"
                        ? "bg-blue-600 text-white border-blue-500"
                        : `${LEVEL_CONFIG[l].bg} ${LEVEL_CONFIG[l].text} ${LEVEL_CONFIG[l].border}`
                      : "bg-slate-800 text-slate-400 border-slate-700 hover:border-slate-500"
                  }`}
                >
                  {l} {l !== "All" && `(${summary[l.toLowerCase()]})`}
                </button>
              ))}
            </div>
            <select
              value={sortBy}
              onChange={e => { setSortBy(e.target.value); setPage(1); }}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none"
            >
              <option value="score">Sort: Risk Score</option>
              <option value="value">Sort: Declared Value</option>
              <option value="dwell">Sort: Dwell Time</option>
            </select>
            <span className="text-xs text-slate-500">{filtered.length} results</span>
          </div>

          {/* Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs text-slate-500 uppercase tracking-wider">
                  <th className="px-4 py-3 text-left font-semibold">Container ID</th>
                  <th className="px-4 py-3 text-left font-semibold">Risk</th>
                  <th className="px-4 py-3 text-left font-semibold">Origin</th>
                  <th className="px-4 py-3 text-left font-semibold hidden md:table-cell">Value</th>
                  <th className="px-4 py-3 text-left font-semibold hidden lg:table-cell">Weight Δ</th>
                  <th className="px-4 py-3 text-left font-semibold hidden lg:table-cell">Dwell (hrs)</th>
                  <th className="px-4 py-3 text-left font-semibold hidden xl:table-cell">Summary</th>
                  <th className="px-4 py-3 text-left font-semibold w-16"></th>
                </tr>
              </thead>
              <tbody>
                {paginated.map((row, i) => (
                  <tr
                    key={row.id}
                    className={`border-b border-slate-800/50 cursor-pointer transition-colors hover:bg-slate-800 ${i % 2 === 0 ? "bg-slate-900" : "bg-slate-900/40"}`}
                    onClick={() => navigate(`/container/${row.id}`)}
                  >
                    <td className="px-4 py-3 font-mono text-slate-300 font-bold text-xs">{row.id}</td>
                    <td className="px-4 py-3"><ScoreBar score={row.score} level={row.level} /></td>
                    <td className="px-4 py-3">
                      <span className="px-2 py-0.5 bg-slate-800 rounded text-slate-300 text-xs font-mono font-semibold border border-slate-700">
                        {row.origin}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-400 hidden md:table-cell text-xs">
                      ${(row.value || 0).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 hidden lg:table-cell">
                      <WeightDelta dw={row.dw} mw={row.mw} />
                    </td>
                    <td className={`px-4 py-3 font-mono text-xs hidden lg:table-cell ${
                      row.dwell > 72 ? "text-orange-400" : row.dwell < 2 ? "text-yellow-400" : "text-slate-400"
                    }`}>
                      {row.dwell}
                    </td>
                    <td className="px-4 py-3 text-slate-500 text-xs max-w-xs truncate hidden xl:table-cell">{row.expl}</td>
                    <td className="px-4 py-3">
                      <span className="text-slate-600 hover:text-slate-300 text-xs transition-colors">Details →</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filtered.length === 0 && (
              <div className="py-16 text-center text-slate-500">
                <p className="text-4xl mb-3">🔍</p>
                <p className="font-medium">No containers match your filters</p>
              </div>
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="px-4 py-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-500">
              <span>
                Showing {(page - 1) * PER_PAGE + 1}–{Math.min(page * PER_PAGE, filtered.length)} of {filtered.length}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300 transition-colors"
                >
                  ← Prev
                </button>
                <span className="px-3 py-1.5 text-slate-400">
                  {page} / {totalPages}
                </span>
                <button
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed text-slate-300 transition-colors"
                >
                  Next →
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
