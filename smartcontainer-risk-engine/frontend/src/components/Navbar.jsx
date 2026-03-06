import { NavLink, useLocation } from "react-router-dom";
import { useData } from "../context/DataContext";

const NAV_ITEMS = [
  { to: "/",         label: "Upload",    icon: "⬆" },
  { to: "/dashboard",label: "Dashboard", icon: "◫" },
  { to: "/reports",  label: "Reports",   icon: "≡" },
];

export default function Navbar() {
  const { uploadMeta, shipments, resetToSample } = useData();
  const location = useLocation();

  return (
    <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">

        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2.5 flex-shrink-0">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-black text-sm shadow-lg shadow-blue-900/40">
            ⛶
          </div>
          <div className="leading-none">
            <p className="text-white font-bold text-sm tracking-tight">CUSTOMS RISK</p>
            <p className="text-slate-500 text-[10px] tracking-widest uppercase">Intel Platform</p>
          </div>
        </NavLink>

        {/* Nav Links */}
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-blue-600 text-white shadow-lg shadow-blue-900/40"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                }`
              }
            >
              <span className="text-xs">{icon}</span>
              <span className="hidden sm:inline">{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Status pill */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <span className={`w-1.5 h-1.5 rounded-full ${uploadMeta.isSample ? "bg-yellow-500" : "bg-green-500"}`} />
            <span className="text-slate-400 font-mono truncate max-w-32">{uploadMeta.filename}</span>
            <span className="text-slate-600">·</span>
            <span className="text-slate-400">{shipments.length} containers</span>
          </div>
          {!uploadMeta.isSample && (
            <button
              onClick={resetToSample}
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors px-2 py-1 rounded hover:bg-slate-800"
              title="Reset to sample data"
            >
              ✕ Reset
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
