import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { DataProvider } from "./context/DataContext";
import Navbar from "./components/Navbar";
import UploadPage from "./pages/UploadPage";
import DashboardPage from "./pages/DashboardPage";
import ContainerDetailPage from "./pages/ContainerDetailPage";
import ReportsPage from "./pages/ReportsPage";

export default function App() {
  return (
    <DataProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-slate-950">
          <Navbar />
          <Routes>
            <Route path="/"                   element={<UploadPage />} />
            <Route path="/dashboard"          element={<DashboardPage />} />
            <Route path="/container/:id"      element={<ContainerDetailPage />} />
            <Route path="/reports"            element={<ReportsPage />} />
            <Route path="*"                   element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </BrowserRouter>
    </DataProvider>
  );
}
