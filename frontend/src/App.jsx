import { useEffect, useState } from "react";
import { AppBar, Box, Button, Chip, CssBaseline, Toolbar, Typography } from "@mui/material";
import { CircleRounded, RadarRounded } from "@mui/icons-material";
import AnalyzerPage from "./AnalyzerPage";
import FormationGuide from "./FormationGuide";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export default function App() {
  const [connectionStatus, setConnectionStatus] = useState("checking");
  const [page, setPage] = useState(window.location.pathname === "/formations" ? "guide" : "analysis");
  const [selectedFormation, setSelectedFormation] = useState("");

  const navigate = (nextPage, formation = "") => {
    const path = nextPage === "guide" ? "/formations" : "/";
    window.history.pushState({}, "", path);
    setSelectedFormation(formation);
    setPage(nextPage);
    window.scrollTo({ top: 0, behavior: "auto" });
  };

  useEffect(() => {
    let active = true;
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/health`);
        const health = await response.json();
        if (!response.ok) throw new Error("Health check failed");
        if (active) setConnectionStatus(health.status === "ok" && health.aiService === "connected" ? "connected" : "degraded");
      } catch {
        if (active) setConnectionStatus("disconnected");
      }
    };
    checkBackend();
    const timer = setInterval(checkBackend, 5000);
    return () => { active = false; clearInterval(timer); };
  }, []);

  useEffect(() => {
    const syncPage = () => setPage(window.location.pathname === "/formations" ? "guide" : "analysis");
    window.addEventListener("popstate", syncPage);
    return () => window.removeEventListener("popstate", syncPage);
  }, []);

  const statusLabel = {
    checking: "Checking services…",
    connected: "AI service connected",
    degraded: "AI service unavailable",
    disconnected: "Backend disconnected",
  }[connectionStatus];

  return (
    <Box className="app-shell">
      <CssBaseline />
      <AppBar position="static" elevation={0} className="analyzer-topbar">
        <Toolbar className="analyzer-toolbar">
          <Box className="analyzer-brand">
            <RadarRounded />
            <Box>
              <Typography className="analyzer-brand-title">Drone Formation</Typography>
            </Box>
          </Box>
          <Box className="analyzer-nav">
            <Button className="header-link" onClick={() => navigate("analysis")}>Image analysis</Button>
            <Button className="header-link" onClick={() => navigate("guide", selectedFormation)}>Formation guide</Button>
            <Chip icon={<CircleRounded />} label={statusLabel} className={`connection-chip ${connectionStatus}`} />
          </Box>
        </Toolbar>
      </AppBar>
      <Box component="main" className="analyzer-main">
        {page === "guide"
          ? <FormationGuide currentFormation={selectedFormation} onBack={() => navigate("analysis")} />
          : <AnalyzerPage onOpenFormationGuide={(formation) => navigate("guide", formation)} />}
      </Box>
    </Box>
  );
}
