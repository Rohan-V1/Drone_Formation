import { useEffect, useState } from "react";
import { AppBar, Box, Chip, CssBaseline, Toolbar, Typography } from "@mui/material";
import { CircleRounded, RadarRounded } from "@mui/icons-material";
import AnalyzerPage from "./AnalyzerPage";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export default function App() {
  const [connectionStatus, setConnectionStatus] = useState("checking");

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
              <Typography className="analyzer-brand-title">Drone Formation Analyzer</Typography>
              <Typography className="analyzer-brand-caption">REAL MEDIA ANALYSIS</Typography>
            </Box>
          </Box>
          <Box className="analyzer-nav">
            <Chip label="Real image & video" variant="outlined" className="mode-chip" />
            <Chip icon={<CircleRounded />} label={statusLabel} className={`connection-chip ${connectionStatus}`} />
          </Box>
        </Toolbar>
      </AppBar>
      <Box component="main" className="analyzer-main">
        <AnalyzerPage />
      </Box>
    </Box>
  );
}
