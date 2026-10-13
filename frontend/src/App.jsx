import { useEffect, useRef, useState } from "react";
import {
  AppBar,
  Box,
  Button,
  Chip,
  CssBaseline,
  Drawer,
  IconButton,
  LinearProgress,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Paper,
  Toolbar,
  Typography,
} from "@mui/material";
import {
  DashboardRounded,
  VideocamRounded,
  RadarRounded,
  AnalyticsRounded,
  SettingsRounded,
  MenuRounded,
  CloudUploadRounded,
  CircleRounded,
  GroupsRounded,
  TimelineRounded,
} from "@mui/icons-material";
import "./App.css";

const drawerWidth = 250;

const navigation = [
  { label: "Overview", icon: <DashboardRounded /> },
  { label: "Live Detection", icon: <VideocamRounded /> },
  { label: "Formation Analysis", icon: <RadarRounded /> },
  { label: "Tracking History", icon: <AnalyticsRounded /> },
  { label: "Settings", icon: <SettingsRounded /> },
];

const formations = [
  { name: "V Formation", description: "Two arms meeting at a leading point" },
  { name: "Echelon", description: "Diagonal arrangement of drones" },
  { name: "Column / File", description: "Drones arranged in a single file" },
  { name: "Line / Abreast", description: "Drones arranged side by side" },
  { name: "Diamond / Box", description: "Four-sided formation arrangement" },
  { name: "Circle", description: "Drones arranged around a center" },
];

function App() {
  const [selectedNav, setSelectedNav] = useState("Overview");
  const [connectionStatus, setConnectionStatus] = useState("checking");

  const [selectedMedia, setSelectedMedia] = useState(null);
const [previewUrl, setPreviewUrl] = useState("");
const [mediaError, setMediaError] = useState("");
const fileInputRef = useRef(null);

const handleMediaChange = (event) => {
  const file = event.target.files?.[0];

  if (!file) return;

  if (
    !file.type.startsWith("image/") &&
    !file.type.startsWith("video/")
  ) {
    setMediaError("Please select a valid image or video file.");
    event.target.value = "";
    return;
  }

  setMediaError("");
  setSelectedMedia(file);
  event.target.value = "";
};

useEffect(() => {
  if (!selectedMedia) {
    setPreviewUrl("");
    return;
  }

  const url = URL.createObjectURL(selectedMedia);
  setPreviewUrl(url);

  return () => URL.revokeObjectURL(url);
}, [selectedMedia]);

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch(
          `${import.meta.env.VITE_API_URL || "http://127.0.0.1:5000"}/health`
        );

        if (!response.ok) {
          throw new Error("Backend health check failed");
        }

        const data = await response.json();

        if (active) {
          setConnectionStatus(
            data.status === "ok" && data.aiService === "connected"
              ? "connected"
              : "degraded"
          );
        }
      } catch {
        if (active) {
          setConnectionStatus("disconnected");
        }
      }
    }

    checkBackend();

    const intervalId = setInterval(checkBackend, 5000);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, []);

  return (
    <Box className="app-shell">
      <CssBaseline />

      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
            borderRight: "1px solid #202b3b",
            background: "#101827",
            color: "#e5edf8",
          },
        }}
      >
        <Box className="brand">
          <Box className="brand-icon">
            <RadarRounded />
          </Box>
          <Box>
            <Typography className="brand-title">AEROSCAN</Typography>
            <Typography className="brand-subtitle">
              DRONE INTELLIGENCE
            </Typography>
          </Box>
        </Box>

        <Typography className="nav-heading">WORKSPACE</Typography>

        <List className="navigation">
          {navigation.map((item) => (
            <ListItemButton
              key={item.label}
              selected={selectedNav === item.label}
              onClick={() => setSelectedNav(item.label)}
              className="nav-item"
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          ))}
        </List>

        <Box className="sidebar-bottom">
          <Box className="system-indicator">
            <CircleRounded className="status-dot" />
            <Box>
              <Typography className="system-label">System status</Typography>
              <Typography className="system-value">Ready for input</Typography>
            </Box>
          </Box>
          <Typography className="sidebar-version">
            DRONE FORMATION IDENTIFICATION
          </Typography>
          <Typography className="sidebar-version">Prototype v0.1.0</Typography>
        </Box>
      </Drawer>

      <Box component="main" className="main-content">
        <AppBar position="static" elevation={0} className="topbar">
          <Toolbar className="topbar-toolbar">
            <Box>
              <Typography className="breadcrumb">
                WORKSPACE / {selectedNav.toUpperCase()}
              </Typography>
              <Typography className="page-title">
                {selectedNav}
              </Typography>
            </Box>

            <Box className="topbar-actions">
              <Chip
                icon={<CircleRounded />}
                label={
                  connectionStatus === "checking"
                    ? "Checking services..."
                    : connectionStatus === "connected"
                      ? "AI service connected"
                      : connectionStatus === "degraded"
                        ? "AI service issue"
                        : "Backend disconnected"
                }
                className={`connection-chip ${connectionStatus}`}
              />
              <IconButton className="mobile-menu" aria-label="Menu">
                <MenuRounded />
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>

  <Box className="welcome-row">
  <Box>
    <Typography className="welcome-title">
      Drone Formation Intelligence
    </Typography>

    <Typography className="welcome-description">
      Detect, track, and classify coordinated drone formations from
      images and videos.
    </Typography>
  </Box>

  {/* Hidden file picker */}
  <input
    ref={fileInputRef}
    type="file"
    accept="image/*,video/*"
    hidden
    onChange={handleMediaChange}
  />

  {/* Upload button */}
  <Button
    variant="contained"
    startIcon={<CloudUploadRounded />}
    className="upload-button"
    onClick={() => fileInputRef.current?.click()}
  >
    Upload Image / Video
  </Button>
</Box>
      </Box>
    </Box>
  );
}

export default App;
