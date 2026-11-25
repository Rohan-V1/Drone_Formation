import { useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { CloudUploadRounded, DownloadRounded, ReplayRounded, ZoomInRounded, ZoomOutRounded } from "@mui/icons-material";
import { API, submitMedia } from "./mediaApi";
import "./AnalyzerPage.css";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]);
const IMAGE_LIMIT = 100 * 1024 * 1024;

function describeLayout(label) {
  const layouts = {
    V: "V formation",
    "Inverted V": "Inverted V formation",
    Wedge: "Wedge formation",
    "Line/Abreast (image-horizontal)": "Horizontal line",
    "Column/File (image-vertical)": "Vertical line",
    "Echelon (Right, image-space)": "Diagonal line",
    "Echelon (Left, image-space)": "Diagonal line",
    "Diamond/Rhombus": "Diamond formation",
    "Box/Square": "Box formation",
    "Circle/Ring": "Circle formation",
    "Line Formation (Multi-row)": "Multi-row formation",
    "Dispersed/Swarm (no fixed pattern)": "Dispersed",
  };
  return layouts[label] || label || "Unclear";
}

export default function AnalyzerPage({ onOpenFormationGuide }) {
  const inputRef = useRef(null);
  const previewUrlRef = useRef("");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [overlayScale, setOverlayScale] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const resetPreview = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = "";
    setPreviewUrl("");
  };

  const chooseFile = (candidate) => {
    if (!candidate || busy) return;
    const extension = `.${candidate.name.split(".").pop()}`.toLowerCase();
    const isImage = IMAGE_EXTENSIONS.has(extension) && (!candidate.type || candidate.type.startsWith("image/"));
    if (!isImage) {
      setError("Choose a still image: JPG, PNG, WebP, BMP, or TIFF. Video analysis is not part of this workflow.");
      return;
    }
    if (candidate.size === 0 || candidate.size > IMAGE_LIMIT) {
      setError(candidate.size === 0 ? "This image file is empty." : "The maximum image upload size is 100 MiB.");
      return;
    }
    resetPreview();
    const url = URL.createObjectURL(candidate);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setFile(candidate);
    setResult(null);
    setOverlayScale(1);
    setError("");
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await submitMedia(file, "image"));
    } catch (err) {
      setError(err instanceof TypeError
        ? "Could not reach the analysis service. Check that the backend and AI service are running, then try again."
        : (err.message || "The image could not be analyzed."));
    } finally {
      setBusy(false);
    }
  };

  const annotatedPath = result?.files?.annotated_image;
  const annotatedUrl = annotatedPath ? `${API}${annotatedPath}` : "";
  const estimatedCount = Number(result?.estimated_distinct_drone_count ?? result?.detection_count ?? 0);
  const prediction = result?.formation_prediction;

  return (
    <Box className="analyzer-page">
      <Box className="analyzer-intro">
        <Typography variant="h3">Real-Time Drone Formation Identification</Typography>
      </Box>

      <Paper
        className="analyzer-upload-card"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          chooseFile(event.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          className="analyzer-file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff"
          onChange={(event) => {
            chooseFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <CloudUploadRounded className="analyzer-upload-icon" />
        <Button variant="outlined" onClick={() => inputRef.current?.click()} disabled={busy}>
          Browse for image
        </Button>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {file && previewUrl && !result && (
        <Paper className="analyzer-media-card">
          <Box className="analyzer-media-heading">
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="subtitle1">{file.name}</Typography>
            </Box>
            <Stack direction="row" spacing={1}>
              <Button onClick={() => inputRef.current?.click()} disabled={busy}>Choose another</Button>
              <Button variant="contained" onClick={analyze} disabled={busy}>
                {busy ? <><CircularProgress size={18} sx={{ mr: 1 }} /> Analyzing…</> : "Analyze image"}
              </Button>
            </Stack>
          </Box>
          <img className="analyzer-preview" src={previewUrl} alt="Selected drone-swarm image" />
        </Paper>
      )}

      {busy && <Paper className="analyzer-status-card"><CircularProgress size={22} /><Typography>Detecting drones and assessing the image layout…</Typography></Paper>}

      {result && (
        <Paper className="analyzer-result-card" aria-live="polite">
          <Box className="analyzer-result-title">
            <Box>
              <Typography variant="h6">Results</Typography>
            </Box>
            {annotatedUrl && <Button component="a" href={annotatedUrl} download="drone-detections.png" startIcon={<DownloadRounded />}>Download boxes</Button>}
          </Box>

          <Box className="analyzer-summary-grid">
            <Box className="analyzer-summary-item">
              <Typography className="analyzer-summary-label">Drone count</Typography>
              <Typography variant="h4">{estimatedCount}</Typography>
            </Box>
            <Box className="analyzer-summary-item">
              <Typography className="analyzer-summary-label">Formation</Typography>
              <Typography variant="h6">{prediction && prediction !== "Unknown" ? describeLayout(prediction) : "Unclear"}</Typography>
            </Box>
          </Box>
          <Divider sx={{ borderColor: "#29374b", mb: 2 }} />
          <Box className="analyzer-overlay-heading">
            <Typography variant="subtitle1">Detection overlay</Typography>
            <Stack direction="row" spacing={0.5} alignItems="center">
              <Button aria-label="Zoom out" size="small" onClick={() => setOverlayScale((scale) => Math.max(1, scale - 0.25))} disabled={overlayScale <= 1}><ZoomOutRounded /></Button>
              <Typography className="analyzer-subtitle">{Math.round(overlayScale * 100)}%</Typography>
              <Button aria-label="Zoom in" size="small" onClick={() => setOverlayScale((scale) => Math.min(3, scale + 0.25))} disabled={overlayScale >= 3}><ZoomInRounded /></Button>
            </Stack>
          </Box>
          <Box className="analyzer-image-viewport">
            <img
              className="analyzer-output"
              style={{ width: `${overlayScale * 100}%` }}
              src={annotatedUrl || previewUrl}
              alt="Drone bounding boxes over the original image"
            />
          </Box>
          <Button
            className="formation-guide-button"
            variant="contained"
            onClick={() => onOpenFormationGuide?.(prediction && prediction !== "Unknown" ? prediction : "")}
          >
            Learn about this formation
          </Button>
        </Paper>
      )}
      {file && !busy && <Button className="analyzer-reset" startIcon={<ReplayRounded />} onClick={() => { setFile(null); setResult(null); setError(""); resetPreview(); }}>Start over</Button>}
    </Box>
  );
}
