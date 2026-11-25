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
import { CloudUploadRounded, DownloadRounded, ReplayRounded } from "@mui/icons-material";
import { API, submitMedia } from "./mediaApi";
import "./AnalyzerPage.css";

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]);
const VIDEO_EXTENSIONS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"]);
const IMAGE_LIMIT = 100 * 1024 * 1024;
const VIDEO_LIMIT = 1024 * 1024 * 1024;

function mediaKind(file) {
  const extension = `.${file.name.split(".").pop()}`.toLowerCase();
  if (IMAGE_EXTENSIONS.has(extension) && (!file.type || file.type.startsWith("image/"))) return "image";
  if (VIDEO_EXTENSIONS.has(extension) && (!file.type || file.type.startsWith("video/"))) return "video";
  return null;
}

function possibleFormation(label) {
  const value = String(label || "");
  return /formation/i.test(value) ? `Possible ${value}` : `Possible ${value} formation`;
}

function formationReason(reason) {
  if (!reason) return "The available observations did not support a reliable formation assessment.";
  const text = String(reason).toLowerCase();
  if (text.includes("insufficient usable tracks")) return "Not enough persistent drone tracks were available in this video window.";
  if (text.includes("sparse/fragmented")) return "The drone tracks were too short or fragmented to assess formation reliably.";
  if (text.includes("heading")) return "The video did not provide reliable travel direction for this formation distinction.";
  if (text.includes("persistent tracked drones")) return "Too few drones stayed tracked for most of this window.";
  if (text.includes("simultaneous persistent tracks")) return "Too few drones were visible and tracked together across the window.";
  if (text.includes("background motion")) return "The scene did not provide a reliable estimate of camera movement, so direction-based labels were withheld.";
  if (text.includes("camera-compensated travel is too small")) return "The tracked group moved too little relative to the scene to establish travel direction.";
  if (text.includes("relative positions changed too much")) return "The relative drone arrangement changed too much during this window.";
  if (text.includes("line-shaped")) return "This group did not have a clear line layout for the direction-based categories.";
  if (text.includes("too little movement")) return "The tracks did not move enough to estimate a reliable travel direction.";
  if (text.includes("motion directions conflict")) return "The tracked drones did not share a consistent direction of travel.";
  if (text.includes("incomplete observation window")) return "The clip ended before a complete observation window was available.";
  if (text.includes("vote share") || text.includes("class margin")) return "The model evidence did not meet its saved abstention checks.";
  if (text.includes("collinear")) return "This arrangement could fit several directional formations; heading evidence is needed to distinguish them.";
  if (text.includes("sparse")) return "Too few distinct drones were detected to assess a spatial arrangement.";
  if (text.includes("overlapping detections")) return "Overlapping boxes make the group arrangement ambiguous.";
  if (text.includes("excluded") && text.includes("overlap") && text.includes("canonical template")) return "One lower-confidence overlapping box was left out; the remaining drones did not match a supported V, Diamond/Box, or Circle pattern closely enough.";
  if (text.includes("canonical template")) return "The detected drone positions did not match a supported spatial pattern closely enough.";
  if (text.includes("ambiguous")) return "More than one formation pattern fits the detected positions.";
  if (text.includes("spatial pattern was consistent")) return "A spatial arrangement remained consistent across the tracked frames.";
  if (text.includes("class inferred from stable line geometry")) return "A stable line layout was compared with camera-compensated direction of travel.";
  if (text.includes("image-axis orientation only")) return "This line direction describes the image layout; it does not prove the drones are travelling that way.";
  if (text.includes("repeated image-space rows")) return "Several roughly parallel rows were found in the detected drone positions.";
  return reason;
}

function videoFormation(windows = []) {
  const accepted = windows.filter((window) => window.prediction && window.prediction !== "Unknown");
  if (!accepted.length) {
    return {
      label: "Formation could not be determined",
      explanation: formationReason(windows.find((window) => window.reason)?.reason),
    };
  }
  const labels = [...new Set(accepted.map((window) => window.prediction))];
  if (labels.length > 1) {
    return {
      label: "Formation could not be determined",
      explanation: "Formation suggestions varied across the clip, so no single result is reported.",
    };
  }
  return {
    label: `Possible ${labels[0]} formation`,
    explanation: `Temporal geometry supported this candidate in ${accepted.length} of ${windows.length} windows. This is an exploratory result, not a verified formation label.`,
  };
}

export default function AnalyzerPage() {
  const inputRef = useRef(null);
  const previewUrlRef = useRef("");
  const [file, setFile] = useState(null);
  const [kind, setKind] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
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
    setResult(null);
    const detectedKind = mediaKind(candidate);
    if (!detectedKind) {
      setFile(null);
      setKind(null);
      resetPreview();
      setError("Choose a supported image (JPG, PNG, WebP, BMP, TIFF) or video (MP4, MOV, AVI, MKV, WebM, MPEG).");
      return;
    }
    const limit = detectedKind === "image" ? IMAGE_LIMIT : VIDEO_LIMIT;
    if (candidate.size === 0) {
      setFile(null);
      setKind(null);
      resetPreview();
      setError("This file is empty. Choose a valid image or video.");
      return;
    }
    if (candidate.size > limit) {
      setFile(null);
      setKind(null);
      resetPreview();
      setError(`This ${detectedKind} exceeds the ${detectedKind === "image" ? "100 MiB" : "1 GiB"} upload limit.`);
      return;
    }
    resetPreview();
    const url = URL.createObjectURL(candidate);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setFile(candidate);
    setKind(detectedKind);
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file || !kind) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const payload = await submitMedia(file, kind);
      setResult({ ...payload, media_kind: kind });
    } catch (err) {
      setError(err instanceof TypeError
        ? "Could not reach the analysis service. Check that both backend and AI service are running, then try again."
        : (err.message || "The selected media could not be analyzed. Check the service status and try again."));
    } finally {
      setBusy(false);
    }
  };

  const resultFiles = result?.files || {};
  const annotatedPath = kind === "image" ? resultFiles.annotated_image : resultFiles.video;
  const annotatedUrl = annotatedPath ? `${API}${annotatedPath}` : "";
  const detectionsPath = resultFiles.detections_csv;
  const detectionCount = Number(result?.detection_count || 0);
  const estimatedDistinctCount = Number(result?.estimated_distinct_drone_count ?? detectionCount);
  const duplicateSuppressions = result?.duplicate_suppressions?.length || 0;
  const formation = kind === "image"
    ? {
      label: result?.formation_prediction && result.formation_prediction !== "Unknown"
        ? possibleFormation(result.formation_prediction)
        : "Formation could not be determined",
      explanation: formationReason(result?.formation_reason),
    }
    : videoFormation(result?.windows || []);

  return (
    <Box className="analyzer-page">
      <Box className="analyzer-intro">
        <Typography variant="h5">Analyze an image or video</Typography>
        <Typography className="analyzer-subtitle">
          Upload media to detect drones. Images can suggest V, inverted V, Wedge, Diamond/Box, Circle, and multi-row layouts; videos add travel-direction evidence.
        </Typography>
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
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,video/mpeg,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.mp4,.mov,.avi,.mkv,.webm,.mpeg,.mpg"
          onChange={(event) => {
            chooseFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <CloudUploadRounded className="analyzer-upload-icon" />
        <Typography variant="h6">Drop an image or video here</Typography>
        <Typography className="analyzer-subtitle">or choose a file · images up to 100 MiB · videos up to 1 GiB</Typography>
        <Button variant="outlined" onClick={() => inputRef.current?.click()} disabled={busy}>
          Choose image or video
        </Button>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {file && previewUrl && (
        <Paper className="analyzer-media-card">
          <Box className="analyzer-media-heading">
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="subtitle1">{file.name}</Typography>
              <Typography className="analyzer-subtitle">
                {kind === "image" ? "Image" : "Video"} · {(file.size / (1024 * 1024)).toFixed(1)} MiB
              </Typography>
            </Box>
            <Stack direction="row" spacing={1}>
              <Button onClick={() => inputRef.current?.click()} disabled={busy}>Choose another</Button>
              <Button variant="contained" onClick={analyze} disabled={busy}>
                {busy ? <><CircularProgress size={18} sx={{ mr: 1 }} /> Analyzing…</> : "Analyze"}
              </Button>
            </Stack>
          </Box>
          {kind === "image"
            ? <img className="analyzer-preview" src={previewUrl} alt="Selected image preview" />
            : <video className="analyzer-preview" src={previewUrl} controls />}
        </Paper>
      )}

      {busy && <Paper className="analyzer-status-card"><CircularProgress size={22} /><Typography>Processing your {kind} with the local drone detector…</Typography></Paper>}

      {result && (
        <Paper className="analyzer-result-card" aria-live="polite">
          <Box className="analyzer-result-title">
            <Box>
              <Typography variant="h6">Analysis results</Typography>
              <Typography className="analyzer-subtitle">
                {kind === "image"
                  ? `${result.width} × ${result.height} image`
                  : `${result.frames_processed} frames processed · ${result.duration_seconds} seconds`}
              </Typography>
            </Box>
            {annotatedUrl && (
              <Button component="a" href={annotatedUrl} download={kind === "image" ? "drone-analysis.png" : "drone-analysis.mp4"} startIcon={<DownloadRounded />}>
                Download annotated {kind}
              </Button>
            )}
          </Box>

          <Box className="analyzer-summary-grid">
            <Box className="analyzer-summary-item">
              <Typography className="analyzer-summary-label">{kind === "image" ? "ESTIMATED DISTINCT DRONES" : "DETECTION BOXES ACROSS VIDEO"}</Typography>
              <Typography variant="h4">{kind === "image" ? estimatedDistinctCount : detectionCount}</Typography>
              {kind === "image" && (
                <Typography className="analyzer-subtitle">
                  {result?.raw_detection_count ?? detectionCount} raw detector boxes
                  {duplicateSuppressions > 0 ? ` · ${duplicateSuppressions} likely duplicates suppressed` : ""}
                </Typography>
              )}
              {kind === "video" && <Typography className="analyzer-subtitle">{result.unique_track_ids_observed || 0} tracks observed</Typography>}
              {(kind === "image" ? estimatedDistinctCount : detectionCount) === 0 && <Typography className="analyzer-subtitle">No drones were detected in this media.</Typography>}
            </Box>
            <Box className="analyzer-summary-item">
              <Typography className="analyzer-summary-label">FORMATION ASSESSMENT</Typography>
              <Typography variant="h6">{formation.label}</Typography>
              <Typography className="analyzer-subtitle">{formation.explanation}</Typography>
            </Box>
          </Box>

          {result.sampling_warning && <Alert severity="info" sx={{ mb: 2 }}>{result.sampling_warning}</Alert>}
          {kind === "video" && Array.isArray(result.windows) && result.windows.length > 0 && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>Formation by video window</Typography>
              <Stack spacing={0.75}>
                {result.windows.map((window) => (
                  <Paper key={window.window} variant="outlined" sx={{ p: 1.25, borderColor: "#29374b", bgcolor: "#111d2d" }}>
                    <Typography variant="body2">
                      {`Window ${window.window} · frames ${window.sample_start}–${window.sample_end} · ${window.prediction === "Unknown" ? "Unknown" : `Possible ${window.prediction}`}`}
                    </Typography>
                    <Typography className="analyzer-subtitle">{window.reason || "Geometry and temporal evidence met the exploratory checks."}</Typography>
                  </Paper>
                ))}
              </Stack>
            </Box>
          )}
          <Divider sx={{ borderColor: "#29374b", mb: 2 }} />
          {kind === "image"
            ? <img className="analyzer-output" src={annotatedUrl} alt="YOLO drone detections and exploratory formation assessment" />
            : <video className="analyzer-output" src={annotatedUrl} controls />}
          {detectionsPath && <Button className="analyzer-secondary-download" href={`${API}${detectionsPath}`} download startIcon={<DownloadRounded />}>Download detection records</Button>}
        </Paper>
      )}

      <Typography className="analyzer-footnote">
        Formation results are exploratory. An image uses only spatial geometry; a video can additionally use temporal tracks. Neither establishes real-world classification accuracy.
      </Typography>
      {file && !busy && <Button className="analyzer-reset" startIcon={<ReplayRounded />} onClick={() => { setFile(null); setKind(null); setResult(null); setError(""); resetPreview(); }}>Start over</Button>}
    </Box>
  );
}
