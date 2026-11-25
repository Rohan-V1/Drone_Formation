import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { CloudUploadRounded, DownloadRounded } from "@mui/icons-material";
import { API, submitMedia } from "./mediaApi";
import "./RealVideoAnalysis.css";

const ACCEPTED = /\.(mp4|mov|avi|mkv|webm|mpeg|mpg)$/i;

function parseCsvRow(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"' && quoted && line[i + 1] === '"') { value += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) { values.push(value); value = ""; }
    else value += char;
  }
  values.push(value);
  return values;
}

async function fetchDetectionPreview(url, maxRows = 30) {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Detections CSV request failed (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const lines = [];
  let pending = "";
  while (lines.length <= maxRows) {
    const { value, done } = await reader.read();
    if (done) {
      pending += decoder.decode();
      if (pending) lines.push(pending);
      break;
    }
    pending += decoder.decode(value, { stream: true });
    const parts = pending.split(/\r?\n/);
    pending = parts.pop() || "";
    lines.push(...parts);
    if (lines.length > maxRows) {
      await reader.cancel();
      break;
    }
  }
  if (!lines.length) return [];
  const headers = parseCsvRow(lines[0].replace(/^\uFEFF/, ""));
  return lines.slice(1, maxRows + 1).filter(Boolean).map((line) => {
    const cells = parseCsvRow(line);
    return Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? ""]));
  });
}

export default function RealVideoAnalysis({ initialFile = null }) {
  const [file, setFile] = useState(initialFile);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [detectionPreview, setDetectionPreview] = useState([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");

  const fileUrls = useMemo(() => {
    if (!result?.files) return {};
    return Object.fromEntries(Object.entries(result.files).map(([key, path]) => [key, `${API}${path}`]));
  }, [result]);
  const detectionCount = result?.detection_count ?? result?.processed_frames_with_detection_count ?? 0;

  useEffect(() => {
    let active = true;
    async function loadPreview() {
      setDetectionPreview([]);
      setPreviewError("");
      setPreviewLoading(false);
      if (!result) return;
      if (Array.isArray(result.detection_preview)) {
        setDetectionPreview(result.detection_preview);
        return;
      }
      if (!fileUrls.detections_csv) {
        setPreviewError("This response did not include a detections CSV.");
        return;
      }
      setPreviewLoading(true);
      try {
        const rows = await fetchDetectionPreview(fileUrls.detections_csv);
        if (active) setDetectionPreview(rows);
      } catch (err) {
        if (active) setPreviewError(err.message || "Could not load detection records.");
      } finally {
        if (active) setPreviewLoading(false);
      }
    }
    loadPreview();
    return () => { active = false; };
  }, [result, fileUrls.detections_csv]);

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    setDetectionPreview([]);
    try {
      const payload = await submitMedia(file, "video");
      setResult(payload);
    } catch (err) {
      setError(err.message || "The video could not be analyzed.");
    } finally {
      setBusy(false);
    }
  };

  const selectFile = (event) => {
    const selected = event.target.files?.[0];
    event.target.value = "";
    setResult(null);
    setError("");
    if (!selected) return;
    if (!ACCEPTED.test(selected.name)) {
      setFile(null);
      setError("Choose an MP4, MOV, AVI, MKV, WebM, MPEG, or MPG video.");
    } else if (selected.size > 1024 ** 3) {
      setFile(null);
      setError("The maximum upload size is 1 GiB.");
    } else {
      setFile(selected);
    }
  };

  return (
    <Box className="real-analysis-page">
      <Box className="real-analysis-heading">
        <Box>
          <Typography variant="h5">Real Video Analysis</Typography>
          <Typography className="real-analysis-subtitle">
            Runs the local drone detector, ByteTrack, and the saved Phase 6H classifier on your uploaded video.
          </Typography>
        </Box>
        <Chip label="Real video · local models" color="primary" variant="outlined" />
      </Box>

      <Alert severity="info" className="real-analysis-notice">
        Formation classes are exploratory. The classifier was evaluated on synthetic data, not validated for real-world accuracy.
        It returns Unknown when its saved evidence rules are not satisfied. Vote share is not a calibrated probability. For a single still image, use Image Analysis.
      </Alert>

      <Paper className="real-upload-card">
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ sm: "center" }}>
          <Button component="label" variant="outlined" startIcon={<CloudUploadRounded />} disabled={busy}>
            Select video
            <input hidden type="file" accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.mpeg,.mpg" onChange={selectFile} />
          </Button>
          <Typography className="real-file-name">{file ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB` : "No video selected"}</Typography>
          <Button variant="contained" onClick={analyze} disabled={!file || busy}>
            {busy ? <><CircularProgress size={18} sx={{ mr: 1 }} /> Processing…</> : "Analyze video"}
          </Button>
        </Stack>
      </Paper>

      {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}

      {result && (
        <>
          <Paper className="real-result-card">
            <Box className="real-analysis-heading">
              <Box>
                <Typography variant="h6">Processed result</Typography>
                <Typography className="real-analysis-subtitle">Job {result.job_id} · {result.frames_processed} frames · {result.duration_seconds}s · {result.device}{result.gpu_name ? ` (${result.gpu_name})` : ""}</Typography>
              </Box>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                <Button size="small" href={fileUrls.detections_csv} download startIcon={<DownloadRounded />}>Detections CSV</Button>
                <Button size="small" href={fileUrls.formations_csv} download startIcon={<DownloadRounded />}>Formation CSV</Button>
                {fileUrls.metadata_json && <Button size="small" href={fileUrls.metadata_json} download startIcon={<DownloadRounded />}>Metadata JSON</Button>}
              </Stack>
            </Box>
            <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 1 }}>
              <Chip size="small" label={`${detectionCount} detections`} />
              {result.tracked_detection_count != null && <Chip size="small" label={`${result.tracked_detection_count} detections assigned a track ID`} />}
              {result.unique_track_ids_observed != null && <Chip size="small" label={`${result.unique_track_ids_observed} unique track IDs`} />}
            </Stack>
            {result.sampling_warning && <Alert severity="warning" sx={{ my: 2 }}>{result.sampling_warning}</Alert>}
            <video className="real-result-video" src={fileUrls.video} controls />
            <Typography variant="subtitle1" sx={{ mt: 2 }}>Detection and tracking records</Typography>
            <Typography className="real-analysis-subtitle" sx={{ mb: 1 }}>
              Showing up to {detectionPreview.length} actual records; download the CSV for the complete per-frame results.
            </Typography>
            {previewLoading && <Typography className="real-analysis-subtitle">Loading detection preview from the saved CSV…</Typography>}
            {previewError && <Alert severity="warning" sx={{ mb: 1 }}>{previewError} The detections CSV remains available to download.</Alert>}
            {detectionPreview.length > 0 ? <Table size="small">
              <TableHead><TableRow><TableCell>Frame</TableCell><TableCell>Track ID</TableCell><TableCell>Confidence</TableCell><TableCell>Box (x1, y1, x2, y2)</TableCell></TableRow></TableHead>
              <TableBody>
                {detectionPreview.map((row, index) => (
                  <TableRow key={`${row.frame_number}-${row.track_id}-${index}`}>
                    <TableCell>{row.frame_number}</TableCell>
                    <TableCell>{row.track_id === "" ? "unassigned" : row.track_id}</TableCell>
                    <TableCell>{Number(row.confidence).toFixed(3)}</TableCell>
                    <TableCell>{row.x1}, {row.y1}, {row.x2}, {row.y2}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table> : detectionCount === 0 ? <Alert severity="info">No drone detections were returned for this video.</Alert> : <Alert severity="info">Detection records are not available in this response. Download the detections CSV for the full results.</Alert>}
          </Paper>

          <Paper className="real-result-card">
            <Typography variant="h6" gutterBottom>Formation windows</Typography>
            <Typography className="real-analysis-subtitle" sx={{ mb: 1 }}>
              Windows use the saved Phase 6H 40-feature extractor and its validation-selected abstention thresholds. No synthetic ground-truth labels are shown here.
            </Typography>
            <Table size="small">
              <TableHead><TableRow><TableCell>Window</TableCell><TableCell>Sample frames</TableCell><TableCell>Result</TableCell><TableCell>Evidence</TableCell><TableCell>Reason</TableCell></TableRow></TableHead>
              <TableBody>
                {result.windows.map((row) => (
                  <TableRow key={row.window}>
                    <TableCell>{row.window}</TableCell>
                    <TableCell>{row.sample_start}–{row.sample_end}</TableCell>
                    <TableCell><Chip size="small" label={row.prediction} color={row.prediction === "Unknown" ? "default" : "success"} /></TableCell>
                    <TableCell>{row.usable_tracks === "" ? "incomplete" : `${row.usable_tracks} tracks · vote ${Number(row.vote_share).toFixed(2)}`}</TableCell>
                    <TableCell>{row.reason}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>
        </>
      )}
    </Box>
  );
}
