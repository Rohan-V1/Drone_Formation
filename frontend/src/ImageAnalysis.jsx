import { useEffect, useMemo, useRef, useState } from "react";
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
import "./ImageAnalysis.css";

const ACCEPTED = /\.(jpg|jpeg|png|webp|bmp|tif|tiff)$/i;

function possibleFormation(label) {
  const value = String(label || "");
  return /formation/i.test(value) ? `Possible ${value}` : `Possible ${value} formation`;
}

export default function ImageAnalysis({ initialFile = null, initialPreviewUrl = "" }) {
  const [file, setFile] = useState(initialFile);
  const [localPreview, setLocalPreview] = useState(initialPreviewUrl);
  const ownedPreviewRef = useRef("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const fileUrls = useMemo(() => {
    if (!result?.files) return {};
    return Object.fromEntries(Object.entries(result.files).map(([key, path]) => [key, `${API}${path}`]));
  }, [result]);

  useEffect(() => () => {
    if (ownedPreviewRef.current) URL.revokeObjectURL(ownedPreviewRef.current);
  }, []);

  const chooseImage = (event) => {
    const selected = event.target.files?.[0];
    event.target.value = "";
    setResult(null);
    setError("");
    if (!selected) return;
    if (ownedPreviewRef.current) URL.revokeObjectURL(ownedPreviewRef.current);
    ownedPreviewRef.current = "";
    if (!selected.type.startsWith("image/") || !ACCEPTED.test(selected.name)) {
      setFile(null);
      setLocalPreview("");
      setError("Choose a JPG, PNG, WebP, BMP, or TIFF image.");
    } else if (selected.size > 100 * 1024 * 1024) {
      setFile(null);
      setLocalPreview("");
      setError("The maximum image upload size is 100 MiB.");
    } else {
      const url = URL.createObjectURL(selected);
      ownedPreviewRef.current = url;
      setLocalPreview(url);
      setFile(selected);
    }
  };

  const scanImage = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const payload = await submitMedia(file, "image");
      setResult(payload);
    } catch (err) {
      setError(err.message || "The image could not be scanned.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box className="image-analysis-page">
      <Box className="image-analysis-heading">
        <Box>
          <Typography variant="h5">Image Analysis</Typography>
          <Typography className="image-analysis-subtitle">Run the local YOLO drone detector on one still image.</Typography>
        </Box>
        <Chip label="Single-image detection" color="primary" variant="outlined" />
      </Box>

      <Alert severity="info">
        Still images can suggest V/inverted V, Wedge, Diamond/Rhombus, Box/Square, Circle/Ring, multi-row lines, and image-oriented single lines. Left/right and travel direction cannot be verified from one still; irregular layouts remain Unknown rather than being assumed dispersed.
      </Alert>

      <Paper className="image-analysis-card">
        <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ sm: "center" }}>
          <Button component="label" variant="outlined" startIcon={<CloudUploadRounded />} disabled={busy}>
            Select image
            <input hidden type="file" accept="image/*,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff" onChange={chooseImage} />
          </Button>
          <Typography className="image-file-name">{file ? `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB` : "No image selected"}</Typography>
          <Button variant="contained" onClick={scanImage} disabled={!file || busy}>
            {busy ? <><CircularProgress size={18} sx={{ mr: 1 }} /> Scanning…</> : "Scan image"}
          </Button>
        </Stack>
      </Paper>

      {error && <Alert severity="error">{error}</Alert>}

      {result ? (
        <Paper className="image-analysis-card">
          <Box className="image-analysis-heading">
            <Box>
              <Typography variant="h6">Scan result</Typography>
              <Typography className="image-analysis-subtitle">
                Job {result.job_id} · {result.width}×{result.height} · {result.device}{result.gpu_name ? ` (${result.gpu_name})` : ""}
              </Typography>
            </Box>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              <Button size="small" href={fileUrls.detections_csv} download startIcon={<DownloadRounded />}>Detections CSV</Button>
              <Button size="small" href={fileUrls.metadata_json} download startIcon={<DownloadRounded />}>Metadata JSON</Button>
            </Stack>
          </Box>
          <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 2 }}>
            <Chip size="small" label={`${result.detection_count} drones detected`} />
            <Chip size="small" label="Track IDs: none (single image)" />
        <Chip size="small" label={result.formation_prediction === "Unknown" ? "Formation could not be determined" : possibleFormation(result.formation_prediction)} />
          </Stack>
          <img className="image-analysis-result" src={fileUrls.annotated_image} alt="Image with YOLO drone detections drawn over it" />
          {result.detection_count === 0 ? (
            <Alert severity="info" sx={{ mt: 2 }}>No drone detections were returned for this image.</Alert>
          ) : (
            <>
              <Typography variant="subtitle1" sx={{ mt: 2 }}>Detected boxes</Typography>
              {result.detections_truncated && <Alert severity="info" sx={{ mb: 1 }}>The table is limited to the first 200 rows; the CSV contains the full detection list.</Alert>}
              <Table size="small">
                <TableHead><TableRow><TableCell>Class</TableCell><TableCell>Confidence</TableCell><TableCell>Box (x1, y1, x2, y2)</TableCell></TableRow></TableHead>
                <TableBody>{result.detections.map((row, index) => (
                  <TableRow key={`${row.class_id}-${index}`}>
                    <TableCell>{row.class_name} ({row.class_id})</TableCell>
                    <TableCell>{Number(row.confidence).toFixed(3)}</TableCell>
                    <TableCell>{row.x1}, {row.y1}, {row.x2}, {row.y2}</TableCell>
                  </TableRow>
                ))}</TableBody>
              </Table>
            </>
          )}
          <Typography className="image-analysis-subtitle" sx={{ mt: 2 }}>{result.formation_reason}</Typography>
        </Paper>
      ) : localPreview ? (
        <Paper className="image-analysis-card">
          <Typography variant="subtitle1" sx={{ mb: 1 }}>Selected image preview</Typography>
          <img className="image-analysis-result" src={localPreview} alt="Selected image preview before scanning" />
        </Paper>
      ) : null}
    </Box>
  );
}
