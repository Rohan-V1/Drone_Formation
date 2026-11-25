import { useEffect, useMemo, useState } from "react";
import {
  Alert, Box, Chip, FormControl, InputLabel, MenuItem, Paper, Select,
  Stack, Typography,
} from "@mui/material";
import "./SyntheticDemo.css";

const short = (text, max = 34) => text.length > max ? `${text.slice(0, max - 1)}…` : text;
const colorFor = (key) => {
  let hash = 0;
  for (const ch of key) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360} 72% 62%)`;
};

function TrackPlot({ detections = [] }) {
  const tracks = useMemo(() => {
    const grouped = new Map();
    detections.forEach((d) => {
      if (!grouped.has(d.track)) grouped.set(d.track, []);
      grouped.get(d.track).push(d);
    });
    return [...grouped.entries()];
  }, [detections]);

  return (
    <div className="demo-plot-wrap">
      <svg className="demo-plot" viewBox="0 0 640 360" role="img" aria-label="Observed synthetic drone detections and track paths">
        <rect width="640" height="360" rx="12" fill="#0b1422" />
        {[1, 2, 3].map((n) => <g key={n} stroke="#26364a" strokeDasharray="4 6"><line x1={n * 160} y1="0" x2={n * 160} y2="360" /><line x1="0" y1={n * 90} x2="640" y2={n * 90} /></g>)}
        {tracks.map(([id, points]) => {
          const color = colorFor(id);
          return <g key={id}>
            {points.length > 1 && <polyline points={points.map((p) => `${p.x},${p.y}`).join(" ")} fill="none" stroke={color} strokeWidth="1.5" opacity=".48" />}
            {points.map((p, index) => <circle key={`${id}-${p.frame}-${index}`} cx={p.x} cy={p.y} r={p.frame === 19 ? 5 : 2.5} fill={color} opacity={p.frame === 19 ? 1 : .58}><title>{`${id}: frame ${p.frame}, confidence ${p.confidence}`}</title></circle>)}
          </g>;
        })}
      </svg>
      <div className="demo-plot-legend"><span>Observed positions through time</span><span>Large dots = final frame</span></div>
    </div>
  );
}

export default function SyntheticDemo() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState("canonical");
  const [selectedId, setSelectedId] = useState("");

  useEffect(() => {
    fetch("/demo/phase6h_synthetic_demo.json")
      .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
      .then(setData)
      .catch((err) => setError(`Could not load the local synthetic demo data (${err.message}).`));
  }, []);

  const items = mode === "canonical" ? (data?.canonicalExamples || []) : (data?.unknownExamples || []);
  const selected = items.find((item) => item.id === selectedId) || items[0];

  if (error) return <Alert severity="error" sx={{ m: 3 }}>{error}</Alert>;
  if (!data) return <Typography sx={{ p: 4, color: "#9baac0" }}>Loading local synthetic demo…</Typography>;

  return (
    <Box className="page-container demo-page">
      <Stack direction={{ xs: "column", md: "row" }} justifyContent="space-between" alignItems={{ md: "center" }} spacing={2} className="demo-heading">
        <Box>
          <Typography className="welcome-title">Synthetic formation demo</Typography>
          <Typography className="welcome-description">Precomputed Phase 6H predictions on held-out Phase 6G simulated observations.</Typography>
        </Box>
        <Chip label="SYNTHETIC ONLY · NOT REAL-WORLD VALIDATION" className="synthetic-chip" />
      </Stack>

      <Alert severity="warning" className="demo-disclaimer">This demo does not run the classifier on uploaded media. Ground-truth simulation labels and model predictions are shown in separate fields. Synthetic results do not establish real-world accuracy.</Alert>

      <Box className="demo-class-grid">
        {data.classes.map((name) => <Paper key={name} className="demo-class-card"><Typography className="demo-class-name">{name}</Typography><Typography className="demo-class-count">{data.metrics.per_class[name]?.support ?? 0} held-out cases</Typography><Typography className="demo-class-score">F1 {Number(data.metrics.per_class[name]?.f1 ?? 0).toFixed(2)}</Typography></Paper>)}
      </Box>

      <Box className="demo-summary-grid">
        <Paper className="demo-summary-card"><Typography className="demo-summary-label">Noisy test accuracy</Typography><Typography className="demo-summary-value">{(data.metrics.accuracy * 100).toFixed(1)}%</Typography></Paper>
        <Paper className="demo-summary-card"><Typography className="demo-summary-label">Macro-F1</Typography><Typography className="demo-summary-value">{(data.metrics.macro_f1 * 100).toFixed(1)}%</Typography></Paper>
        <Paper className="demo-summary-card"><Typography className="demo-summary-label">Unknown challenge</Typography><Typography className="demo-summary-value">{data.unknownSummary.rejected_unknown}/{data.unknownSummary.n} rejected</Typography></Paper>
        <Paper className="demo-summary-card"><Typography className="demo-summary-label">Canonical test abstentions</Typography><Typography className="demo-summary-value">{data.metrics.selective_unknown_count}/{data.metrics.n}</Typography></Paper>
      </Box>

      <Box className="demo-controls">
        <FormControl size="small" className="demo-select"><InputLabel id="demo-mode-label">Sample set</InputLabel><Select labelId="demo-mode-label" value={mode} label="Sample set" onChange={(event) => { setMode(event.target.value); setSelectedId(""); }}><MenuItem value="canonical">Canonical held-out (36)</MenuItem><MenuItem value="unknown">Unknown challenge (160)</MenuItem></Select></FormControl>
        <FormControl size="small" className="demo-select sample-select"><InputLabel id="demo-sample-label">Example</InputLabel><Select labelId="demo-sample-label" value={selected?.id || ""} label="Example" onChange={(event) => setSelectedId(event.target.value)}>{items.map((item) => <MenuItem key={item.id} value={item.id}>{short(item.id)} · {item.prediction}</MenuItem>)}</Select></FormControl>
        <Chip label={mode === "canonical" ? "Noisy detector simulation" : "Independent synthetic Unknown challenge"} className="neutral-chip" />
      </Box>

      {selected && <Box className="demo-detail-grid">
        <Paper className="panel demo-evidence">
          <Box className="panel-header"><Box><Typography className="panel-title">Observed geometry</Typography><Typography className="panel-subtitle">{mode === "canonical" ? `${selected.id} · 20 simulated frames` : `${selected.id} · ${selected.kind} challenge`}</Typography></Box><Chip label={`${selected.usableTracks} usable tracks`} className="count-chip" /></Box>
          {mode === "canonical" ? <TrackPlot detections={selected.detections} /> : <Box className="unknown-evidence"><Typography className="demo-summary-value">{selected.usableTracks} usable tracks</Typography><Typography className="welcome-description">Mean track persistence {(selected.trackPersistence * 100).toFixed(0)}% · apparent motion {selected.motionReliable ? "reliable" : "unreliable"}</Typography><Typography className="unknown-reason">Rejection/decision evidence: {selected.reason}</Typography></Box>}
          <Typography className="demo-footnote">Dots and paths are synthetic observed detections, not ground-truth trajectories. Confidence is detector confidence, not formation probability.</Typography>
        </Paper>
        <Paper className="panel demo-decision">
          <Typography className="panel-title">Separate reference and output</Typography>
          {mode === "canonical" && <Box className="label-row"><Typography className="demo-field-label">SIMULATOR GROUND TRUTH</Typography><Chip label={selected.groundTruth} className="truth-chip" /></Box>}
          {mode === "unknown" && <Box className="label-row"><Typography className="demo-field-label">CHALLENGE REFERENCE</Typography><Chip label="Unknown / noncanonical" className="truth-chip" /></Box>}
          <Box className="label-row"><Typography className="demo-field-label">MODEL PREDICTION</Typography><Chip label={selected.prediction} className={selected.prediction === "Unknown" ? "unknown-chip" : "prediction-chip"} /></Box>
          {mode === "canonical" && <Box className="label-row"><Typography className="demo-field-label">RAW SIX-CLASS OUTPUT</Typography><Chip label={selected.rawPrediction} className="neutral-chip" /></Box>}
          <Box className="score-panel"><Typography className="demo-field-label">TOP VOTE SHARE · NOT CALIBRATED</Typography><Typography className="demo-score">{(selected.voteShare * 100).toFixed(1)}%</Typography><Typography className="demo-footnote">Top-two margin {(selected.margin * 100).toFixed(1)}%</Typography></Box>
          <Typography className="demo-field-label">EVIDENCE / DECISION REASON</Typography><Typography className="decision-reason">{selected.reason}</Typography>
          <Box className="decision-stats"><span>Persistence</span><strong>{(selected.trackPersistence * 100).toFixed(0)}%</strong><span>Motion gate</span><strong>{selected.motionReliable ? "passed" : "not reliable"}</strong>{mode === "canonical" && <><span>Observed detections</span><strong>{selected.detections.length}</strong></>}</Box>
        </Paper>
      </Box>}
      <Typography className="demo-footer-note">Class-wise test scores are synthetic-only. Unknown results are abstentions; canonical test ground truth is displayed for demonstration and was not used as an input feature.</Typography>
    </Box>
  );
}
