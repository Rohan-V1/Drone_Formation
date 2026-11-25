
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const { Readable } = require("node:stream");

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 5000;
const AI_SERVICE_URL =
  process.env.AI_SERVICE_URL || "http://127.0.0.1:8001";

app.use(cors());
app.use(express.json({ limit: "1mb" }));

const JOB_FILE_ALLOWLIST = new Set([
  "annotated.mp4",
  "annotated.png",
  "detections.csv",
  "formations.csv",
  "metadata.json",
]);

app.post("/api/analyze", async (req, res) => {
  const contentType = String(req.headers["content-type"] || "").toLowerCase();
  const fileName = String(req.headers["x-file-name"] || "");
  const contentLength = Number(req.headers["content-length"] || 0);

  if (!contentType.startsWith("video/") && !contentType.startsWith("application/octet-stream")) {
    return res.status(415).json({ error: "Upload a video file." });
  }
  if (!fileName || fileName.length > 240 || /[\r\n]/.test(fileName)) {
    return res.status(400).json({ error: "A valid video filename is required." });
  }
  if (contentLength > 1024 * 1024 * 1024) {
    return res.status(413).json({ error: "Video exceeds the 1 GiB upload limit." });
  }

  try {
    const headers = { "content-type": contentType, "x-file-name": fileName };
    if (contentLength > 0) headers["content-length"] = String(contentLength);
    const aiResponse = await fetch(`${AI_SERVICE_URL}/api/analyze`, {
      method: "POST",
      headers,
      body: req,
      duplex: "half",
      signal: AbortSignal.timeout(30 * 60 * 1000),
    });
    const body = await aiResponse.text();
    res.status(aiResponse.status).type("application/json").send(body);
  } catch (error) {
    console.error("AI video analysis request failed:", error.message);
    res.status(502).json({ error: "Could not reach or complete the AI video pipeline.", detail: error.message });
  }
});

app.post("/api/analyze-image", async (req, res) => {
  const contentType = String(req.headers["content-type"] || "").toLowerCase();
  const fileName = String(req.headers["x-file-name"] || "");
  const contentLength = Number(req.headers["content-length"] || 0);
  if (!contentType.startsWith("image/") && !contentType.startsWith("application/octet-stream")) {
    return res.status(415).json({ error: "Upload an image file." });
  }
  if (!fileName || fileName.length > 240 || /[\r\n]/.test(fileName)) {
    return res.status(400).json({ error: "A valid image filename is required." });
  }
  if (contentLength > 100 * 1024 * 1024) {
    return res.status(413).json({ error: "Image exceeds the 100 MiB upload limit." });
  }
  try {
    const headers = { "content-type": contentType, "x-file-name": fileName };
    if (contentLength > 0) headers["content-length"] = String(contentLength);
    const aiResponse = await fetch(`${AI_SERVICE_URL}/api/analyze-image`, {
      method: "POST", headers, body: req, duplex: "half", signal: AbortSignal.timeout(5 * 60 * 1000),
    });
    const body = await aiResponse.text();
    res.status(aiResponse.status).type("application/json").send(body);
  } catch (error) {
    console.error("AI image analysis request failed:", error.message);
    res.status(502).json({ error: "Could not reach or complete the AI image pipeline.", detail: error.message });
  }
});

app.get("/api/jobs/:jobId/:filename", async (req, res) => {
  const { jobId, filename } = req.params;
  if (!/^[0-9a-f-]{36}$/i.test(jobId) || !JOB_FILE_ALLOWLIST.has(filename)) {
    return res.status(404).json({ error: "Job artifact not found." });
  }
  try {
    const response = await fetch(`${AI_SERVICE_URL}/api/jobs/${jobId}/${filename}`);
    if (!response.ok || !response.body) {
      const body = await response.text();
      return res.status(response.status || 502).type("application/json").send(body || "{}");
    }
    res.status(response.status);
    res.setHeader("Content-Type", response.headers.get("content-type") || "application/octet-stream");
    const length = response.headers.get("content-length");
    if (length) res.setHeader("Content-Length", length);
    Readable.fromWeb(response.body).pipe(res);
  } catch (error) {
    res.status(502).json({ error: "Could not retrieve the requested AI artifact.", detail: error.message });
  }
});

app.get("/health", async (req, res) => {
  let aiService = "unavailable";

  try {
    const response = await fetch(`${AI_SERVICE_URL}/health`);
    aiService = response.ok ? "connected" : "error";
  } catch {
    aiService = "unavailable";
  }

  res.status(200).json({
    status: "ok",
    service: "drone-formation-backend",
    aiService,
  });
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`Backend running at http://127.0.0.1:${PORT}`);
});
