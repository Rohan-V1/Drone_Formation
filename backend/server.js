
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 5000;
const AI_SERVICE_URL =
  process.env.AI_SERVICE_URL || "http://127.0.0.1:8001";

app.use(cors());
app.use(express.json());

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
