# Drone Swarm Image Analyzer

This final-year project analyzes a **single still image** of a drone swarm. It runs a YOLOv8 detector, estimates the number of visible drones, and assesses their spatial arrangement. It does not process video or infer travel direction in the current user workflow.

## Run locally on Windows

Open three PowerShell terminals from the project root.

**1. AI service**

```powershell
Set-Location .\ai-service
& ..\.venv311\Scripts\python.exe -m uvicorn src.app:app --host 127.0.0.1 --port 8001
```

**2. Node API**

```powershell
Set-Location .\backend
npm start
```

**3. React frontend**

```powershell
Set-Location .\frontend
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173/`. Upload a JPG, PNG, WebP, BMP, or TIFF image and select **Analyze image**.

## Current limits

- The drone count is an estimate. Tiny, overlapping, or distant drones can be missed or falsely detected.
- Formation names describe image-space geometry only. A still image cannot show travel direction.
- Formation assessments are exploratory and can return **Layout unclear** when detections are sparse, overlapping, or do not fit a supported pattern.
- The existing detector has not yet been replaced by a completed Seraphim-trained model. The Seraphim training attempt was stopped before its first epoch completed.
- Results do not establish deployment readiness or real-world formation accuracy.

The existing detector checkpoint and formation-assessment code remain separate: changing the image-only interface does not retrain or replace either model.
