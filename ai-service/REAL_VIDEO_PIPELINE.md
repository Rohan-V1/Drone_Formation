# Real video analysis

> Scope update: the current React application is intentionally image-only. The video endpoint and this historical pipeline documentation remain in the repository as preserved implementation, but video analysis is not linked from the current user interface.

## Runtime path

The React page **Real Video Analysis** uploads the selected video as a streamed raw request to Express (`POST /api/analyze`). Express forwards the byte stream to FastAPI. FastAPI saves the upload into a unique job directory, validates it with OpenCV, then reads frames sequentially. A new Ultralytics YOLO instance per job runs the local `C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt` checkpoint with ByteTrack. Detection records use original frame coordinates. Annotated MP4, detections CSV, formation windows CSV, and metadata JSON are served through an allowlisted backend route.

The Phase 6H feature implementation is loaded from `outputs/phase6h_simulator_trained_classifier/run_001/train_evaluate.py`. Its `features(...)` wrapper calls the saved Phase 6A `extract_features(...)` and appends the same seven quality features. The saved model, `feature_schema.json`, `model_config.json`, and class list are checked at startup. The saved `predict_selective(...)` rules and validation-selected thresholds are used without lowering them. Inputs are sampled at approximately 10 Hz and evaluated in 20-sample windows, matching the Phase 6H observation-window length. Videos below 10 FPS, over five minutes, unsupported, or unreadable are rejected. A partial tail window is marked Unknown.

Vote share and margin are model diagnostics, not calibrated probabilities. The Phase 6H classifier was evaluated on synthetic observations; predictions on uploaded real video are exploratory and have no established real-world accuracy. Unknown is an expected outcome when track or classifier evidence is insufficient.

## Still-image analysis

The main **Drone Formation Analyzer** accepts either a still image or a video. For one image it runs the local YOLO checkpoint and then applies the unchanged Phase 5A spatial template rules to normalized detection centers. It only offers V, Diamond/Box, or Circle candidates when the geometry passes those conservative rules; it never assigns Echelon, Column/File, or Line/Abreast from a still image because those require heading. Sparse detections and overlapping boxes are reported as “Formation could not be determined.” A still image has no tracking history, so no track IDs are invented. The original-resolution annotated PNG, detections CSV, and metadata are returned; supported formats are JPG, PNG, WebP, BMP, and TIFF, up to 100 MiB.

When an older video job response lacks inline `detection_preview` rows, the video page reads only the first 30 CSV records from the existing detections artifact. This avoids requiring a service restart just to display the detection table.

## Start locally (three PowerShell windows)

From the project root, in the first window:

```powershell
Set-Location .\ai-service
& "..\.venv311\Scripts\python.exe" -m uvicorn src.app:app --host 127.0.0.1 --port 8001
```

Second window:

```powershell
Set-Location .\backend
node .\server.js
```

Third window:

```powershell
Set-Location .\frontend
node .\node_modules\vite\bin\vite.js --host 127.0.0.1
```

Open `http://127.0.0.1:5173/`. The main **Drone Formation Analyzer** page accepts either media type and selects the real inference endpoint automatically. **Synthetic Demo** remains a separate optional mode. The backend health chip should report the AI service connected.

## End-to-end local test

With all three services running, open a fourth PowerShell window at the project root and run:

```powershell
Set-Location "C:\Users\Legion\Documents\project\Drone_Formation"
& ".\tools\test_video_pipeline.ps1"
```

The test script creates a new short clip from 60 original UAVSwarm-29 training images in a unique `outputs/real_video_jobs/integration_test_<id>` folder. It uploads the clip through Express, downloads the generated artifacts, decodes every output frame, and checks CSV columns, box bounds, and window correspondence. It fails if no detector boxes or tracker IDs are returned. Each successful inference job is stored separately under `outputs/real_video_jobs/<job-id>`; these outputs are never overwritten.

The user previously reported a successful video end-to-end run: 60 frames decoded, 40 detections, 25 detection rows with track IDs, and one formation window. This execution context could not rerun Python or visually control the browser. The new still-image endpoint has only received static syntax/build checks here; run its test below in the user's normal PowerShell.

For image upload and output verification, with the same three services running:

```powershell
Set-Location "C:\Users\Legion\Documents\project\Drone_Formation"
& ".\tools\test_image_pipeline.ps1"
```

This reads one existing Dataset A training image as a pipeline smoke test, uploads it, downloads the annotated PNG and CSV, and validates dimensions, class IDs, coordinate bounds, and the explicit Unknown formation result. It does not read or modify labels, and this smoke test is not an independent model-quality evaluation.
