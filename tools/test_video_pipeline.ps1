param(
  [string]$BackendUrl = "http://127.0.0.1:5000"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv311\Scripts\python.exe"
$checkpoint = "C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt"
if (-not (Test-Path -LiteralPath $python)) { throw "Missing explicit project interpreter: $python" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing expected detector checkpoint: $checkpoint" }

$health = Invoke-RestMethod -Uri "$BackendUrl/health" -TimeoutSec 10
if ($health.status -ne "ok" -or $health.aiService -ne "connected") {
  throw "Backend/AI health check failed: $($health | ConvertTo-Json -Compress)"
}

$testDir = Join-Path $root ("outputs\real_video_jobs\integration_test_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $testDir | Out-Null
$clip = Join-Path $testDir "uavswarm29_train_clip.mp4"
$responsePath = Join-Path $testDir "response.json"
& $python (Join-Path $root "tools\create_pipeline_test_video.py") $clip --frames 60
if ($LASTEXITCODE -ne 0) { throw "Could not create local sample video (exit $LASTEXITCODE)." }

Write-Host "Uploading 60 original UAVSwarm-29 train frames through $BackendUrl/api/analyze ..."
$result = Invoke-RestMethod -Uri "$BackendUrl/api/analyze" -Method Post -InFile $clip `
  -ContentType "video/mp4" -Headers @{ "X-File-Name" = "uavswarm29_train_clip.mp4" } -TimeoutSec 1800
$result | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $responsePath -Encoding utf8
if (-not $result.job_id -or $result.frames_processed -le 0) { throw "API returned no completed video job." }
if ($result.feature_count -ne 40) { throw "Unexpected classifier feature count: $($result.feature_count)" }
if ($result.detector_checkpoint -ne $checkpoint) { throw "Pipeline used unexpected detector: $($result.detector_checkpoint)" }

foreach ($entry in @(
  @{ key = "video"; name = "annotated.mp4" },
  @{ key = "detections_csv"; name = "detections.csv" },
  @{ key = "formations_csv"; name = "formations.csv" },
  @{ key = "metadata_json"; name = "metadata.json" }
)) {
  if (-not $result.files.($entry.key)) { throw "API response is missing artifact URL: $($entry.key)" }
  Invoke-WebRequest -Uri ($BackendUrl + $result.files.($entry.key)) -OutFile (Join-Path $testDir $entry.name) -TimeoutSec 300
}

& $python (Join-Path $root "tools\verify_pipeline_artifacts.py") $testDir
if ($LASTEXITCODE -ne 0) { throw "Downloaded artifact verification failed (exit $LASTEXITCODE)." }
$detectionRows = (Import-Csv (Join-Path $testDir "detections.csv")).Count
$trackedRows = @(Import-Csv (Join-Path $testDir "detections.csv") | Where-Object { $_.track_id }).Count
if ($detectionRows -eq 0) { throw "Detector produced zero detections; review the clip/checkpoint before claiming end-to-end detection." }
if ($trackedRows -eq 0) { throw "No ByteTrack IDs were emitted on the sample; review tracker setup before claiming tracking success." }
Write-Host "PASS: upload -> detector/tracker -> Phase 6H-compatible inference -> downloads verified."
Write-Host "Job: $($result.job_id)"
Write-Host "Server job output: outputs\real_video_jobs\$($result.job_id)"
Write-Host "Local test artifacts: $testDir"
Write-Host "Detection rows: $detectionRows; rows with track IDs: $trackedRows; formation windows: $($result.windows.Count)"
