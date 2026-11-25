param(
  [string]$BackendUrl = "http://127.0.0.1:5000",
  [string]$ImagePath = "datasets\detection\drone_swarm_yolo\images\train\drone_swarm_0001.png"
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $root ".venv311\Scripts\python.exe"
$image = if ([System.IO.Path]::IsPathRooted($ImagePath)) { $ImagePath } else { Join-Path $root $ImagePath }
if (-not (Test-Path -LiteralPath $python)) { throw "Missing explicit project interpreter: $python" }
if (-not (Test-Path -LiteralPath $image)) { throw "Sample image not found: $image" }
$health = Invoke-RestMethod -Uri "$BackendUrl/health" -TimeoutSec 10
if ($health.status -ne "ok" -or $health.aiService -ne "connected") { throw "Backend/AI health check failed." }

$testDir = Join-Path $root ("outputs\real_image_jobs\integration_test_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $testDir | Out-Null
$resultPath = Join-Path $testDir "response.json"
$result = Invoke-RestMethod -Uri "$BackendUrl/api/analyze-image" -Method Post -InFile $image `
  -ContentType "image/jpeg" -Headers @{ "X-File-Name" = [System.IO.Path]::GetFileName($image) } -TimeoutSec 300
$result | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $resultPath -Encoding utf8
if (-not $result.job_id) { throw "Image API did not return a valid job result." }
$allowedFormations = @("Unknown", "V", "Echelon", "Column/File", "Line/Abreast", "Diamond/Box", "Circle")
if ($result.formation_prediction -notin $allowedFormations) { throw "Unexpected formation result: $($result.formation_prediction)" }
if ($result.raw_detection_count -ne $result.detection_count) { throw "Raw detection count does not match detection_count." }
if ($result.estimated_distinct_drone_count -ne $result.cleaned_detections.Count) { throw "Estimated distinct count does not match cleaned detections." }
if (($result.estimated_distinct_drone_count -lt 4 -or $result.overlapping_detection_pairs.Count -gt 0) -and $result.formation_prediction -ne "Unknown") {
  throw "Sparse cleaned detections or unresolved overlaps must not produce an image formation candidate."
}
if ($result.detector_checkpoint -ne "C:\DroneTrainingRuns\phase3h_resume02\weights\best.pt") { throw "Unexpected detector checkpoint: $($result.detector_checkpoint)" }
foreach ($entry in @(
  @{ key = "annotated_image"; name = "annotated.png" },
  @{ key = "detections_csv"; name = "detections.csv" },
  @{ key = "metadata_json"; name = "metadata.json" }
)) {
  $url = $result.files.($entry.key)
  if (-not $url) { throw "API response missing artifact URL: $($entry.key)" }
  Invoke-WebRequest -Uri ($BackendUrl + $url) -OutFile (Join-Path $testDir $entry.name) -TimeoutSec 120
}
& $python (Join-Path $root "tools\verify_image_pipeline_artifacts.py") $testDir
if ($LASTEXITCODE -ne 0) { throw "Image artifact validation failed (exit $LASTEXITCODE)." }
Write-Host "PASS: image upload -> YOLO detection -> spatial assessment -> annotated image/results verified."
Write-Host "Job: $($result.job_id); raw boxes: $($result.raw_detection_count); estimated distinct drones: $($result.estimated_distinct_drone_count); suppressed likely duplicates: $($result.duplicate_suppressions.Count); formation: $($result.formation_prediction); device: $($result.device)"
Write-Host "Server job output: outputs\real_image_jobs\$($result.job_id)"
Write-Host "Local test artifacts: $testDir"
