param(
    [switch]$RunSpatialSmoke,
    [switch]$SkipStatistics,
    [switch]$SkipClassifierQuality
)

$ErrorActionPreference = "Stop"

function Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Python($Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $Python $($Arguments -join ' ')"
    }
}

function Resolve-RequiredPath($PathValue, $Description) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "Missing $Description`: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

$Root = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$Root = (Resolve-Path -LiteralPath $Root).Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}
Write-Host "Using Python: $Python" -ForegroundColor DarkGray

$OffloadSummary = "results\full_offload_experiments\wpt_fix_thesis_run_offload\experiment_summary.json"
$OffloadDataset = "results\full_offload_experiments\wpt_fix_thesis_run_offload\datasets\offload_dataset_runtime_candidate.npz"
$SurrogateCheckpoint = "results\full_offload_experiments\wpt_fix_thesis_run_offload\checkpoints\offload_policy_surrogate_runtime.pt"
$ReportsDir = "results\full_offload_experiments\wpt_fix_thesis_run_offload\reports"
$FullRunRoot = "results\full_runs\wpt_fix_thesis_run"
$OffloadRoot = "results\full_offload_experiments\wpt_fix_thesis_run_offload"

$OffloadSummary = Resolve-RequiredPath $OffloadSummary "offload experiment summary"
$OffloadDataset = Resolve-RequiredPath $OffloadDataset "offload dataset"
$SurrogateCheckpoint = Resolve-RequiredPath $SurrogateCheckpoint "surrogate checkpoint"
$FullRunRoot = Resolve-RequiredPath $FullRunRoot "full trajectory run root"
$OffloadRoot = Resolve-RequiredPath $OffloadRoot "offload experiment root"

Step "Checking Python scripts"
Invoke-Python @(
    "-m", "py_compile",
    "analyze_experiment_statistics.py",
    "analyze_offload_classifier_quality.py",
    "generate_thesis_figures.py",
    "run_joint_trajectory_offload_experiment.py"
)

if (-not $SkipStatistics) {
    Step "Regenerating seed-wise statistics"
    New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
    Invoke-Python @(
        "analyze_experiment_statistics.py",
        $OffloadSummary,
        "--reference", "heuristic_offloading",
        "--output_json", (Join-Path $ReportsDir "runtime_offload_policy_statistics.json"),
        "--output_md", (Join-Path $ReportsDir "runtime_offload_policy_statistics.md")
    )
}

if (-not $SkipClassifierQuality) {
    Step "Regenerating offload classifier quality summary"
    New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
    Invoke-Python @(
        "analyze_offload_classifier_quality.py",
        "--dataset", $OffloadDataset,
        "--checkpoint", $SurrogateCheckpoint,
        "--output_json", (Join-Path $ReportsDir "classifier_quality_summary.json")
    )
}

if ($RunSpatialSmoke) {
    Step "Running tiny spatial-trace smoke test"
    Invoke-Python @(
        "run_joint_trajectory_offload_experiment.py",
        "--name", "spatial_trace_smoke",
        "--trajectory_run_root", $FullRunRoot,
        "--offload_experiment_root", $OffloadRoot,
        "--combos", "smoke_uncoord_heuristic:uncoordinated_greedy:heuristic",
        "--seeds", "42",
        "--episodes_per_seed", "1",
        "--steps_per_episode", "2",
        "--record_spatial_trace",
        "--spatial_trace_interval", "1"
    )
}

Step "Generating all thesis candidate figures"
Invoke-Python @("generate_thesis_figures.py")

Step "Done"
Write-Host "Figures are in: docs\figures" -ForegroundColor Green
Write-Host "Figure catalog: docs\figures\FIGURE_CATALOG.md" -ForegroundColor Green
Write-Host ""
Write-Host "Recommended command:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_all_thesis_figures.ps1"
Write-Host ""
Write-Host "Optional spatial trace smoke:" -ForegroundColor Yellow
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_all_thesis_figures.ps1 -RunSpatialSmoke"
