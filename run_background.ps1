# Safe background training launcher (11_test)
# Usage: .\run_background.ps1
# Stop:  python train_background.py --stop

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

Write-Host "[11_test] Starting safe background training supervisor..."
python train_background.py @args
