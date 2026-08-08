# One-time engine setup: virtualenv, dependencies, config files.
# Usage:  powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = 'Stop'
$engine = $PSScriptRoot
Set-Location $engine

Write-Host "`n=== JARVIS engine setup ===`n" -ForegroundColor Cyan

if (-not (Test-Path "$engine\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    py -3 -m venv "$engine\.venv"
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor DarkGray
}

$py = "$engine\.venv\Scripts\python.exe"

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& $py -m pip install --upgrade pip --quiet

Write-Host "Installing dependencies (this downloads ~400 MB, please wait)..." -ForegroundColor Yellow
& $py -m pip install -r "$engine\requirements.txt"

if (-not (Test-Path "$engine\config.yaml")) {
    Copy-Item "$engine\config.example.yaml" "$engine\config.yaml"
    Write-Host "Created config.yaml from the example." -ForegroundColor Green
}

if (-not (Test-Path "$engine\.env")) {
    Copy-Item "$engine\.env.example" "$engine\.env"
    Write-Host "Created .env - add your ANTHROPIC_API_KEY to it." -ForegroundColor Green
}

Write-Host "`nRunning environment check...`n" -ForegroundColor Cyan
& $py -m jarvis --doctor

Write-Host "Setup complete. Start the assistant with:  .\run-engine.bat" -ForegroundColor Cyan
