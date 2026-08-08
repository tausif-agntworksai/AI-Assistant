# Freezes the Python engine into desktop/../engine/dist/jarvis-engine.
# Run via `npm run build:engine`; `npm run dist` calls it before packaging.

$ErrorActionPreference = 'Stop'
$engine = Resolve-Path (Join-Path $PSScriptRoot '..\..\engine')
Set-Location $engine

Write-Host "`n=== Freezing the Jarvis engine ===`n" -ForegroundColor Cyan

$py = Join-Path $engine '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    throw "No virtualenv at $py. Run engine\setup.ps1 first."
}

Write-Host "Ensuring PyInstaller is installed..." -ForegroundColor Yellow
& $py -m pip install --quiet --upgrade pyinstaller

Write-Host "Building (this takes a few minutes)..." -ForegroundColor Yellow
& $py -m PyInstaller --noconfirm --clean jarvis-engine.spec

$exe = Join-Path $engine 'dist\jarvis-engine\jarvis-engine.exe'
if (-not (Test-Path $exe)) {
    throw "Build finished but $exe is missing."
}

# A frozen build that can't import its own dependencies still produces an exe,
# so prove it actually runs before letting electron-builder ship it.
Write-Host "`nVerifying the frozen engine..." -ForegroundColor Yellow
& $exe --doctor
if ($LASTEXITCODE -ne 0) {
    throw "The frozen engine failed its own environment check (exit $LASTEXITCODE)."
}

$size = [math]::Round((Get-ChildItem (Split-Path $exe) -Recurse |
    Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "`nEngine built: $exe ($size MB)" -ForegroundColor Green
