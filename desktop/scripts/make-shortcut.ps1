<#
  Puts a "Jarvis" shortcut on the Desktop so the app opens with one
  double-click — no terminal, no console window.

  -Source repo launches this folder with the local Electron runtime, reading
  config.yaml and .env straight out of engine\. It needs the repo to stay put.

  -Source packaged points at the .exe from `npm run dist`, which keeps its
  settings in %LOCALAPPDATA%\Jarvis instead.

  -Source auto (the default) prefers the packaged build when one exists.
#>
[CmdletBinding()]
param(
  # Where to put the shortcut. Defaults to the current user's Desktop.
  [string] $Destination = [Environment]::GetFolderPath('Desktop'),
  [string] $Name = 'Jarvis',
  [ValidateSet('auto', 'repo', 'packaged')]
  [string] $Source = 'auto'
)

$ErrorActionPreference = 'Stop'

$desktopDir = Split-Path -Parent $PSScriptRoot
$icon       = Join-Path $desktopDir 'build\icon.ico'
$packaged   = Join-Path $desktopDir 'release\win-unpacked\Jarvis.exe'
$electron   = Join-Path $desktopDir 'node_modules\electron\dist\electron.exe'

$usePackaged = switch ($Source) {
  'packaged' {
    if (-not (Test-Path -LiteralPath $packaged)) {
      throw "No packaged build at $packaged. Run 'npm run dist' first."
    }
    $true
  }
  'repo' {
    if (-not (Test-Path -LiteralPath $electron)) {
      throw "Electron isn't installed yet. Run 'npm install' in $desktopDir first."
    }
    $false
  }
  default { Test-Path -LiteralPath $packaged }
}

if ($usePackaged) {
  $target = $packaged
  $arguments = ''
} elseif (Test-Path -LiteralPath $electron) {
  # electron.exe is a GUI app, so launching it shows no console window.
  $target = $electron
  $arguments = """$desktopDir"""
} else {
  throw "Electron isn't installed yet. Run 'npm install' in $desktopDir first."
}

if (-not (Test-Path -LiteralPath (Join-Path $desktopDir 'dist\main.js'))) {
  Write-Warning "desktop\dist\main.js is missing - run 'npm run build' or the shortcut won't start."
}

# The frozen Python engine is what actually listens; without it the app opens
# and immediately reports it can't start.
$engine = Join-Path $desktopDir '..\engine\dist\jarvis-engine\jarvis-engine.exe'
if ($target -eq $electron -and -not (Test-Path -LiteralPath $engine)) {
  Write-Warning "engine\dist\jarvis-engine is missing - run 'npm run build:engine' or Jarvis won't start."
}

$linkPath = Join-Path $Destination "$Name.lnk"
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($linkPath)
$link.TargetPath = $target
$link.Arguments = $arguments
$link.WorkingDirectory = $desktopDir
$link.Description = 'Jarvis - a bilingual voice assistant for Windows'
$link.WindowStyle = 1
if (Test-Path -LiteralPath $icon) { $link.IconLocation = "$icon,0" }
$link.Save()

Write-Output "Shortcut created: $linkPath"
Write-Output "  target : $target $arguments"
