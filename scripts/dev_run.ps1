Param(
  [int]$Seconds = 15,
  [switch]$Gui
)

$ErrorActionPreference = 'Stop'

Write-Host "Starting atomd (daemon)..."
$daemon = Start-Process cargo -ArgumentList 'run','-q','-p','atomd' -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 1

if ($Gui) {
  Write-Host "Starting atom-ide (GUI: winit-ui)..."
  $ide = Start-Process cargo -ArgumentList 'run','-q','-p','atom-ide','--features','winit-ui' -PassThru
} else {
  Write-Host "Starting atom-ide (headless ping)..."
  $ide = Start-Process cargo -ArgumentList 'run','-q','-p','atom-ide' -PassThru -NoNewWindow
}

Write-Host "Running for $Seconds seconds..."
Start-Sleep -Seconds $Seconds

Write-Host "Stopping processes..."
if ($ide -and !$ide.HasExited) { Stop-Process -Id $ide.Id -Force }
if ($daemon -and !$daemon.HasExited) { Stop-Process -Id $daemon.Id -Force }

Write-Host "Done."
