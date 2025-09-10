Param(
  [int]$Port = 8099,
  [string]$Dir = "reports",
  [string]$ApiBase = "http://127.0.0.1:8080",
  [switch]$Stop
)

$ErrorActionPreference = 'SilentlyContinue'
$pidFile = Join-Path $env:TEMP 'dashboard_static.pid'

if ($Stop) {
  if (Test-Path $pidFile) {
    $pid = Get-Content $pidFile
    Stop-Process -Id $pid -ErrorAction SilentlyContinue
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
  Write-Host "Dashboard static server stopped."
  exit 0
}

if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir | Out-Null }

if (Test-Path $pidFile) {
  $pid = Get-Content $pidFile
  $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
  if ($proc) {
    Write-Host "[dashboard] Static server already running (pid=$pid)"
    Write-Host "Open: http://127.0.0.1:$Port/dashboard.html?api=$ApiBase"
    exit 0
  }
}

# Prefer 'py -3', fallback to 'python'
$argsPy = @('-3','-m','http.server', "$Port", '--directory', "$Dir")
$proc = $null
try {
  $proc = Start-Process -WindowStyle Hidden -PassThru -FilePath 'py' -ArgumentList $argsPy
} catch {
  $argsPy = @('-m','http.server', "$Port", '--directory', "$Dir")
  $proc = Start-Process -WindowStyle Hidden -PassThru -FilePath 'python' -ArgumentList $argsPy
}

if ($proc -and $proc.Id) {
  Set-Content -Path $pidFile -Value $proc.Id
  Write-Host "[dashboard] Serving $Dir on http://127.0.0.1:$Port (static)"
  Write-Host "Open: http://127.0.0.1:$Port/dashboard.html?api=$ApiBase"
} else {
  Write-Error "Failed to start Python http.server"
  exit 1
}

