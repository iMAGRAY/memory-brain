#requires -Version 5.1
param()

$ErrorActionPreference = 'Stop'
Write-Host '== AI Memory Service: Deterministic Verification (Windows) ==' -ForegroundColor Cyan

Set-Location (Join-Path $PSScriptRoot '..')

# Global timeout guard (seconds)
$GLOBAL_TIMEOUT_SEC = [int]([Environment]::GetEnvironmentVariable('VERIFY_TIMEOUT_SEC'))
if (-not $GLOBAL_TIMEOUT_SEC -or $GLOBAL_TIMEOUT_SEC -le 0) { $GLOBAL_TIMEOUT_SEC = 600 }
$__verifyStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
function Assert-GlobalTimeout {
  if ($__verifyStopwatch.Elapsed.TotalSeconds -ge $GLOBAL_TIMEOUT_SEC) {
    Write-Host ("Global timeout reached ({0}s). Aborting verify." -f $GLOBAL_TIMEOUT_SEC) -ForegroundColor Red
    try { if ($proc -and $proc.Id) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } } catch {}
    exit 124
  }
}

# 1) Ensure embedding server on :8090 (prefer REAL server, not mocks)
Write-Host '[1/6] Ensuring embedding server on :8090'
$embedUrl = $env:EMBEDDING_SERVER_URL
if (-not $embedUrl) { $embedUrl = 'http://127.0.0.1:8090' }
try { $tcp = Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue } catch { $tcp = $null }
if (-not $tcp) {
  # Try to start real embedding server if embedding_server.py exists and model path is provided
  if (Test-Path 'embedding_server.py') {
    if (-not $env:EMBEDDING_MODEL_PATH) {
      throw 'EMBEDDING_MODEL_PATH not set and embedding server is not running.'
    }
    if (-not (Test-Path '.\.venv')) { python -m venv .\.venv | Out-Null }
    . .\.venv\Scripts\Activate.ps1
    python -m pip install --quiet aiohttp aiohttp_cors numpy | Out-Null
    $env:EMBEDDING_SERVER_PORT = '8090'
    Start-Process -FilePath python -ArgumentList 'embedding_server.py' -NoNewWindow
    Start-Sleep -Seconds 2
  } else {
    throw 'embedding_server.py not found and :8090 not listening. Please start real embedding server or set EMBEDDING_SERVER_URL.'
  }
}
Assert-GlobalTimeout
Assert-GlobalTimeout
try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 ($embedUrl.TrimEnd('/') + '/health')).Content | Write-Output } catch { throw 'Embedding server not responding' }

# 2) Ensure neo4j-test container
Write-Host '[2/6] Ensuring neo4j-test container'
$neo = docker ps --format '{{.Names}}' | Select-String -SimpleMatch 'neo4j-test'
if (-not $neo) {
  docker rm -f neo4j-test 2>$null | Out-Null
  docker run -d --name neo4j-test -p 7475:7474 -p 7688:7687 -e NEO4J_AUTH=neo4j/testpass neo4j:5-community | Out-Null
  Start-Sleep -Seconds 3
}

# 3) Build service (release)
Write-Host '[3/6] Building (release)'
Assert-GlobalTimeout
cargo build --release | Out-Null

# 4) Start memory-server
Write-Host '[4/6] Starting memory-server'
Get-Process | Where-Object { $_.Path -like '*target*release*memory-server*' } | ForEach-Object { $_.Kill() } | Out-Null

$env:RUST_LOG = 'info'
$env:EMBEDDING_SERVER_URL = 'http://127.0.0.1:8090'
$env:NEO4J_URI = 'bolt://localhost:7688'
$env:NEO4J_USER = 'neo4j'
$env:NEO4J_PASSWORD = 'testpass'
$env:ORCHESTRATOR_FORCE_DISABLE = 'true'
$env:DISABLE_SCHEDULERS = 'true'
$env:SERVICE_HOST = '0.0.0.0'
$env:SERVICE_PORT = '8080'

$proc = Start-Process -FilePath '.\target\release\memory-server.exe' -NoNewWindow -PassThru
Write-Host ("PID={0}" -f $proc.Id)

# 5) Wait for health up to 30s
$ready = $false
for ($i=0; $i -lt 30; $i++) {
  Assert-GlobalTimeout
  try {
    $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://127.0.0.1:8080/health'
    $ready = $true
    break
  } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
  Write-Host 'Server did not become ready in time.' -ForegroundColor Red
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  exit 2
}

# 6) API checks
Write-Host 'HEALTH:' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/health').Content | Write-Output

Write-Host 'STORE:' -ForegroundColor Yellow
Assert-GlobalTimeout
$store = Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/memory' -Method Post -ContentType 'application/json' -Body '{"content":"Deterministic test memory FINAL","context_hint":"tests/deterministic"}'
$store.Content | Write-Output

Write-Host 'SEARCH (primary):' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/search' -Method Post -ContentType 'application/json' -Body '{"query":"Deterministic test memory FINAL","limit":5}').Content | Write-Output

Write-Host 'SEARCH (compat):' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/memories/search' -Method Post -ContentType 'application/json' -Body '{"query":"Deterministic test memory FINAL","limit":5}').Content | Write-Output

Write-Host 'RECENT:' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/memories/recent?limit=5').Content | Write-Output

Write-Host 'CONTEXTS:' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/contexts').Content | Write-Output

Write-Host 'STATS:' -ForegroundColor Yellow
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/stats').Content | Write-Output

# 7) Synthetic test: 50 similar fragments -> consolidate -> tick -> stats should show fewer active
Write-Host '[7/6] Synthetic test (50 items + consolidate + tick)' -ForegroundColor Cyan

# Seed 50 similar items
for ($i=1; $i -le 50; $i++) {
  if (($i % 10) -eq 0) { Assert-GlobalTimeout }
  $body = @{ content = "Synthetic repeated content number $i about deterministic consolidation test"; context_hint = "tests/synthetic" } | ConvertTo-Json -Compress
  $null = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080/memory' -Method Post -ContentType 'application/json' -Body $body
}

# Stats before
Assert-GlobalTimeout
$statsBefore = Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/stats' | Select-Object -ExpandProperty Content | ConvertFrom-Json
$before = [int]$statsBefore.statistics.active_memories
Write-Host ("Active before: {0}" -f $before)

# Consolidate
$consBody = @{ context = 'tests/synthetic'; similarity_threshold = 0.92; max_items = 120 } | ConvertTo-Json -Compress
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/maintenance/consolidate' -Method Post -ContentType 'application/json' -Body $consBody).Content | Write-Output

# Tick (virtual days)
$tickBody = @{ ticks = 5 } | ConvertTo-Json -Compress
Assert-GlobalTimeout
(Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/maintenance/tick' -Method Post -ContentType 'application/json' -Body $tickBody).Content | Write-Output

# Stats after
Assert-GlobalTimeout
$statsAfter = Invoke-WebRequest -UseBasicParsing -TimeoutSec 8 'http://127.0.0.1:8080/stats' | Select-Object -ExpandProperty Content | ConvertFrom-Json
$after = [int]$statsAfter.statistics.active_memories
Write-Host ("Active after:  {0}" -f $after)

if ($after -lt $before) {
  Write-Host 'OK: Active memories decreased after consolidate+tick' -ForegroundColor Green
} else {
  Write-Host ("ERROR: Active memories did not decrease (before={0}, after={1})" -f $before, $after) -ForegroundColor Red
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  exit 3
}

try {
  $MIN_P5 = if ($env:MIN_P5) { [double]$env:MIN_P5 } else { 0.95 }
  $MIN_MRR = if ($env:MIN_MRR) { [double]$env:MIN_MRR } else { 0.95 }
  $MIN_NDCG = if ($env:MIN_NDCG) { [double]$env:MIN_NDCG } else { 0.95 }
  Write-Host "[7.1/6] Quality evaluation with strict gates (P5=$MIN_P5, MRR=$MIN_MRR, nDCG=$MIN_NDCG)" -ForegroundColor Yellow
  $args = @(
    'scripts/quality_eval.py', '--host','127.0.0.1','--port','8080','--k','5',
    '--dataset','datasets/quality/dataset.json','--relevance','content',
    '--out','reports/quality_report.json', '--min-p5',"$MIN_P5",'--min-mrr',"$MIN_MRR",'--min-ndcg',"$MIN_NDCG"
  )
  $p = Start-Process -FilePath python -ArgumentList $args -NoNewWindow -PassThru
  $p.WaitForExit()
  if ($p.ExitCode -ne 0) {
    Write-Host "ERROR: Quality gates failed" -ForegroundColor Red
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    exit 4
  }
} catch {
  Write-Host "Quality eval failed: $($_.Exception.Message)" -ForegroundColor Red
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  exit 4
}

Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
Write-Host '== Verification completed ==' -ForegroundColor Cyan
