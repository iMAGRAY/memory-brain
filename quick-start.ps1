# AI Memory Service - Quick Start PowerShell Script
# Self-contained deployment requiring only Docker

param(
    [switch]$Monitoring,
    [switch]$NoTest,
    [switch]$Help
)

# Color output functions
function Write-Log { 
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Blue
}

function Write-Success { 
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning { 
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error { 
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Show help
if ($Help) {
    Write-Host "AI Memory Service Quick Start" -ForegroundColor Cyan
    Write-Host "Usage: .\quick-start.ps1 [options]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -Monitoring    Enable monitoring services automatically" -ForegroundColor Gray
    Write-Host "  -NoTest       Skip service connectivity tests" -ForegroundColor Gray
    Write-Host "  -Help         Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "This script sets up a complete self-contained AI Memory Service" -ForegroundColor White
    Write-Host "requiring only Docker to be installed on your system." -ForegroundColor White
    exit 0
}

# Configuration
$ProjectDir = $PSScriptRoot
$EnvFile = Join-Path $ProjectDir ".env"
$EnvExample = Join-Path $ProjectDir ".env.example"

Write-Host "🚀 AI Memory Service - Quick Start" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Self-contained deployment with everything included!" -ForegroundColor White
Write-Host ""

# Check system requirements
function Test-Requirements {
    Write-Log "🔍 Checking system requirements..."
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        if ($LASTEXITCODE -ne 0) { throw "Docker not found" }
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop."
        Write-Host "Visit: https://docs.docker.com/desktop/install/windows/" -ForegroundColor Yellow
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker not running" }
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose --version
        if ($LASTEXITCODE -ne 0) { 
            docker compose version | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Docker Compose not found" }
        }
    }
    catch {
        Write-Error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    }
    
    # Check available memory
    $memory = Get-WmiObject -Class Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
    $memoryGB = [math]::Round($memory / 1GB, 1)
    if ($memoryGB -lt 4) {
        Write-Warning "System has less than 4GB RAM ($memoryGB GB). Performance may be limited."
    }
    
    Write-Success "✅ System requirements satisfied"
}

# Generate secure environment configuration
function New-SecureEnvironment {
    Write-Log "🔐 Generating secure environment configuration..."
    
    if (Test-Path $EnvFile) {
        Write-Warning ".env file already exists"
        $response = Read-Host "Do you want to regenerate credentials? [y/N]"
        if ($response -notmatch '^[Yy]$') {
            Write-Log "Using existing .env file"
            return
        }
    }
    
    if (-not (Test-Path $EnvExample)) {
        Write-Error ".env.example file not found"
        exit 1
    }
    
    # Generate secure passwords using .NET crypto
    function New-SecurePassword {
        param([int]$Length = 24)
        $chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        $password = ""
        $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
        $bytes = New-Object byte[] 1
        
        for ($i = 0; $i -lt $Length; $i++) {
            $rng.GetBytes($bytes)
            $password += $chars[$bytes[0] % $chars.Length]
        }
        $rng.Dispose()
        return $password
    }
    
    function New-ApiKey {
        $bytes = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
        $rng.GetBytes($bytes)
        $apiKey = [System.BitConverter]::ToString($bytes) -replace '-', ''
        $rng.Dispose()
        return $apiKey.ToLower()
    }
    
    # Generate credentials
    $neo4jPassword = New-SecurePassword -Length 24
    $redisPassword = New-SecurePassword -Length 20
    $grafanaPassword = New-SecurePassword -Length 16
    $adminApiKey = New-ApiKey
    
    # Create .env file with replacements
    $envContent = Get-Content $EnvExample -Raw
    $envContent = $envContent -replace 'NEO4J_PASSWORD_PLACEHOLDER', $neo4jPassword
    $envContent = $envContent -replace 'REDIS_PASSWORD_PLACEHOLDER', $redisPassword
    $envContent = $envContent -replace 'GRAFANA_PASSWORD_PLACEHOLDER', $grafanaPassword
    $envContent = $envContent -replace 'ADMIN_API_KEY_PLACEHOLDER', $adminApiKey
    
    $envContent | Out-File -FilePath $EnvFile -Encoding UTF8
    
    # Set restrictive permissions (Windows equivalent)
    $acl = Get-Acl $EnvFile
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($env:USERNAME, "FullControl", "Allow")))
    Set-Acl -Path $EnvFile -AclObject $acl
    
    Write-Success "✅ Secure credentials generated"
    
    Write-Host ""
    Write-Host "🔐 Generated Credentials (SAVE THESE SECURELY):" -ForegroundColor Cyan
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "Neo4j Password: $neo4jPassword" -ForegroundColor White
    Write-Host "Redis Password: $redisPassword" -ForegroundColor White
    Write-Host "Grafana Password: $grafanaPassword" -ForegroundColor White
    Write-Host "Admin API Key: $adminApiKey" -ForegroundColor White
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host "⚠️  These credentials are saved in .env file" -ForegroundColor Yellow
    Write-Host ""
}

# Build and start services
function Start-Services {
    Write-Log "🚀 Building and starting AI Memory Service..."
    
    Set-Location $ProjectDir
    
    # Validate docker-compose configuration
    try {
        docker-compose config | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Invalid configuration" }
    }
    catch {
        Write-Error "Docker Compose configuration is invalid"
        docker-compose config
        exit 1
    }
    
    # Build images
    Write-Log "📦 Building container images (this may take 10-15 minutes)..."
    docker-compose build --parallel
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to build container images"
        exit 1
    }
    
    # Start core services
    Write-Log "🎯 Starting core services..."
    docker-compose up -d neo4j python-embeddings ai-memory-service
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ Failed to start services"
        exit 1
    }
    
    # Wait for services to be healthy
    Write-Log "⏳ Waiting for services to be ready..."
    $timeout = 600  # 10 minutes
    $elapsed = 0
    
    while ($elapsed -lt $timeout) {
        try {
            $psOutput = docker-compose ps --format json | ConvertFrom-Json
            $healthyCount = ($psOutput | Where-Object { $_.Health -eq "healthy" }).Count
            
            if ($healthyCount -ge 3) {
                Write-Success "✅ All core services are healthy!"
                break
            }
            
            Start-Sleep -Seconds 15
            $elapsed += 15
            Write-Log "Waiting... ($elapsed/$timeout s) - Healthy services: $healthyCount/3"
        }
        catch {
            Start-Sleep -Seconds 15
            $elapsed += 15
            Write-Log "Waiting... ($elapsed/$timeout s) - Checking services..."
        }
    }
    
    if ($elapsed -ge $timeout) {
        Write-Error "❌ Services failed to start within timeout"
        Write-Log "Service status:"
        docker-compose ps
        Write-Log "Logs:"
        docker-compose logs --tail=20
        exit 1
    }
}

# Test service connectivity
function Test-Services {
    if ($NoTest) { return }
    
    Write-Log "🧪 Testing service connectivity..."
    
    # Test main API
    try {
        Invoke-WebRequest -Uri "http://localhost:8080/health" -Method GET -TimeoutSec 5 | Out-Null
        Write-Success "✅ Memory API: Healthy"
    }
    catch {
        Write-Warning "⚠️  Memory API: Not responding"
    }
    
    # Test Python embeddings
    try {
        Invoke-WebRequest -Uri "http://localhost:8001/health" -Method GET -TimeoutSec 5 | Out-Null
        Write-Success "✅ Python Embeddings: Healthy"
    }
    catch {
        Write-Warning "⚠️  Python Embeddings: Not responding"
    }
    
    # Test Neo4j
    try {
        Invoke-WebRequest -Uri "http://localhost:7474" -Method GET -TimeoutSec 5 | Out-Null
        Write-Success "✅ Neo4j Browser: Accessible"
    }
    catch {
        Write-Warning "⚠️  Neo4j Browser: Not accessible"
    }
}

# Show service information
function Show-ServiceInfo {
    Write-Log "📋 Service Information"
    Write-Host "======================" -ForegroundColor White
    
    # Load environment for port info
    if (Test-Path $EnvFile) {
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match '^([^#=]+)=(.*)$') {
                Set-Variable -Name $matches[1] -Value $matches[2] -Scope Global -ErrorAction SilentlyContinue
            }
        }
    }
    
    Write-Host "🌐 Access Points:" -ForegroundColor Cyan
    Write-Host "  • Memory API:         http://localhost:$($API_PORT ?? 8080)" -ForegroundColor White
    Write-Host "  • Admin Panel:        http://localhost:$($ADMIN_PORT ?? 8081)" -ForegroundColor White
    Write-Host "  • Python Embeddings:  http://localhost:8001" -ForegroundColor White
    Write-Host "  • Neo4j Browser:      http://localhost:$($NEO4J_HTTP_PORT ?? 7474)" -ForegroundColor White
    Write-Host ""
    Write-Host "🧪 Quick API Test:" -ForegroundColor Cyan
    Write-Host "  Invoke-WebRequest -Uri http://localhost:8080/health" -ForegroundColor Gray
    Write-Host ""
    Write-Host "🛠️  Management Commands:" -ForegroundColor Cyan
    Write-Host "  • View logs:       docker-compose logs -f" -ForegroundColor Gray
    Write-Host "  • Stop services:   docker-compose down" -ForegroundColor Gray
    Write-Host "  • Start monitoring: docker-compose --profile monitoring up -d" -ForegroundColor Gray
    Write-Host "  • Update services: docker-compose pull; docker-compose up -d" -ForegroundColor Gray
    Write-Host ""
    Write-Host "📊 Current Status:" -ForegroundColor Cyan
    docker-compose ps
    Write-Host ""
}

# Optional monitoring services
function Enable-Monitoring {
    if (-not $Monitoring) {
        $response = Read-Host "Do you want to enable monitoring (Prometheus + Grafana)? [y/N]"
        if ($response -notmatch '^[Yy]$') { return }
    }
    
    Write-Log "📊 Starting monitoring services..."
    docker-compose --profile monitoring up -d
    
    Write-Host "📊 Monitoring Access:" -ForegroundColor Cyan
    Write-Host "  • Grafana Dashboard: http://localhost:$($GRAFANA_PORT ?? 3000)" -ForegroundColor White
    Write-Host "  • Prometheus:        http://localhost:$($PROMETHEUS_PORT ?? 9090)" -ForegroundColor White
    Write-Host "  • Health Monitor:    docker-compose logs -f healthcheck" -ForegroundColor White
}

# Cleanup function
function Stop-OnError {
    Write-Log "🧹 Cleaning up due to error..."
    try { docker-compose down } catch { }
}

# Main execution
try {
    # Run setup steps
    Test-Requirements
    New-SecureEnvironment
    Start-Services
    Test-Services
    Show-ServiceInfo
    Enable-Monitoring
    
    Write-Success "🎉 AI Memory Service is running!"
    
    Write-Host ""
    Write-Host "🔗 Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Visit http://localhost:8080/health to verify the service" -ForegroundColor White
    Write-Host "  2. Check the API documentation in docs/api.md" -ForegroundColor White
    Write-Host "  3. Try the example requests in examples/" -ForegroundColor White
    Write-Host ""
    Write-Host "📖 For troubleshooting, see DEPLOYMENT.md" -ForegroundColor White
}
catch {
    Write-Error "❌ Setup failed: $($_.Exception.Message)"
    Stop-OnError
    exit 1
}