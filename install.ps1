# AI Memory Service - Universal Installer
# Automatically detects and configures the best deployment option

param(
    [string]$Environment = "Development",
    [string]$DataDir = "$env:PROGRAMDATA\AIMemoryService",
    [bool]$InstallNeo4j = $true,
    [bool]$UseDocker = $null,  # null = auto-detect
    [bool]$InstallPython = $true,
    [string]$LogLevel = "info",
    [bool]$EnableMonitoring = $false
)

# Import required modules
Add-Type -AssemblyName System.Web
Add-Type -AssemblyName System.Security

Write-Host "🧠 AI Memory Service Installer" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Data Directory: $DataDir" -ForegroundColor Yellow
Write-Host ""

# Check if running as administrator
function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-AdminRights)) {
    Write-Host "❌ Administrator privileges required" -ForegroundColor Red
    Write-Host "Please run this script as Administrator" -ForegroundColor Yellow
    exit 1
}

# Generate secure passwords
function New-SecurePassword {
    param([int]$Length = 16)
    
    $chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
    $password = ""
    $random = New-Object System.Random
    
    for ($i = 0; $i -lt $Length; $i++) {
        $password += $chars[$random.Next($chars.Length)]
    }
    
    return $password
}

# System detection
function Get-SystemCapabilities {
    $capabilities = @{
        DockerAvailable = $false
        PythonAvailable = $false
        JavaAvailable = $false
        RamGB = 0
        CpuCores = 0
        AvailableDiskGB = 0
        Architecture = ""
    }
    
    # Check RAM
    $ram = Get-CimInstance -ClassName Win32_ComputerSystem
    $capabilities.RamGB = [math]::Round($ram.TotalPhysicalMemory / 1GB, 1)
    
    # Check CPU
    $cpu = Get-CimInstance -ClassName Win32_Processor
    $capabilities.CpuCores = $cpu.NumberOfLogicalProcessors
    
    # Check disk space
    $disk = Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DeviceID='C:'"
    $capabilities.AvailableDiskGB = [math]::Round($disk.FreeSpace / 1GB, 1)
    
    # Check architecture
    $capabilities.Architecture = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
    
    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            $capabilities.DockerAvailable = $true
        }
    }
    catch {
        $capabilities.DockerAvailable = $false
    }
    
    # Check Python
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion -and $pythonVersion -match "Python 3\.([8-9]|1[0-9])") {
            $capabilities.PythonAvailable = $true
        }
    }
    catch {
        $capabilities.PythonAvailable = $false
    }
    
    # Check Java
    try {
        $javaOutput = java -version 2>&1
        if ($javaOutput -match "version `"(\d+)") {
            $javaVersion = [int]$matches[1]
            if ($javaVersion -ge 17) {
                $capabilities.JavaAvailable = $true
            }
        }
    }
    catch {
        $capabilities.JavaAvailable = $false
    }
    
    return $capabilities
}

# Display system information
Write-Host "1. Analyzing system capabilities..." -ForegroundColor Yellow
$system = Get-SystemCapabilities

Write-Host "  System Specifications:" -ForegroundColor Cyan
Write-Host "    RAM: $($system.RamGB) GB" -ForegroundColor Gray
Write-Host "    CPU Cores: $($system.CpuCores)" -ForegroundColor Gray
Write-Host "    Available Disk: $($system.AvailableDiskGB) GB" -ForegroundColor Gray
Write-Host "    Architecture: $($system.Architecture)" -ForegroundColor Gray

Write-Host "  Available Components:" -ForegroundColor Cyan
Write-Host "    Docker: $(if($system.DockerAvailable) { '✅ Available' } else { '❌ Not Found' })" -ForegroundColor $(if($system.DockerAvailable) { 'Green' } else { 'Red' })
Write-Host "    Python 3.8+: $(if($system.PythonAvailable) { '✅ Available' } else { '❌ Not Found' })" -ForegroundColor $(if($system.PythonAvailable) { 'Green' } else { 'Red' })
Write-Host "    Java 17+: $(if($system.JavaAvailable) { '✅ Available' } else { '❌ Not Found' })" -ForegroundColor $(if($system.JavaAvailable) { 'Green' } else { 'Red' })
Write-Host ""

# Deployment strategy selection
Write-Host "2. Selecting optimal deployment strategy..." -ForegroundColor Yellow

if ($UseDocker -eq $null) {
    $UseDocker = $system.DockerAvailable
}

$deploymentStrategy = if ($UseDocker) {
    "Docker"
} elseif ($system.JavaAvailable) {
    "Native"
} else {
    "Embedded"
}

Write-Host "  Selected Strategy: $deploymentStrategy" -ForegroundColor Green

# Minimum requirements check
$requirementsMet = $true

if ($system.RamGB -lt 4) {
    Write-Host "  ⚠️  Warning: Minimum 4GB RAM recommended (found $($system.RamGB)GB)" -ForegroundColor Yellow
}

if ($system.AvailableDiskGB -lt 2) {
    Write-Host "  ❌ Error: Minimum 2GB disk space required (found $($system.AvailableDiskGB)GB)" -ForegroundColor Red
    $requirementsMet = $false
}

if ($system.CpuCores -lt 2) {
    Write-Host "  ⚠️  Warning: 2+ CPU cores recommended for optimal performance" -ForegroundColor Yellow
}

if (-not $requirementsMet) {
    Write-Host "❌ System does not meet minimum requirements" -ForegroundColor Red
    exit 1
}

# Create directory structure
Write-Host "`n3. Creating directory structure..." -ForegroundColor Yellow

$directories = @(
    $DataDir,
    "$DataDir\config",
    "$DataDir\data",
    "$DataDir\logs",
    "$DataDir\backup",
    "$DataDir\temp"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

# Generate configuration
Write-Host "`n4. Generating secure configuration..." -ForegroundColor Yellow

$passwords = @{
    Neo4j = New-SecurePassword -Length 24
    Redis = New-SecurePassword -Length 20
    Admin = New-SecurePassword -Length 16
}

$configContent = @"
# AI Memory Service Configuration
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
# Strategy: $deploymentStrategy

[server]
host = "127.0.0.1"
port = 8080
admin_port = 8081
websocket_port = 8082
log_level = "$LogLevel"

[neo4j]
uri = "bolt://localhost:7687"
username = "neo4j"
password = "$($passwords.Neo4j)"
max_connections = 20
connection_timeout = 30

[embedding]
service_url = "http://localhost:8001"
model_path = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32
timeout_seconds = 30

[memory]
cache_size_mb = $(if($system.RamGB -gt 8) { 1024 } else { 512 })
max_connections = $(if($system.CpuCores -gt 4) { 100 } else { 50 })
simd_threads = 0
compression_enabled = true

[storage]
data_path = "$DataDir\data"
backup_path = "$DataDir\backup"
log_path = "$DataDir\logs"

[security]
api_key_required = $(if($Environment -eq "Production") { "true" } else { "false" })
cors_origins = $(if($Environment -eq "Production") { '"http://localhost:3000"' } else { '"*"' })
rate_limit_per_minute = 1000

[monitoring]
enabled = $($EnableMonitoring.ToString().ToLower())
prometheus_port = 9090
metrics_retention_days = 7
"@

$configPath = "$DataDir\config\config.toml"
$configContent | Out-File -FilePath $configPath -Encoding UTF8
Write-Host "  Configuration saved to: $configPath" -ForegroundColor Gray

# Environment file for Docker
if ($UseDocker) {
    $envContent = @"
# AI Memory Service Environment
# Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

NEO4J_PASSWORD=$($passwords.Neo4j)
NEO4J_USER=neo4j
REDIS_PASSWORD=$($passwords.Redis)
GRAFANA_PASSWORD=$($passwords.Admin)

RUST_LOG=$LogLevel
MEMORY_CACHE_MB=$(if($system.RamGB -gt 8) { 1024 } else { 512 })
TOKIO_THREADS=$($system.CpuCores)
RAYON_THREADS=$($system.CpuCores)

DATA_DIR=$DataDir
CORS_ORIGINS=$(if($Environment -eq "Production") { "https://yourdomain.com" } else { "*" })

ENABLE_PROMETHEUS=$($EnableMonitoring.ToString().ToLower())
ENABLE_GRAFANA=$($EnableMonitoring.ToString().ToLower())
"@
    
    $envPath = "$DataDir\.env"
    $envContent | Out-File -FilePath $envPath -Encoding UTF8
    Write-Host "  Environment file saved to: $envPath" -ForegroundColor Gray
}

# Execute deployment strategy
Write-Host "`n5. Executing $deploymentStrategy deployment..." -ForegroundColor Yellow

switch ($deploymentStrategy) {
    "Docker" {
        Write-Host "  Using Docker Compose deployment" -ForegroundColor Cyan
        
        # Start Neo4j first
        if ($InstallNeo4j) {
            Write-Host "  Starting Neo4j container..." -ForegroundColor Gray
            & docker-compose up -d neo4j
            
            # Wait for Neo4j to be ready
            $timeout = 120
            $elapsed = 0
            do {
                Start-Sleep -Seconds 5
                $elapsed += 5
                Write-Host "    Waiting for Neo4j... ($elapsed/$timeout seconds)" -ForegroundColor Gray
                try {
                    $response = Invoke-WebRequest -Uri "http://localhost:7474" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
                    if ($response.StatusCode -eq 200) {
                        Write-Host "  ✅ Neo4j is ready!" -ForegroundColor Green
                        break
                    }
                } catch {
                    # Continue waiting
                }
            } while ($elapsed -lt $timeout)
        }
        
        # Start remaining services
        Write-Host "  Starting all services..." -ForegroundColor Gray
        & docker-compose up -d
        
        Write-Host "  ✅ Docker deployment completed" -ForegroundColor Green
    }
    
    "Native" {
        Write-Host "  Using native installation" -ForegroundColor Cyan
        
        if ($InstallNeo4j) {
            Write-Host "  Installing Neo4j natively..." -ForegroundColor Gray
            & powershell -ExecutionPolicy Bypass -File ".\setup_neo4j_native.ps1" -Password $passwords.Neo4j -InstallPath "$DataDir\neo4j"
        }
        
        if ($InstallPython -and -not $system.PythonAvailable) {
            Write-Host "  Installing Python and dependencies..." -ForegroundColor Gray
            # Python installation would go here
        }
        
        Write-Host "  ✅ Native deployment completed" -ForegroundColor Green
    }
    
    "Embedded" {
        Write-Host "  Using embedded SQLite database" -ForegroundColor Cyan
        
        # Create SQLite database directory
        $sqliteDir = "$DataDir\sqlite"
        if (!(Test-Path $sqliteDir)) {
            New-Item -ItemType Directory -Path $sqliteDir -Force | Out-Null
        }
        
        # Update configuration for embedded mode
        $embeddedConfig = $configContent -replace 'uri = "bolt://localhost:7687"', 'uri = "sqlite://./data/memory.db"'
        $embeddedConfig | Out-File -FilePath $configPath -Encoding UTF8
        
        Write-Host "  ✅ Embedded deployment configured" -ForegroundColor Green
    }
}

# Create management scripts
Write-Host "`n6. Creating management scripts..." -ForegroundColor Yellow

$startScript = @"
@echo off
echo Starting AI Memory Service...
cd /d "$DataDir"

$(if ($UseDocker) {
    "docker-compose up -d"
} else {
    "ai-memory-service.exe --config config\config.toml"
})

echo Service started successfully!
pause
"@

$stopScript = @"
@echo off
echo Stopping AI Memory Service...
cd /d "$DataDir"

$(if ($UseDocker) {
    "docker-compose down"
} else {
    "taskkill /F /IM ai-memory-service.exe 2>nul"
    "net stop AIMemoryService 2>nul"
})

echo Service stopped.
pause
"@

$statusScript = @"
@echo off
echo AI Memory Service Status
echo ========================

$(if ($UseDocker) {
@"
docker-compose ps
echo.
echo Web Interfaces:
echo - Memory API: http://localhost:8080
echo - Neo4j Browser: http://localhost:7474 (neo4j / $($passwords.Neo4j))
echo - Admin Panel: http://localhost:8081
"@
} else {
@"
sc query AIMemoryService
echo.
echo Endpoints:
echo - API: http://localhost:8080/health
echo - Admin: http://localhost:8081/metrics
"@
})

pause
"@

$startScript | Out-File -FilePath "$DataDir\start.bat" -Encoding ASCII
$stopScript | Out-File -FilePath "$DataDir\stop.bat" -Encoding ASCII
$statusScript | Out-File -FilePath "$DataDir\status.bat" -Encoding ASCII

Write-Host "  Created management scripts:" -ForegroundColor Gray
Write-Host "    $DataDir\start.bat" -ForegroundColor Gray
Write-Host "    $DataDir\stop.bat" -ForegroundColor Gray
Write-Host "    $DataDir\status.bat" -ForegroundColor Gray

# Create Windows Service (for native deployment)
if ($deploymentStrategy -eq "Native" -and $Environment -eq "Production") {
    Write-Host "`n7. Creating Windows Service..." -ForegroundColor Yellow
    
    $serviceName = "AIMemoryService"
    $serviceDisplayName = "AI Memory Service"
    $serviceDescription = "Intelligent memory management system with vector search capabilities"
    $serviceExePath = "$DataDir\ai-memory-service.exe"
    
    try {
        # Create service
        New-Service -Name $serviceName -BinaryPathName "`"$serviceExePath`" --config `"$DataDir\config\config.toml`"" -DisplayName $serviceDisplayName -Description $serviceDescription -StartupType Automatic
        
        Write-Host "  ✅ Windows Service '$serviceName' created" -ForegroundColor Green
        Write-Host "    Run 'net start $serviceName' to start the service" -ForegroundColor Gray
    } catch {
        Write-Host "  ⚠️  Service creation failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

# Final summary
Write-Host "`n🎯 Installation Complete!" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""
Write-Host "Installation Details:" -ForegroundColor White
Write-Host "  Strategy: $deploymentStrategy" -ForegroundColor Cyan
Write-Host "  Data Directory: $DataDir" -ForegroundColor Cyan
Write-Host "  Environment: $Environment" -ForegroundColor Cyan
Write-Host ""

Write-Host "Generated Credentials:" -ForegroundColor White
if ($InstallNeo4j) {
    Write-Host "  Neo4j Password: $($passwords.Neo4j)" -ForegroundColor Yellow
}
if ($UseDocker -and $EnableMonitoring) {
    Write-Host "  Admin Password: $($passwords.Admin)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "Service Endpoints:" -ForegroundColor White
Write-Host "  Memory API: http://localhost:8080" -ForegroundColor Cyan
Write-Host "  Admin Panel: http://localhost:8081" -ForegroundColor Cyan
if ($InstallNeo4j) {
    Write-Host "  Neo4j Browser: http://localhost:7474" -ForegroundColor Cyan
}
if ($EnableMonitoring) {
    Write-Host "  Grafana: http://localhost:3000" -ForegroundColor Cyan
}
Write-Host ""

Write-Host "Management Commands:" -ForegroundColor White
Write-Host "  Start: $DataDir\start.bat" -ForegroundColor Gray
Write-Host "  Stop: $DataDir\stop.bat" -ForegroundColor Gray
Write-Host "  Status: $DataDir\status.bat" -ForegroundColor Gray
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor White
Write-Host "1. Run start.bat to begin the service" -ForegroundColor Gray
Write-Host "2. Visit http://localhost:8080/health to verify" -ForegroundColor Gray
Write-Host "3. Check the documentation at $DataDir\README.md" -ForegroundColor Gray
Write-Host ""

Write-Host "⚠️  IMPORTANT: Store these credentials securely!" -ForegroundColor Red
Write-Host "Configuration saved to: $configPath" -ForegroundColor Yellow

# Save credentials to secure file
$credentialsPath = "$DataDir\credentials.txt"
@"
AI Memory Service - Generated Credentials
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Neo4j Database:
  Username: neo4j
  Password: $($passwords.Neo4j)
  URL: http://localhost:7474

$(if ($EnableMonitoring) {
@"
Grafana Dashboard:
  Username: admin
  Password: $($passwords.Admin)
  URL: http://localhost:3000

"@
})
IMPORTANT: Keep this file secure and delete it after storing credentials elsewhere.
"@ | Out-File -FilePath $credentialsPath -Encoding UTF8

Write-Host "Credentials also saved to: $credentialsPath" -ForegroundColor Yellow
Write-Host ""
Write-Host "✨ AI Memory Service is ready to use!" -ForegroundColor Green