# AI Memory Service - Quick Start Script for Windows
# This script helps you quickly start the AI Memory Service with all dependencies

# Set strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Colors for output
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Write-Error-Message {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

# ASCII Art Banner
Write-Host @"

╔══════════════════════════════════════════════════════════════╗
║                    AI MEMORY SERVICE                         ║
║              Intelligent Memory System for AI                ║
║                  Powered by GPT-5-nano                       ║
╚══════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Check prerequisites
Write-Info "Checking prerequisites..."

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Info "Docker is installed: $dockerVersion"
} catch {
    Write-Error-Message "Docker is not installed. Please install Docker Desktop for Windows."
    exit 1
}

# Check if Docker is running
try {
    docker ps | Out-Null
    Write-Info "Docker is running"
} catch {
    Write-Error-Message "Docker is not running. Please start Docker Desktop."
    exit 1
}

# Check if Docker Compose is installed
try {
    $composeVersion = docker-compose --version
    Write-Info "Docker Compose is installed: $composeVersion"
} catch {
    Write-Error-Message "Docker Compose is not installed. It should come with Docker Desktop."
    exit 1
}

# Check if Rust is installed (for local development)
try {
    $cargoVersion = cargo --version
    Write-Info "Rust is installed: $cargoVersion"
} catch {
    Write-Warn "Rust is not installed. You'll need it for local development."
    Write-Warn "Install from: https://rustup.rs/"
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Info "Python is installed: $pythonVersion"
} catch {
    Write-Warn "Python is not installed. You'll need it for the embedding service."
    Write-Warn "Install from: https://www.python.org/downloads/"
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Warn ".env file not found. Creating from .env.example..."
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Info "Created .env file. Please edit it with your configuration:"
        Write-Info "  - Set OPENAI_API_KEY for GPT-5-nano"
        Write-Info "  - Set NEO4J_PASSWORD for database"
        Write-Info "  - Adjust other settings as needed"
        Write-Host ""
        Read-Host "Press Enter after you've configured .env file"
    } else {
        Write-Error-Message ".env.example not found. Cannot create .env file."
        exit 1
    }
}

# Load environment variables
Write-Info "Loading environment variables..."
$envContent = Get-Content ".env" | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '=' }
foreach ($line in $envContent) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

# Validate critical environment variables
$openaiKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
if ([string]::IsNullOrEmpty($openaiKey) -or $openaiKey -eq "sk-your-openai-api-key-here") {
    Write-Error-Message "OPENAI_API_KEY is not set or still has the default value."
    Write-Error-Message "Please edit .env file and set your OpenAI API key."
    exit 1
}

if (-not ($openaiKey -match "^sk-")) {
    Write-Warn "OPENAI_API_KEY might be invalid (should start with 'sk-')."
}

# Create necessary directories
Write-Info "Creating necessary directories..."
$directories = @(
    "data",
    "logs",
    "models",
    "cache",
    "config",
    "monitoring\prometheus",
    "monitoring\grafana\dashboards",
    "monitoring\grafana\datasources"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$Url,
        [string]$ServiceName,
        [int]$MaxAttempts = 30
    )
    
    $attempt = 0
    while ($attempt -lt $MaxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Info "$ServiceName is running!"
                return $true
            }
        } catch {
            # Service not ready yet
        }
        $attempt++
        Start-Sleep -Seconds 1
    }
    Write-Warn "$ServiceName is still starting up..."
    return $false
}

# Show menu
Write-Host ""
Write-Host "Select run mode:" -ForegroundColor Cyan
Write-Host "1) Docker Compose (Recommended for production)"
Write-Host "2) Local Development (Rust + Python)"
Write-Host "3) Docker Compose with Monitoring (Prometheus + Grafana)"
Write-Host "4) Exit"
$choice = Read-Host "Enter your choice (1-4)"

switch ($choice) {
    "1" {
        Write-Info "Starting services with Docker Compose..."
        
        # Build images if needed
        Write-Info "Building Docker images..."
        docker-compose build
        
        # Start services
        Write-Info "Starting services..."
        docker-compose up -d
        
        # Wait for services to be healthy
        Write-Info "Waiting for services to be healthy..."
        Start-Sleep -Seconds 10
        
        # Check service health
        Test-ServiceHealth -Url "http://localhost:8080/health" -ServiceName "Memory Service"
        Test-ServiceHealth -Url "http://localhost:5000/health" -ServiceName "Embedding Service"
        
        Write-Info "Services started successfully!"
        Write-Info "  - REST API: http://localhost:8080"
        Write-Info "  - Embedding Service: http://localhost:5000"
        Write-Info "  - Neo4j Browser: http://localhost:7474"
        Write-Host ""
        Write-Info "To view logs: docker-compose logs -f"
        Write-Info "To stop services: docker-compose down"
    }
    
    "2" {
        Write-Info "Starting in local development mode..."
        
        # Check if Python embedding service is running
        $embeddingRunning = Test-ServiceHealth -Url "http://localhost:5000/health" -ServiceName "Embedding Service" -MaxAttempts 1
        
        if (-not $embeddingRunning) {
            Write-Info "Starting Python embedding service..."
            $embeddingProcess = Start-Process python -ArgumentList "embedding_service.py" -PassThru -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        
        # Build Rust project
        Write-Info "Building Rust project..."
        cargo build --release
        
        # Start main service
        Write-Info "Starting AI Memory Service..."
        try {
            cargo run --release --bin memory-server
        } finally {
            # Cleanup on exit
            if ($embeddingProcess) {
                Write-Info "Stopping embedding service..."
                Stop-Process -Id $embeddingProcess.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    
    "3" {
        Write-Info "Starting services with monitoring..."
        
        # Build images
        Write-Info "Building Docker images..."
        docker-compose build
        
        # Start all services including monitoring
        Write-Info "Starting services with monitoring stack..."
        docker-compose --profile monitoring up -d
        
        # Wait for services
        Write-Info "Waiting for services to be healthy..."
        Start-Sleep -Seconds 15
        
        # Check service health
        Test-ServiceHealth -Url "http://localhost:8080/health" -ServiceName "Memory Service"
        Test-ServiceHealth -Url "http://localhost:5000/health" -ServiceName "Embedding Service"
        Test-ServiceHealth -Url "http://localhost:9091/-/ready" -ServiceName "Prometheus"
        Test-ServiceHealth -Url "http://localhost:3000/api/health" -ServiceName "Grafana"
        
        Write-Info "Services started successfully!"
        Write-Info "  - REST API: http://localhost:8080"
        Write-Info "  - Embedding Service: http://localhost:5000"
        Write-Info "  - Neo4j Browser: http://localhost:7474"
        Write-Info "  - Prometheus: http://localhost:9091"
        Write-Info "  - Grafana: http://localhost:3000 (admin/admin)"
        Write-Host ""
        Write-Info "To view logs: docker-compose logs -f"
        Write-Info "To stop services: docker-compose --profile monitoring down"
    }
    
    "4" {
        Write-Info "Exiting..."
        exit 0
    }
    
    default {
        Write-Error-Message "Invalid choice. Exiting."
        exit 1
    }
}

Write-Host ""
Write-Info "Quick test command:"
Write-Host @'
Invoke-RestMethod -Method Post -Uri "http://localhost:8080/api/v1/memory" `
    -ContentType "application/json" `
    -Body '{"content": "Test memory", "context_hint": "testing"}'
'@ -ForegroundColor Gray