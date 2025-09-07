# AI Memory Service - Automated Testing Script for Windows
# Runs all tests and generates report

param(
    [switch]$Quick,        # Run only unit tests
    [switch]$Integration,  # Run only integration tests
    [switch]$Benchmark,    # Run benchmarks
    [switch]$Coverage,     # Generate code coverage
    [switch]$Verbose       # Verbose output
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($text, $color) {
    Write-Host $text -ForegroundColor $color
}

Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "AI Memory Service - Test Runner" "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

# Check prerequisites
function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." "Yellow"
    
    # Check Rust installation
    if (!(Get-Command cargo -ErrorAction SilentlyContinue)) {
        Write-ColorOutput "ERROR: Cargo not found. Please install Rust." "Red"
        exit 1
    }
    
    # Check Docker for integration tests
    if ($Integration -or (!$Quick)) {
        if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
            Write-ColorOutput "WARNING: Docker not found. Integration tests will be skipped." "Yellow"
            $script:SkipIntegration = $true
        } else {
            # Check if Neo4j is running
            $neo4jRunning = docker ps --format "table {{.Names}}" | Select-String "neo4j"
            if (!$neo4jRunning) {
                Write-ColorOutput "Starting Neo4j for integration tests..." "Yellow"
                docker-compose up -d neo4j
                Start-Sleep -Seconds 10
            }
        }
    }
    
    # Check ONNX Runtime
    $ortPath = Join-Path $PSScriptRoot "..\onnxruntime\lib\onnxruntime.dll"
    if (!(Test-Path $ortPath)) {
        Write-ColorOutput "WARNING: ONNX Runtime not found. Running download script..." "Yellow"
        & "$PSScriptRoot\download_onnx_runtime.ps1"
    }
    
    # Check model files
    $modelPath = Join-Path $PSScriptRoot "..\models\embeddinggemma-300m-ONNX\model.onnx"
    if (!(Test-Path $modelPath)) {
        Write-ColorOutput "WARNING: Model files not found. Some tests may fail." "Yellow"
    }
    
    Write-ColorOutput "Prerequisites check completed!" "Green"
    Write-Host ""
}

# Set environment variables
function Set-TestEnvironment {
    Write-ColorOutput "Setting up test environment..." "Yellow"
    
    $env:RUST_BACKTRACE = "1"
    $env:ORT_DYLIB_PATH = Join-Path $PSScriptRoot "..\onnxruntime\lib\onnxruntime.dll"
    
    # Set test database credentials if not already set
    if (!$env:NEO4J_TEST_USER) {
        $env:NEO4J_TEST_USER = "neo4j"
    }
    if (!$env:NEO4J_TEST_PASSWORD) {
        $env:NEO4J_TEST_PASSWORD = "test_password"
    }
    if (!$env:NEO4J_TEST_URI) {
        $env:NEO4J_TEST_URI = "bolt://localhost:7687"
    }
    
    # Set test model paths
    $env:TEST_MODEL_PATH = Join-Path $PSScriptRoot "..\models\embeddinggemma-300m-ONNX\model.onnx"
    $env:TEST_TOKENIZER_PATH = Join-Path $PSScriptRoot "..\models\embeddinggemma-300m-ONNX\tokenizer.json"
    
    if ($Verbose) {
        $env:RUST_LOG = "debug"
    } else {
        $env:RUST_LOG = "warn"
    }
    
    Write-ColorOutput "Environment configured!" "Green"
    Write-Host ""
}

# Run unit tests
function Run-UnitTests {
    Write-ColorOutput "Running unit tests..." "Cyan"
    
    $testArgs = @("test", "--lib")
    if ($Verbose) {
        $testArgs += "--", "--nocapture"
    }
    
    cargo $testArgs
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Unit tests failed!" "Red"
        return $false
    }
    
    Write-ColorOutput "Unit tests passed!" "Green"
    return $true
}

# Run integration tests
function Run-IntegrationTests {
    if ($script:SkipIntegration) {
        Write-ColorOutput "Skipping integration tests (Docker not available)" "Yellow"
        return $true
    }
    
    Write-ColorOutput "Running integration tests..." "Cyan"
    
    # Run API integration tests
    Write-ColorOutput "Testing API endpoints..." "Yellow"
    cargo test --test api_integration_test
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "API integration tests failed!" "Red"
        return $false
    }
    
    # Run Neo4j integration tests
    Write-ColorOutput "Testing Neo4j operations..." "Yellow"
    cargo test --test neo4j_integration_test
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Neo4j integration tests failed!" "Red"
        return $false
    }
    
    # Run general integration tests
    Write-ColorOutput "Testing memory operations..." "Yellow"
    cargo test --test integration_test
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Memory integration tests failed!" "Red"
        return $false
    }
    
    Write-ColorOutput "Integration tests passed!" "Green"
    return $true
}

# Run benchmarks
function Run-Benchmarks {
    Write-ColorOutput "Running benchmarks..." "Cyan"
    
    cargo bench --no-fail-fast
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Benchmarks failed!" "Red"
        return $false
    }
    
    Write-ColorOutput "Benchmarks completed!" "Green"
    return $true
}

# Generate code coverage
function Generate-Coverage {
    Write-ColorOutput "Generating code coverage..." "Cyan"
    
    # Check if tarpaulin is installed
    if (!(cargo install --list | Select-String "cargo-tarpaulin")) {
        Write-ColorOutput "Installing cargo-tarpaulin..." "Yellow"
        cargo install cargo-tarpaulin
    }
    
    cargo tarpaulin --out Html --output-dir ./target/coverage
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "Coverage report generated at: target/coverage/tarpaulin-report.html" "Green"
        return $true
    } else {
        Write-ColorOutput "Coverage generation failed!" "Red"
        return $false
    }
}

# Generate test report
function Generate-Report {
    param($results)
    
    Write-Host ""
    Write-ColorOutput "========================================" "Cyan"
    Write-ColorOutput "Test Results Summary" "Cyan"
    Write-ColorOutput "========================================" "Cyan"
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "Timestamp: $timestamp"
    Write-Host ""
    
    foreach ($key in $results.Keys) {
        $status = if ($results[$key]) { "PASSED" } else { "FAILED" }
        $color = if ($results[$key]) { "Green" } else { "Red" }
        Write-ColorOutput "$key : $status" $color
    }
    
    # Save report to file
    $reportPath = Join-Path $PSScriptRoot "..\target\test_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    $reportContent = @"
AI Memory Service Test Report
Generated: $timestamp

Test Results:
$(foreach ($key in $results.Keys) {
    "$key : $(if ($results[$key]) { 'PASSED' } else { 'FAILED' })`n"
})

Environment:
- Rust Version: $(rustc --version)
- OS: Windows
- ONNX Runtime: $env:ORT_DYLIB_PATH
- Neo4j URI: $env:NEO4J_TEST_URI
"@
    
    $reportContent | Out-File -FilePath $reportPath
    Write-Host ""
    Write-ColorOutput "Report saved to: $reportPath" "Yellow"
}

# Main execution
function Main {
    $startTime = Get-Date
    $results = @{}
    
    # Check prerequisites
    Test-Prerequisites
    
    # Set up environment
    Set-TestEnvironment
    
    # Run tests based on flags
    if ($Quick) {
        # Quick mode - only unit tests
        $results["Unit Tests"] = Run-UnitTests
    } elseif ($Integration) {
        # Only integration tests
        $results["Integration Tests"] = Run-IntegrationTests
    } else {
        # Full test suite
        $results["Unit Tests"] = Run-UnitTests
        
        if (!$script:SkipIntegration) {
            $results["Integration Tests"] = Run-IntegrationTests
        }
        
        if ($Benchmark) {
            $results["Benchmarks"] = Run-Benchmarks
        }
    }
    
    # Generate coverage if requested
    if ($Coverage) {
        $results["Code Coverage"] = Generate-Coverage
    }
    
    # Generate report
    Generate-Report $results
    
    # Calculate execution time
    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Host ""
    Write-ColorOutput "Total execution time: $($duration.TotalSeconds) seconds" "Cyan"
    
    # Exit with appropriate code
    $failed = $results.Values | Where-Object { $_ -eq $false }
    if ($failed) {
        Write-ColorOutput "Some tests failed. Please review the results above." "Red"
        exit 1
    } else {
        Write-ColorOutput "All tests passed successfully!" "Green"
        exit 0
    }
}

# Run main function
Main