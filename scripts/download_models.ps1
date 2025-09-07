# Download Models Script for AI Memory Service
# Downloads EmbeddingGemma-300m ONNX model and tokenizer from Hugging Face

param(
    [switch]$Force = $false,
    [string]$ModelsDir = "models",
    [switch]$Verify = $true
)

$ErrorActionPreference = "Stop"

# Model information
$REPO_ID = "onnx-community/embeddinggemma-300m-ONNX"
$BASE_URL = "https://huggingface.co/$REPO_ID/resolve/main"

$FILES = @(
    @{
        Name = "gemma-300m.onnx"
        Url = "$BASE_URL/onnx/model.onnx"
        Size = 620MB
        Description = "EmbeddingGemma-300m ONNX Model"
    },
    @{
        Name = "gemma-300m.onnx_data"
        Url = "$BASE_URL/onnx/model.onnx_data"
        Size = 600MB
        Description = "EmbeddingGemma-300m ONNX Weights"
    },
    @{
        Name = "tokenizer.json"  
        Url = "$BASE_URL/tokenizer.json"
        Size = 2.1MB
        Description = "Tokenizer Configuration"
    }
)

Write-Host "AI Memory Service - Model Download Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Create models directory
if (!(Test-Path $ModelsDir)) {
    Write-Host "Creating models directory: $ModelsDir" -ForegroundColor Green
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

# Check if Git LFS is available for large file handling
$gitLfsAvailable = $false
try {
    git lfs version | Out-Null
    $gitLfsAvailable = $true
    Write-Host "Git LFS detected - using for large file downloads" -ForegroundColor Yellow
} catch {
    Write-Host "Git LFS not available - using direct download" -ForegroundColor Yellow
}

# Download each file
foreach ($file in $FILES) {
    $filePath = Join-Path $ModelsDir $file.Name
    $fileExists = Test-Path $filePath
    
    Write-Host ""
    Write-Host "Processing: $($file.Description)" -ForegroundColor Cyan
    Write-Host "  File: $($file.Name)"
    Write-Host "  Size: $($file.Size)"
    
    if ($fileExists -and !$Force) {
        $existingSize = (Get-Item $filePath).Length
        Write-Host "  Status: Already exists ($([math]::Round($existingSize/1MB, 1))MB)" -ForegroundColor Green
        
        if ($Verify) {
            Write-Host "  Verifying file integrity..." -ForegroundColor Yellow
            # Basic size check - in production, use proper checksums
            if ($existingSize -lt 1MB) {
                Write-Host "  Warning: File appears incomplete, re-downloading..." -ForegroundColor Red
                $fileExists = $false
            } else {
                Write-Host "  Verification: OK" -ForegroundColor Green
            }
        }
    }
    
    if (!$fileExists -or $Force) {
        Write-Host "  Status: Downloading..." -ForegroundColor Yellow
        
        try {
            # Use Invoke-WebRequest with progress
            $ProgressPreference = 'Continue'
            Invoke-WebRequest -Uri $file.Url -OutFile $filePath -UseBasicParsing
            
            $downloadedSize = (Get-Item $filePath).Length
            Write-Host "  Completed: $([math]::Round($downloadedSize/1MB, 1))MB downloaded" -ForegroundColor Green
            
        } catch {
            Write-Host "  Error: Failed to download - $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Download Summary:" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan

$totalSize = 0
foreach ($file in $FILES) {
    $filePath = Join-Path $ModelsDir $file.Name
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length
        $totalSize += $size
        Write-Host "  ✓ $($file.Name) - $([math]::Round($size/1MB, 1))MB" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($file.Name) - Missing" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Total model size: $([math]::Round($totalSize/1MB, 1))MB" -ForegroundColor Cyan

# Verify configuration file
$configPath = "config.toml"
if (Test-Path $configPath) {
    Write-Host ""
    Write-Host "Verifying configuration..." -ForegroundColor Cyan
    
    $configContent = Get-Content $configPath -Raw
    $modelPathCorrect = $configContent -match 'model_path\s*=\s*"\.\/models\/gemma-300m\.onnx"'
    $tokenizerPathCorrect = $configContent -match 'tokenizer_path\s*=\s*"\.\/models\/tokenizer\.json"'
    
    if ($modelPathCorrect -and $tokenizerPathCorrect) {
        Write-Host "  Configuration: OK" -ForegroundColor Green
    } else {
        Write-Host "  Configuration: Needs update" -ForegroundColor Yellow
        Write-Host "  Please ensure config.toml has correct model paths:"
        Write-Host "    model_path = `"./models/gemma-300m.onnx`""
        Write-Host "    tokenizer_path = `"./models/tokenizer.json`""
    }
} else {
    Write-Host ""
    Write-Host "Warning: config.toml not found" -ForegroundColor Yellow
    Write-Host "Please create configuration file with model paths"
}

Write-Host ""
Write-Host "Model download completed successfully! 🎉" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Verify config.toml has correct model paths"
Write-Host "2. Start Neo4j database: docker run -d --name neo4j-memory -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.0"  
Write-Host "3. Run the service: cargo run --release"
Write-Host ""