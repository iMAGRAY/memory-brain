# PowerShell script to download and install ONNX Runtime 1.22.x for Windows x64
# Compatible with ort crate 2.0.0-rc.10

param(
    [string]$Version = "1.22.1",
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "================================" -ForegroundColor Cyan
Write-Host "ONNX Runtime Installer v1.0" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$baseUrl = "https://github.com/microsoft/onnxruntime/releases/download"
$fileName = "onnxruntime-win-x64-$Version.zip"
$downloadUrl = "$baseUrl/v$Version/$fileName"
$tempDir = ".\temp_onnxrt"
$extractPath = "$tempDir\extracted"
$targetDir = ".\onnxruntime\lib"
$backupDir = ".\onnxruntime\lib_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host "[INFO] Target ONNX Runtime version: $Version" -ForegroundColor Green
Write-Host "[INFO] Download URL: $downloadUrl" -ForegroundColor Gray
Write-Host ""

# Check if backup needed
if (Test-Path $targetDir) {
    $existingDll = Join-Path $targetDir "onnxruntime.dll"
    if (Test-Path $existingDll) {
        Write-Host "[WARNING] Existing ONNX Runtime found in $targetDir" -ForegroundColor Yellow
        
        if (-not $Force) {
            $response = Read-Host "Do you want to backup existing files? (Y/N)"
            if ($response -ne 'Y' -and $response -ne 'y') {
                Write-Host "[INFO] Installation cancelled by user" -ForegroundColor Red
                exit 0
            }
        }
        
        Write-Host "[INFO] Creating backup at: $backupDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        Copy-Item -Path "$targetDir\*" -Destination $backupDir -Recurse
        Write-Host "[SUCCESS] Backup created" -ForegroundColor Green
    }
}

# Create temp directory
Write-Host "[INFO] Creating temporary directory..." -ForegroundColor Gray
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
New-Item -ItemType Directory -Path $extractPath -Force | Out-Null

# Download
try {
    Write-Host "[INFO] Downloading ONNX Runtime $Version..." -ForegroundColor Cyan
    $downloadPath = Join-Path $tempDir $fileName
    
    # Use progress bar
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $downloadPath -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    # Verify download
    if (-not (Test-Path $downloadPath)) {
        throw "Download failed: file not found"
    }
    
    $fileSize = (Get-Item $downloadPath).Length / 1MB
    Write-Host "[SUCCESS] Downloaded: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    
} catch {
    Write-Host "[ERROR] Failed to download: $_" -ForegroundColor Red
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

# Extract
try {
    Write-Host "[INFO] Extracting archive..." -ForegroundColor Cyan
    Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    
    # Find the lib directory
    $libDir = Get-ChildItem -Path $extractPath -Filter "lib" -Recurse -Directory | Select-Object -First 1
    
    if (-not $libDir) {
        throw "Could not find 'lib' directory in extracted files"
    }
    
    Write-Host "[SUCCESS] Extraction complete" -ForegroundColor Green
    
} catch {
    Write-Host "[ERROR] Failed to extract: $_" -ForegroundColor Red
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 1
}

# Install DLLs
try {
    Write-Host "[INFO] Installing ONNX Runtime DLLs..." -ForegroundColor Cyan
    
    # Create target directory if it doesn't exist
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    
    # Copy all DLL files
    $dllFiles = Get-ChildItem -Path $libDir.FullName -Filter "*.dll"
    
    if ($dllFiles.Count -eq 0) {
        throw "No DLL files found in lib directory"
    }
    
    foreach ($dll in $dllFiles) {
        $targetPath = Join-Path $targetDir $dll.Name
        Copy-Item -Path $dll.FullName -Destination $targetPath -Force
        Write-Host "  + Installed: $($dll.Name)" -ForegroundColor Gray
    }
    
    Write-Host "[SUCCESS] Installation complete!" -ForegroundColor Green
    
} catch {
    Write-Host "[ERROR] Failed to install DLLs: $_" -ForegroundColor Red
    
    # Restore backup if available
    if (Test-Path $backupDir) {
        Write-Host "[INFO] Restoring backup..." -ForegroundColor Yellow
        Copy-Item -Path "$backupDir\*" -Destination $targetDir -Force
        Write-Host "[SUCCESS] Backup restored" -ForegroundColor Green
    }
    
    exit 1
    
} finally {
    # Cleanup
    Write-Host "[INFO] Cleaning up temporary files..." -ForegroundColor Gray
    Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
}

# Verify installation
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$installedDlls = Get-ChildItem -Path $targetDir -Filter "*.dll"
Write-Host "[INFO] Installed DLLs in $targetDir`:" -ForegroundColor Green
foreach ($dll in $installedDlls) {
    $size = [math]::Round($dll.Length / 1KB, 2)
    Write-Host "  - $($dll.Name) ($size KB)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[SUCCESS] ONNX Runtime $Version is ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run 'cargo clean' to clear old build artifacts" -ForegroundColor Gray
Write-Host "2. Run 'cargo build --release' to rebuild with new ONNX Runtime" -ForegroundColor Gray
Write-Host "3. Run 'cargo test' to verify everything works" -ForegroundColor Gray
Write-Host ""

if (Test-Path $backupDir) {
    Write-Host "[NOTE] Old version backed up to: $backupDir" -ForegroundColor Yellow
    Write-Host "       You can delete it once everything is verified to work" -ForegroundColor Gray
}