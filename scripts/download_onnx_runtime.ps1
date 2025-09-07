# Download ONNX Runtime for Windows
$onnxVersion = "1.19.2"
$platform = "win-x64"
$downloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$onnxVersion/onnxruntime-$platform-$onnxVersion.zip"
$targetDir = "$PSScriptRoot\..\onnxruntime"
$zipFile = "$targetDir\onnxruntime.zip"

Write-Host "Downloading ONNX Runtime v$onnxVersion for Windows x64..." -ForegroundColor Cyan

# Create target directory
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
}

# Download ONNX Runtime
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    Write-Host "Download completed!" -ForegroundColor Green
} catch {
    Write-Host "Error downloading ONNX Runtime: $_" -ForegroundColor Red
    exit 1
}

# Extract the archive
Write-Host "Extracting ONNX Runtime..." -ForegroundColor Cyan
Expand-Archive -Path $zipFile -DestinationPath $targetDir -Force

# Move files to correct location
$extractedDir = Get-ChildItem -Path $targetDir -Directory | Where-Object { $_.Name -like "onnxruntime-*" } | Select-Object -First 1
if ($extractedDir) {
    # Copy DLLs to lib directory
    $libDir = "$targetDir\lib"
    if (-not (Test-Path $libDir)) {
        New-Item -ItemType Directory -Path $libDir | Out-Null
    }
    
    Copy-Item -Path "$($extractedDir.FullName)\lib\*.dll" -Destination $libDir -Force
    Write-Host "ONNX Runtime DLLs copied to: $libDir" -ForegroundColor Green
    
    # Clean up
    Remove-Item -Path $zipFile -Force
    Remove-Item -Path $extractedDir.FullName -Recurse -Force
    
    Write-Host "ONNX Runtime setup completed successfully!" -ForegroundColor Green
    Write-Host "DLL location: $libDir\onnxruntime.dll" -ForegroundColor Yellow
} else {
    Write-Host "Error: Could not find extracted ONNX Runtime directory" -ForegroundColor Red
    exit 1
}