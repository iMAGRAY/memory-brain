# PowerShell script to download ONNX Runtime 1.22.x for Windows x64

$version = "1.19.2"  # Using 1.19.2 to match current DLLs for now
$baseUrl = "https://github.com/microsoft/onnxruntime/releases/download"
$fileName = "onnxruntime-win-x64-$version.zip"
$downloadUrl = "$baseUrl/v$version/$fileName"
$outputPath = "onnxruntime-download.zip"
$extractPath = "onnxruntime-new"

Write-Host "Downloading ONNX Runtime $version..."
Write-Host "URL: $downloadUrl"

# Download the file
try {
    Invoke-WebRequest -Uri $downloadUrl -OutFile $outputPath
    Write-Host "Download completed: $outputPath"
} catch {
    Write-Host "Error downloading file: $_"
    exit 1
}

# Extract the archive
Write-Host "Extracting archive..."
Expand-Archive -Path $outputPath -DestinationPath $extractPath -Force

# Show extracted files
Write-Host "Extracted files:"
Get-ChildItem -Path $extractPath -Recurse | Where-Object { $_.Extension -eq ".dll" }

Write-Host "`nTo update ONNX Runtime:"
Write-Host "1. Stop any running processes using the DLLs"
Write-Host "2. Copy DLLs from $extractPath\onnxruntime-win-x64-$version\lib to .\onnxruntime\lib\"
Write-Host "3. Run tests again"