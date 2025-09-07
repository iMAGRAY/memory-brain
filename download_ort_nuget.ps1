# Download ONNX Runtime 1.22.x from NuGet
$version = "1.19.2"  # Using compatible version
$packageName = "Microsoft.ML.OnnxRuntime"
$nugetUrl = "https://www.nuget.org/api/v2/package/$packageName/$version"
$outputPath = "onnxruntime-nuget.zip"
$extractPath = "onnxruntime-nuget"
$targetDir = ".\onnxruntime\lib"

Write-Host "Downloading $packageName version $version from NuGet..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $nugetUrl -OutFile $outputPath -UseBasicParsing

Write-Host "Extracting package..." -ForegroundColor Cyan
Expand-Archive -Path $outputPath -DestinationPath $extractPath -Force

Write-Host "Looking for native DLLs..." -ForegroundColor Cyan
$nativePath = Get-ChildItem -Path $extractPath -Filter "native" -Recurse -Directory | Select-Object -First 1

if ($nativePath) {
    $dllPath = Join-Path $nativePath.FullName "win-x64"
    if (Test-Path $dllPath) {
        Write-Host "Found DLLs at: $dllPath" -ForegroundColor Green
        
        # Create backup
        $backupDir = "$targetDir`_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        if (Test-Path $targetDir) {
            Copy-Item -Path $targetDir -Destination $backupDir -Recurse
            Write-Host "Backup created at: $backupDir" -ForegroundColor Yellow
        }
        
        # Copy DLLs
        if (-not (Test-Path $targetDir)) {
            New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
        }
        
        Get-ChildItem -Path $dllPath -Filter "*.dll" | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $targetDir -Force
            Write-Host "  + Installed: $($_.Name)" -ForegroundColor Gray
        }
        
        Write-Host "Installation complete!" -ForegroundColor Green
    }
} else {
    Write-Host "Could not find native DLLs in package" -ForegroundColor Red
}

# Cleanup
Remove-Item -Path $outputPath -Force -ErrorAction SilentlyContinue
Remove-Item -Path $extractPath -Recurse -Force -ErrorAction SilentlyContinue