# PowerShell script to remove PATH duplicates safely
# This script will:
# 1. Read current PATH
# 2. Remove duplicates while preserving order
# 3. Add Python if not present
# 4. Update the PATH

Write-Host "PATH Duplicate Cleaner" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "Process")
$entries = $currentPath -split ';'

Write-Host "Current PATH has $($entries.Count) entries" -ForegroundColor Yellow
Write-Host ""

# Create a hashtable to track unique entries (case-insensitive, normalized)
$uniqueEntries = [System.Collections.ArrayList]::new()
$seenPaths = @{}

foreach ($entry in $entries) {
    if ($entry -and $entry.Trim()) {
        # Normalize the path (remove trailing backslash, convert to lowercase for comparison)
        $normalizedKey = $entry.TrimEnd('\').ToLower()
        
        # If we haven't seen this path before, add it
        if (-not $seenPaths.ContainsKey($normalizedKey)) {
            $seenPaths[$normalizedKey] = $true
            [void]$uniqueEntries.Add($entry)
        }
    }
}

Write-Host "After removing duplicates: $($uniqueEntries.Count) unique entries" -ForegroundColor Green
Write-Host ""

# Check if Python is in PATH
$pythonPath = "C:\Users\1\AppData\Local\Programs\Python\Python313"
$pythonScriptsPath = "C:\Users\1\AppData\Local\Programs\Python\Python313\Scripts"

$hasPython = $false
$hasPythonScripts = $false

foreach ($entry in $uniqueEntries) {
    $normalized = $entry.TrimEnd('\').ToLower()
    if ($normalized -eq $pythonPath.ToLower()) {
        $hasPython = $true
    }
    if ($normalized -eq $pythonScriptsPath.ToLower()) {
        $hasPythonScripts = $true
    }
}

# Add Python paths if not present
if (-not $hasPython) {
    Write-Host "Adding Python to PATH: $pythonPath" -ForegroundColor Green
    [void]$uniqueEntries.Add($pythonPath)
} else {
    Write-Host "Python already in PATH" -ForegroundColor Cyan
}

if (-not $hasPythonScripts) {
    Write-Host "Adding Python Scripts to PATH: $pythonScriptsPath" -ForegroundColor Green
    [void]$uniqueEntries.Add($pythonScriptsPath)
} else {
    Write-Host "Python Scripts already in PATH" -ForegroundColor Cyan
}

# Build the new PATH string
$newPath = $uniqueEntries -join ';'

Write-Host ""
Write-Host "Statistics:" -ForegroundColor Yellow
Write-Host "  Original entries: $($entries.Count)"
Write-Host "  Unique entries: $($uniqueEntries.Count)"
Write-Host "  Duplicates removed: $($entries.Count - $uniqueEntries.Count)"
Write-Host ""

# Update the PATH for the current process
[Environment]::SetEnvironmentVariable("PATH", $newPath, "Process")
Write-Host "✓ PATH updated for current process" -ForegroundColor Green

# Verify Python is now accessible
Write-Host ""
Write-Host "Verifying Python installation:" -ForegroundColor Yellow
& where python 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python is now accessible in PATH" -ForegroundColor Green
    
    # Show Python version
    & python --version
} else {
    Write-Host "⚠ Python still not found. You may need to restart your terminal." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To make changes permanent, run this script with admin privileges and add:" -ForegroundColor Cyan
Write-Host '[Environment]::SetEnvironmentVariable("PATH", $newPath, "User")' -ForegroundColor White