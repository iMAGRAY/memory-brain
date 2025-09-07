# Simple script to check PATH duplicates
$path = [Environment]::GetEnvironmentVariable("PATH", "Process")
$entries = $path -split ';'

Write-Host "Total PATH entries: $($entries.Count)"
Write-Host ""

# Count occurrences
$counts = @{}
foreach ($entry in $entries) {
    if ($entry) {
        $key = $entry.TrimEnd('\').ToLower()
        if ($counts.ContainsKey($key)) {
            $counts[$key]++
        } else {
            $counts[$key] = 1
        }
    }
}

# Show duplicates
$hasDuplicates = $false
Write-Host "Duplicate entries found:"
foreach ($key in $counts.Keys | Sort-Object) {
    if ($counts[$key] -gt 1) {
        Write-Host "  $($counts[$key])x: $key"
        $hasDuplicates = $true
    }
}

if (-not $hasDuplicates) {
    Write-Host "  No duplicates found"
}

Write-Host ""
Write-Host "Python check:"
& where python 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Python not found in PATH"
}