# PowerShell script for complete Python DLL system configuration
# Requires Administrator privileges

param(
    [string]$PythonPath = "C:\Users\1\AppData\Local\Programs\Python\Python313",
    [switch]$Force
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python DLL System Configuration Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "⚠ WARNING: Not running as Administrator" -ForegroundColor Yellow
    Write-Host "Some operations may require admin privileges" -ForegroundColor Yellow
    Write-Host ""
}

# 1. Verify Python installation
Write-Host "1. Verifying Python installation..." -ForegroundColor Yellow
if (-not (Test-Path $PythonPath)) {
    Write-Host "❌ Python not found at: $PythonPath" -ForegroundColor Red
    exit 1
}

$pythonExe = Join-Path $PythonPath "python.exe"
$pythonVersion = & $pythonExe --version 2>&1
Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green

# 2. Locate Python DLLs
Write-Host ""
Write-Host "2. Locating Python DLLs..." -ForegroundColor Yellow
$requiredDlls = @(
    "python313.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll"
)

$missingDlls = @()
foreach ($dll in $requiredDlls) {
    $dllPath = Join-Path $PythonPath $dll
    if (Test-Path $dllPath) {
        Write-Host "  ✓ Found: $dll" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $dll" -ForegroundColor Red
        $missingDlls += $dll
    }
}

if ($missingDlls.Count -gt 0) {
    Write-Host "❌ Missing required DLLs. Please reinstall Python." -ForegroundColor Red
    exit 1
}

# 3. Configure System Environment Variables
Write-Host ""
Write-Host "3. Configuring System Environment Variables..." -ForegroundColor Yellow

# Get current PATH
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
$systemPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")

# Check if Python paths are already in PATH
$pythonInUserPath = $userPath -like "*$PythonPath*"
$pythonInSystemPath = $systemPath -like "*$PythonPath*"

if (-not $pythonInUserPath -and -not $pythonInSystemPath) {
    Write-Host "  Adding Python to PATH..." -ForegroundColor Cyan
    
    # Add to User PATH
    $newUserPath = "$PythonPath;$PythonPath\Scripts;$userPath"
    [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
    Write-Host "  ✓ Added to User PATH" -ForegroundColor Green
} else {
    Write-Host "  ✓ Python already in PATH" -ForegroundColor Green
}

# Set PYTHONHOME
$currentPythonHome = [Environment]::GetEnvironmentVariable("PYTHONHOME", "User")
if ($currentPythonHome -ne $PythonPath) {
    [Environment]::SetEnvironmentVariable("PYTHONHOME", $PythonPath, "User")
    Write-Host "  ✓ Set PYTHONHOME=$PythonPath" -ForegroundColor Green
} else {
    Write-Host "  ✓ PYTHONHOME already set" -ForegroundColor Green
}

# Set PYO3_PYTHON for Rust builds
$pyo3Python = Join-Path $PythonPath "python.exe"
[Environment]::SetEnvironmentVariable("PYO3_PYTHON", $pyo3Python, "User")
Write-Host "  ✓ Set PYO3_PYTHON=$pyo3Python" -ForegroundColor Green

# 4. Register DLLs in System32 (requires admin)
Write-Host ""
Write-Host "4. System DLL Registration..." -ForegroundColor Yellow

if ($isAdmin) {
    $system32Path = "C:\Windows\System32"
    
    foreach ($dll in @("python313.dll", "python3.dll")) {
        $sourcePath = Join-Path $PythonPath $dll
        $targetPath = Join-Path $system32Path $dll
        
        if (-not (Test-Path $targetPath) -or $Force) {
            try {
                Copy-Item -Path $sourcePath -Destination $targetPath -Force
                Write-Host "  ✓ Copied $dll to System32" -ForegroundColor Green
            } catch {
                Write-Host "  ✗ Failed to copy $dll`: $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host "  ✓ $dll already in System32" -ForegroundColor Green
        }
    }
} else {
    Write-Host "  ⚠ Skipping (requires Administrator)" -ForegroundColor Yellow
    Write-Host "  Run this script as Administrator to register DLLs in System32" -ForegroundColor Yellow
}

# 5. Create symbolic links (alternative to System32)
Write-Host ""
Write-Host "5. Creating Symbolic Links..." -ForegroundColor Yellow

$linkPath = "C:\PythonDLL"
if (-not (Test-Path $linkPath)) {
    if ($isAdmin) {
        New-Item -ItemType Directory -Path $linkPath -Force | Out-Null
        Write-Host "  ✓ Created directory: $linkPath" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Cannot create $linkPath (requires Administrator)" -ForegroundColor Yellow
    }
}

if (Test-Path $linkPath) {
    foreach ($dll in $requiredDlls) {
        $sourcePath = Join-Path $PythonPath $dll
        $targetPath = Join-Path $linkPath $dll
        
        if (-not (Test-Path $targetPath)) {
            try {
                if ($isAdmin) {
                    New-Item -ItemType SymbolicLink -Path $targetPath -Target $sourcePath -Force | Out-Null
                    Write-Host "  ✓ Created symlink for $dll" -ForegroundColor Green
                } else {
                    Copy-Item -Path $sourcePath -Destination $targetPath -Force
                    Write-Host "  ✓ Copied $dll to $linkPath" -ForegroundColor Green
                }
            } catch {
                Write-Host "  ✗ Failed to create link for $dll`: $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host "  ✓ $dll already linked" -ForegroundColor Green
        }
    }
    
    # Add to PATH if not present
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    if ($currentPath -notlike "*$linkPath*") {
        [Environment]::SetEnvironmentVariable("PATH", "$linkPath;$currentPath", "User")
        Write-Host "  ✓ Added $linkPath to PATH" -ForegroundColor Green
    }
}

# 6. Verify DLL visibility
Write-Host ""
Write-Host "6. Verifying DLL Visibility..." -ForegroundColor Yellow

# Refresh environment
$env:Path = [Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [Environment]::GetEnvironmentVariable("PATH", "Machine")
$env:PYTHONHOME = [Environment]::GetEnvironmentVariable("PYTHONHOME", "User")

foreach ($dll in @("python313.dll", "python3.dll")) {
    $found = $false
    
    # Check multiple locations
    $searchPaths = @(
        $PythonPath,
        "C:\Windows\System32",
        "C:\PythonDLL",
        $env:TEMP
    )
    
    foreach ($searchPath in $searchPaths) {
        $dllPath = Join-Path $searchPath $dll
        if (Test-Path $dllPath) {
            $found = $true
            break
        }
    }
    
    if ($found) {
        Write-Host "  ✓ $dll is accessible" -ForegroundColor Green
    } else {
        # Try where command
        $whereResult = where.exe $dll 2>$null
        if ($whereResult) {
            Write-Host "  ✓ $dll found via PATH: $whereResult" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ $dll not found in PATH" -ForegroundColor Yellow
        }
    }
}

# 7. Test Python embedding
Write-Host ""
Write-Host "7. Testing Python Embedding..." -ForegroundColor Yellow

$testCode = @'
import sys
print(f"Python {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Path: {sys.path[:2]}")
'@

try {
    $result = & $pythonExe -c $testCode 2>&1
    Write-Host "  ✓ Python embedding test successful:" -ForegroundColor Green
    $result | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} catch {
    Write-Host "  ✗ Python embedding test failed: $_" -ForegroundColor Red
}

# 8. Generate diagnostics report
Write-Host ""
Write-Host "8. Generating Diagnostics Report..." -ForegroundColor Yellow

$reportPath = "python_dll_diagnostics.txt"
$report = @"
Python DLL Configuration Diagnostics
Generated: $(Get-Date)
=====================================

Python Installation:
  Path: $PythonPath
  Version: $pythonVersion
  Executable: $pythonExe

Environment Variables:
  PYTHONHOME: $env:PYTHONHOME
  PYO3_PYTHON: $(Get-Item Env:PYO3_PYTHON -ErrorAction SilentlyContinue)
  PATH (Python entries):
$(($env:Path -split ';' | Where-Object { $_ -like "*Python*" }) -join "`n    ")

DLL Locations:
"@

foreach ($dll in $requiredDlls) {
    $locations = @()
    $searchPaths = @($PythonPath, "C:\Windows\System32", "C:\PythonDLL")
    foreach ($path in $searchPaths) {
        if (Test-Path (Join-Path $path $dll)) {
            $locations += $path
        }
    }
    $report += "`n  ${dll}: $($locations -join ', ')"
}

$report | Out-File -FilePath $reportPath -Encoding UTF8
Write-Host "  ✓ Report saved to: $reportPath" -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Configuration Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($isAdmin) {
    Write-Host "✅ Full configuration completed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Partial configuration completed." -ForegroundColor Yellow
    Write-Host "   Run as Administrator for full system integration." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Close and reopen your terminal to apply changes" -ForegroundColor White
Write-Host "2. Run 'cargo clean' to clear build cache" -ForegroundColor White
Write-Host "3. Run 'cargo build --release' to rebuild with new configuration" -ForegroundColor White
Write-Host "4. Test with 'cargo test --release'" -ForegroundColor White

# Check if we need to restart the shell
$needsRestart = $false
$currentPath = $env:Path
$newPath = [Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [Environment]::GetEnvironmentVariable("PATH", "Machine")
if ($currentPath -ne $newPath) {
    $needsRestart = $true
}

if ($needsRestart) {
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Terminal restart required for PATH changes!" -ForegroundColor Yellow
    Write-Host "   Close this terminal and open a new one." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Run with -Force to overwrite existing configurations" -ForegroundColor Gray
Write-Host "Example: .\setup_python_dll.ps1 -Force" -ForegroundColor Gray