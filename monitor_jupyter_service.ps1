# JupyterLab Service Monitoring Script
# Version: 2.0.0 - Production-ready with proper exit conditions

[CmdletBinding()]
param(
    [int]$CheckInterval = 60,          # Check interval in seconds
    [int]$MaxChecks = 0,                # Maximum number of checks (0 = until stopped)
    [int]$MaxRuntime = 0,               # Maximum runtime in minutes (0 = unlimited)
    [string]$LogPath = "",              # Log path (empty = auto-detect)
    [string]$JupyterPath = "",          # Jupyter executable path (empty = auto-detect)
    [switch]$AutoRestart = $false,      # Auto-restart failed service
    [switch]$ExitOnFailure = $false     # Exit on service failure
)

# Initialize configuration from environment or defaults
if (-not $LogPath) {
    $LogPath = if ($env:JUPYTER_LOG_DIR) {
        Join-Path $env:JUPYTER_LOG_DIR "monitor.log"
    } else {
        Join-Path $env:TEMP "jupyter_monitor.log"
    }
}

if (-not $JupyterPath) {
    $JupyterPath = if ($env:JUPYTER_PATH) {
        $env:JUPYTER_PATH
    } else {
        # Try to find jupyter in common locations
        $possiblePaths = @(
            "$env:LOCALAPPDATA\Programs\Python\Python313\Scripts\jupyter.exe",
            "$env:APPDATA\Python\Python313\Scripts\jupyter.exe",
            "jupyter.exe"
        )
        $foundPath = $possiblePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($foundPath) { $foundPath } else { "jupyter.exe" }
    }
}

# Global monitoring state
$script:MonitoringActive = $true
$script:StartTime = Get-Date
$script:ChecksDone = 0

# Register cleanup handler for Ctrl+C
$null = Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    $script:MonitoringActive = $false
    Write-Log "Monitoring stopped by user" "INFO"
}

# Function to write timestamped log
function Write-Log {
    param(
        [string]$Message, 
        [string]$Level = "INFO",
        [switch]$NoConsole
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    if (-not $NoConsole) {
        $color = switch($Level) {
            "ERROR" { "Red" }
            "WARNING" { "Yellow" }
            "SUCCESS" { "Green" }
            "DEBUG" { "Gray" }
            default { "White" }
        }
        Write-Host $logEntry -ForegroundColor $color
    }
    
    try {
        # Ensure log directory exists
        $logDir = Split-Path -Parent $LogPath
        if ($logDir -and -not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        
        Add-Content -Path $LogPath -Value $logEntry -ErrorAction Stop
    } catch {
        if (-not $NoConsole) {
            Write-Host "Failed to write to log: $_" -ForegroundColor Red
        }
    }
}

# Function to check if monitoring should continue
function Test-ShouldContinue {
    # Check if monitoring is still active
    if (-not $script:MonitoringActive) {
        Write-Log "Monitoring flag set to false" "DEBUG"
        return $false
    }
    
    # Check max checks limit
    if ($MaxChecks -gt 0 -and $script:ChecksDone -ge $MaxChecks) {
        Write-Log "Maximum check count ($MaxChecks) reached" "INFO"
        return $false
    }
    
    # Check max runtime limit
    if ($MaxRuntime -gt 0) {
        $elapsed = (Get-Date) - $script:StartTime
        if ($elapsed.TotalMinutes -ge $MaxRuntime) {
            Write-Log "Maximum runtime ($MaxRuntime minutes) reached" "INFO"
            return $false
        }
    }
    
    # Check for user interruption
    if ([Console]::KeyAvailable) {
        $key = [Console]::ReadKey($true)
        if ($key.Key -eq [ConsoleKey]::Q -or $key.Key -eq [ConsoleKey]::Escape) {
            Write-Log "Monitoring stopped by user input (Q/ESC)" "INFO"
            return $false
        }
    }
    
    return $true
}

# Function to check service status
function Test-JupyterService {
    param([string]$ServiceName = "JupyterLab")
    
    try {
        $service = Get-Service -Name $ServiceName -ErrorAction Stop
        
        $status = @{
            Exists = $true
            Running = $service.Status -eq 'Running'
            Status = $service.Status
            StartType = $service.StartType
        }
        
        $logLevel = if ($status.Running) { "SUCCESS" } else { "WARNING" }
        Write-Log "Service '$ServiceName': $($service.Status) (StartType: $($service.StartType))" $logLevel
        
        return $status
    } catch {
        Write-Log "Service '$ServiceName' not found: $_" "ERROR"
        return @{ Exists = $false; Running = $false; Status = "NotFound" }
    }
}

# Function to check HTTP endpoint
function Test-JupyterEndpoint {
    param(
        [string]$BaseUrl = "http://127.0.0.1:8888",
        [int]$Timeout = 5
    )
    
    $endpoints = @(
        @{ Path = "/api"; Description = "API endpoint" },
        @{ Path = "/api/status"; Description = "Status endpoint" },
        @{ Path = "/lab"; Description = "Lab interface" }
    )
    
    $results = @{}
    
    foreach ($endpoint in $endpoints) {
        $url = "$BaseUrl$($endpoint.Path)"
        try {
            $response = Invoke-WebRequest -Uri $url -TimeoutSec $Timeout -UseBasicParsing -Method Head -ErrorAction Stop
            $results[$endpoint.Path] = @{
                Success = $true
                StatusCode = $response.StatusCode
                Description = $endpoint.Description
            }
            Write-Log "$($endpoint.Description) accessible at $url" "SUCCESS"
        } catch {
            $results[$endpoint.Path] = @{
                Success = $false
                Error = $_.Exception.Message
                Description = $endpoint.Description
            }
            Write-Log "$($endpoint.Description) failed: $($_.Exception.Message)" "WARNING"
        }
    }
    
    return $results
}

# Function to check available kernels
function Test-JupyterKernels {
    try {
        # Use jupyter command directly
        $kernelOutput = & $JupyterPath kernelspec list 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Failed to list kernels (exit code: $LASTEXITCODE)" "WARNING"
            return @()
        }
        
        $kernels = $kernelOutput | Select-String "^\s+(\w+)" | ForEach-Object {
            $matches = [regex]::Match($_.Line, '^\s+(\S+)\s+(.+)$')
            if ($matches.Success) {
                @{
                    Name = $matches.Groups[1].Value
                    Path = $matches.Groups[2].Value.Trim()
                }
            }
        }
        
        if ($kernels.Count -gt 0) {
            $kernelNames = ($kernels | ForEach-Object { $_.Name }) -join ', '
            Write-Log "Found $($kernels.Count) kernel(s): $kernelNames" "SUCCESS"
        } else {
            Write-Log "No kernels found" "WARNING"
        }
        
        return $kernels
    } catch {
        Write-Log "Failed to enumerate kernels: $_" "ERROR"
        return @()
    }
}

# Function to check system resources
function Test-SystemResources {
    param(
        [int]$MemoryWarningThreshold = 85,
        [int]$CPUWarningThreshold = 80
    )
    
    try {
        # Memory check
        $memory = Get-CimInstance Win32_OperatingSystem
        $memoryUsage = [math]::Round((($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / $memory.TotalVisibleMemorySize) * 100, 2)
        
        # CPU check (average over last few samples)
        $cpuSamples = 1..3 | ForEach-Object {
            (Get-Counter '\Processor(_Total)\% Processor Time' -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            Start-Sleep -Milliseconds 500
        }
        $cpuUsage = [math]::Round(($cpuSamples | Measure-Object -Average).Average, 2)
        
        # Disk check for Jupyter root
        $jupyterRoot = if ($env:JUPYTER_ROOT_DIR) { $env:JUPYTER_ROOT_DIR } else { "C:\" }
        $drive = Get-PSDrive -Name ($jupyterRoot.Substring(0,1)) -ErrorAction SilentlyContinue
        $diskFreeGB = if ($drive) { [math]::Round($drive.Free / 1GB, 2) } else { "Unknown" }
        
        $logLevel = "INFO"
        if ($memoryUsage -gt $MemoryWarningThreshold -or $cpuUsage -gt $CPUWarningThreshold) {
            $logLevel = "WARNING"
        }
        
        Write-Log "Resources - CPU: $cpuUsage%, Memory: $memoryUsage%, Disk Free: ${diskFreeGB}GB" $logLevel
        
        return @{
            CPU = $cpuUsage
            Memory = $memoryUsage
            DiskFreeGB = $diskFreeGB
            Warnings = @(
                if ($memoryUsage -gt $MemoryWarningThreshold) { "High memory usage" }
                if ($cpuUsage -gt $CPUWarningThreshold) { "High CPU usage" }
                if ($diskFreeGB -is [double] -and $diskFreeGB -lt 1) { "Low disk space" }
            ) | Where-Object { $_ }
        }
    } catch {
        Write-Log "Failed to check system resources: $_" "ERROR"
        return @{ CPU = 0; Memory = 0; DiskFreeGB = 0; Warnings = @("Resource check failed") }
    }
}

# Function to handle service recovery
function Invoke-ServiceRecovery {
    param(
        [string]$ServiceName = "JupyterLab",
        [int]$WaitSeconds = 10
    )
    
    if (-not $AutoRestart) {
        Write-Log "Auto-restart is disabled. Manual intervention required." "WARNING"
        return $false
    }
    
    Write-Log "Attempting to restart service '$ServiceName'..." "WARNING"
    
    try {
        # Stop the service if running
        $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
        if ($service -and $service.Status -ne 'Stopped') {
            Stop-Service -Name $ServiceName -Force -ErrorAction Stop
            Write-Log "Service stopped" "INFO"
            Start-Sleep -Seconds 2
        }
        
        # Start the service
        Start-Service -Name $ServiceName -ErrorAction Stop
        Write-Log "Service start command issued" "INFO"
        
        # Wait for service to stabilize
        Start-Sleep -Seconds $WaitSeconds
        
        # Verify service is running
        $service = Get-Service -Name $ServiceName -ErrorAction Stop
        if ($service.Status -eq 'Running') {
            Write-Log "Service successfully restarted" "SUCCESS"
            return $true
        } else {
            Write-Log "Service failed to start properly (Status: $($service.Status))" "ERROR"
            return $false
        }
    } catch {
        Write-Log "Failed to restart service: $_" "ERROR"
        return $false
    }
}

# Main monitoring function
function Start-Monitoring {
    Write-Log "=" * 70
    Write-Log "JupyterLab Service Monitor v2.0.0 Started"
    Write-Log "Configuration:"
    Write-Log "  Check Interval: $CheckInterval seconds" "INFO" 
    Write-Log "  Max Checks: $(if ($MaxChecks -gt 0) { $MaxChecks } else { 'Unlimited' })" "INFO"
    Write-Log "  Max Runtime: $(if ($MaxRuntime -gt 0) { "$MaxRuntime minutes" } else { 'Unlimited' })" "INFO"
    Write-Log "  Auto-Restart: $AutoRestart" "INFO"
    Write-Log "  Exit on Failure: $ExitOnFailure" "INFO"
    Write-Log "  Log Path: $LogPath" "INFO"
    Write-Log "  Jupyter Path: $JupyterPath" "INFO"
    Write-Log "Press Q or ESC to stop monitoring gracefully" "INFO"
    Write-Log "=" * 70
    
    $consecutiveFailures = 0
    $maxConsecutiveFailures = 3
    
    while (Test-ShouldContinue) {
        $script:ChecksDone++
        Write-Log "Health Check #$($script:ChecksDone)" "INFO"
        
        $health = @{
            Timestamp = Get-Date
            CheckNumber = $script:ChecksDone
            Service = Test-JupyterService
            Endpoints = @{}
            Kernels = @()
            Resources = @{}
            Overall = "Unknown"
        }
        
        # Only check endpoints if service is running
        if ($health.Service.Running) {
            $health.Endpoints = Test-JupyterEndpoint
            $health.Kernels = Test-JupyterKernels
        }
        
        # Always check system resources
        $health.Resources = Test-SystemResources
        
        # Determine overall health
        $apiHealthy = ($health.Endpoints.Values | Where-Object { $_.Success }).Count -gt 0
        $health.Overall = if ($health.Service.Running -and $apiHealthy) {
            "Healthy"
        } elseif ($health.Service.Running) {
            "Degraded"
        } else {
            "Failed"
        }
        
        # Handle failures
        if ($health.Overall -eq "Failed") {
            $consecutiveFailures++
            Write-Log "Service health: FAILED ($consecutiveFailures/$maxConsecutiveFailures consecutive)" "ERROR"
            
            if ($consecutiveFailures -ge $maxConsecutiveFailures) {
                if (Invoke-ServiceRecovery) {
                    $consecutiveFailures = 0
                } elseif ($ExitOnFailure) {
                    Write-Log "Exiting due to service failure (ExitOnFailure is set)" "ERROR"
                    $script:MonitoringActive = $false
                    break
                }
            }
        } else {
            if ($consecutiveFailures -gt 0) {
                Write-Log "Service recovered after $consecutiveFailures failure(s)" "SUCCESS"
            }
            $consecutiveFailures = 0
            Write-Log "Service health: $($health.Overall)" $(if ($health.Overall -eq "Healthy") { "SUCCESS" } else { "WARNING" })
        }
        
        # Export health data for external monitoring
        try {
            $healthFile = Join-Path (Split-Path $LogPath -Parent) "health_status.json"
            $health | ConvertTo-Json -Depth 3 | Set-Content $healthFile -Force
        } catch {
            Write-Log "Failed to export health status: $_" "DEBUG" -NoConsole
        }
        
        Write-Log ("-" * 70)
        
        # Wait for next check
        $waitStart = Get-Date
        while ((Test-ShouldContinue) -and ((Get-Date) - $waitStart).TotalSeconds -lt $CheckInterval) {
            Start-Sleep -Milliseconds 500
        }
    }
    
    # Cleanup
    Write-Log "Monitoring stopped after $($script:ChecksDone) checks" "INFO"
    $runtime = (Get-Date) - $script:StartTime
    Write-Log "Total runtime: $([math]::Round($runtime.TotalMinutes, 2)) minutes" "INFO"
    Write-Log "=" * 70
}

# Entry point
if ($MyInvocation.InvocationName -ne '.') {
    Start-Monitoring
} else {
    Write-Host "JupyterLab Monitor loaded. Run Start-Monitoring to begin." -ForegroundColor Cyan
}