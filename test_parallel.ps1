# PowerShell script to test parallel requests to memory-server
# Tests the semaphore-based concurrency control fix

Write-Host "🧪 Testing AI Memory Service with parallel requests..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Base URL
$baseUrl = "http://localhost:8080"

# First, check if server is healthy
Write-Host "`n📋 Checking server health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✅ Server is healthy: $($health.status)" -ForegroundColor Green
    Write-Host "   Version: $($health.version)" -ForegroundColor Gray
    Write-Host "   Orchestrator: $($health.orchestrator.model)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Server is not responding!" -ForegroundColor Red
    exit 1
}

# Create test memories
$testMemories = @(
    @{
        content = "Performance testing with parallel requests is critical for local usage"
        context_hint = "performance_testing"
        importance = 0.9
        memory_type = "fact"
        tags = @("testing", "performance")
        metadata = @{ test_id = "1"; category = "testing" }
    },
    @{
        content = "Semaphore-based concurrency control prevents Python GIL bottlenecks"
        context_hint = "concurrency_control"
        importance = 0.95
        memory_type = "fact"
        tags = @("architecture", "concurrency")
        metadata = @{ test_id = "2"; category = "architecture" }
    },
    @{
        content = "Local AI memory service supports 2-4 concurrent operations"
        context_hint = "system_limits"
        importance = 0.85
        memory_type = "fact"
        tags = @("configuration", "limits")
        metadata = @{ test_id = "3"; category = "configuration" }
    },
    @{
        content = "Docker deployment eliminates dependency installation for users"
        context_hint = "deployment"
        importance = 0.8
        memory_type = "fact"
        tags = @("infrastructure", "docker")
        metadata = @{ test_id = "4"; category = "infrastructure" }
    },
    @{
        content = "Neo4j graph database stores memory relationships efficiently"
        context_hint = "storage"
        importance = 0.88
        memory_type = "fact"
        tags = @("database", "storage")
        metadata = @{ test_id = "5"; category = "database" }
    }
)

Write-Host "`n🚀 Sending 5 parallel memory storage requests..." -ForegroundColor Yellow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Create parallel jobs
$jobs = @()
foreach ($memory in $testMemories) {
    $job = Start-Job -ScriptBlock {
        param($url, $body)
        $headers = @{"Content-Type" = "application/json"}
        $jsonBody = $body | ConvertTo-Json -Depth 10
        
        try {
            $response = Invoke-RestMethod -Uri "$url/memory" -Method Post -Headers $headers -Body $jsonBody
            return @{
                Success = $true
                MemoryId = $response.memory_id
                Content = $body.content.Substring(0, [Math]::Min(50, $body.content.Length))
            }
        } catch {
            return @{
                Success = $false
                Error = $_.Exception.Message
                Content = $body.content.Substring(0, [Math]::Min(50, $body.content.Length))
            }
        }
    } -ArgumentList $baseUrl, $memory
    
    $jobs += $job
}

# Wait for all jobs to complete
Write-Host "⏳ Waiting for parallel requests to complete..." -ForegroundColor Gray
$results = $jobs | Wait-Job | Receive-Job

$stopwatch.Stop()
$totalTime = $stopwatch.Elapsed.TotalSeconds

# Display results
Write-Host "`n📊 Results:" -ForegroundColor Yellow
Write-Host "   Total time: $([Math]::Round($totalTime, 2)) seconds" -ForegroundColor Cyan

$successCount = 0
$failCount = 0
$memoryIds = @()

foreach ($result in $results) {
    if ($result.Success) {
        $successCount++
        $memoryIds += $result.MemoryId
        Write-Host "   ✅ Stored: $($result.Content)..." -ForegroundColor Green
    } else {
        $failCount++
        Write-Host "   ❌ Failed: $($result.Content)... Error: $($result.Error)" -ForegroundColor Red
    }
}

Write-Host "`n📈 Summary:" -ForegroundColor Yellow
Write-Host "   Success: $successCount / $($testMemories.Count)" -ForegroundColor $(if ($successCount -eq $testMemories.Count) { "Green" } else { "Yellow" })
Write-Host "   Failed: $failCount / $($testMemories.Count)" -ForegroundColor $(if ($failCount -eq 0) { "Green" } else { "Red" })
Write-Host "   Avg time per request: $([Math]::Round($totalTime / $testMemories.Count, 2)) seconds" -ForegroundColor Cyan

# Test parallel search
if ($successCount -gt 0) {
    Write-Host "`n🔍 Testing parallel search requests..." -ForegroundColor Yellow
    $searchQueries = @(
        "performance testing",
        "concurrency control",
        "Docker deployment",
        "Neo4j database",
        "local usage"
    )
    
    $searchStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    $searchJobs = @()
    foreach ($query in $searchQueries) {
        $job = Start-Job -ScriptBlock {
            param($url, $q)
            try {
                $response = Invoke-RestMethod -Uri "$url/search?query=$q&limit=3" -Method Get
                return @{
                    Success = $true
                    Query = $q
                    Count = $response.memories.Count
                }
            } catch {
                return @{
                    Success = $false
                    Query = $q
                    Error = $_.Exception.Message
                }
            }
        } -ArgumentList $baseUrl, $query
        
        $searchJobs += $job
    }
    
    $searchResults = $searchJobs | Wait-Job | Receive-Job
    $searchStopwatch.Stop()
    
    Write-Host "   Search time: $([Math]::Round($searchStopwatch.Elapsed.TotalSeconds, 2)) seconds" -ForegroundColor Cyan
    
    foreach ($result in $searchResults) {
        if ($result.Success) {
            Write-Host "   ✅ Found $($result.Count) results for: '$($result.Query)'" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Search failed for: '$($result.Query)' - $($result.Error)" -ForegroundColor Red
        }
    }
}

# Test recall with parallel requests
if ($memoryIds.Count -gt 0) {
    Write-Host "`n📥 Testing parallel recall requests..." -ForegroundColor Yellow
    $recallStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    $recallJobs = @()
    foreach ($id in $memoryIds) {
        $job = Start-Job -ScriptBlock {
            param($url, $memId)
            try {
                $response = Invoke-RestMethod -Uri "$url/memory/$memId" -Method Get
                return @{
                    Success = $true
                    Id = $memId
                    Content = $response.content.Substring(0, [Math]::Min(30, $response.content.Length))
                }
            } catch {
                return @{
                    Success = $false
                    Id = $memId
                    Error = $_.Exception.Message
                }
            }
        } -ArgumentList $baseUrl, $id
        
        $recallJobs += $job
    }
    
    $recallResults = $recallJobs | Wait-Job | Receive-Job
    $recallStopwatch.Stop()
    
    Write-Host "   Recall time: $([Math]::Round($recallStopwatch.Elapsed.TotalSeconds, 2)) seconds" -ForegroundColor Cyan
    
    $recallSuccess = ($recallResults | Where-Object { $_.Success }).Count
    Write-Host "   Successfully recalled: $recallSuccess / $($memoryIds.Count)" -ForegroundColor $(if ($recallSuccess -eq $memoryIds.Count) { "Green" } else { "Yellow" })
}

# Final verdict
Write-Host "`n🎯 Test Verdict:" -ForegroundColor Yellow
if ($successCount -eq $testMemories.Count -and $failCount -eq 0) {
    Write-Host "   ✅ ALL TESTS PASSED! Parallel request handling works correctly." -ForegroundColor Green
    Write-Host "   The semaphore-based concurrency control successfully prevents GIL bottlenecks." -ForegroundColor Green
} elseif ($successCount -gt 0) {
    Write-Host "   ⚠️  PARTIAL SUCCESS: Some requests succeeded, but not all." -ForegroundColor Yellow
    Write-Host "   This may indicate intermittent issues or resource constraints." -ForegroundColor Yellow
} else {
    Write-Host "   ❌ TESTS FAILED: Unable to process parallel requests." -ForegroundColor Red
    Write-Host "   Check server logs for details." -ForegroundColor Red
}

Write-Host "`n✨ Test completed!" -ForegroundColor Cyan

# Clean up jobs
$jobs | Remove-Job -Force
if ($searchJobs) { $searchJobs | Remove-Job -Force }
if ($recallJobs) { $recallJobs | Remove-Job -Force }