@echo off
setlocal enabledelayedexpansion
title AI Memory Service - Process Cleanup

echo 🔍 Starting safe process cleanup for AI Memory Service...
echo ==========================================

REM Section 1: Kill memory-server processes with validation
echo 📍 Checking for memory-server.exe processes...
tasklist /FI "IMAGENAME eq memory-server.exe" /FO CSV | findstr "memory-server.exe" >nul
if !ERRORLEVEL! == 0 (
    echo 🎯 Found memory-server.exe processes, terminating...
    taskkill /F /IM memory-server.exe /T >nul 2>&1
    if !ERRORLEVEL! == 0 (
        echo ✅ Successfully killed memory-server.exe processes
    ) else (
        echo ❌ Failed to kill some memory-server.exe processes
    )
) else (
    echo ℹ️  No memory-server.exe processes found
)

REM Section 2: Kill Python embedding_server processes with precise filtering
echo.
echo 📍 Checking for embedding_server.py processes...
set FOUND_EMBEDDING=0

REM Get all Python processes with command line and filter precisely
for /f "skip=1 tokens=2,9* delims=," %%A in ('wmic process where "name='python.exe'" get ProcessId^,CommandLine /format:csv 2^>nul') do (
    set "CMDLINE=%%C"
    set "PID=%%B"
    
    REM Remove quotes and check if command line contains embedding_server.py
    set "CMDLINE=!CMDLINE:"=!"
    echo !CMDLINE! | findstr /C:"embedding_server.py" >nul
    if !ERRORLEVEL! == 0 (
        set FOUND_EMBEDDING=1
        echo 🎯 Found embedding_server.py process [PID: !PID!]
        echo    Command: !CMDLINE!
        taskkill /F /PID !PID! >nul 2>&1
        if !ERRORLEVEL! == 0 (
            echo ✅ Successfully killed PID !PID!
        ) else (
            echo ❌ Failed to kill PID !PID!
        )
    )
)

if !FOUND_EMBEDDING! == 0 (
    echo ℹ️  No embedding_server.py processes found
) else (
    echo ✅ Embedding server cleanup completed
)

REM Section 3: Verify cleanup success
echo.
echo 🔍 Verifying cleanup results...
echo ==========================================

REM Check remaining memory-server processes
tasklist /FI "IMAGENAME eq memory-server.exe" /FO CSV | findstr "memory-server.exe" >nul
if !ERRORLEVEL! == 0 (
    echo ⚠️  Warning: Some memory-server.exe processes may still be running
) else (
    echo ✅ All memory-server.exe processes terminated
)

REM Check remaining embedding_server processes
set REMAINING=0
for /f "skip=1 tokens=2,9* delims=," %%A in ('wmic process where "name='python.exe'" get ProcessId^,CommandLine /format:csv 2^>nul') do (
    set "CMDLINE=%%C"
    set "CMDLINE=!CMDLINE:"=!"
    echo !CMDLINE! | findstr /C:"embedding_server.py" >nul
    if !ERRORLEVEL! == 0 set REMAINING=1
)

if !REMAINING! == 1 (
    echo ⚠️  Warning: Some embedding_server.py processes may still be running
) else (
    echo ✅ All embedding_server.py processes terminated
)

echo.
echo 🧹 Process cleanup completed successfully
echo Ready for compilation and forensic investigation
pause