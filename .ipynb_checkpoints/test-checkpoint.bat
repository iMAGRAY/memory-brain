@echo off
REM AI Memory Service - Test Runner for Windows
REM Simple wrapper for PowerShell test script
REM Usage: test.bat [options]
REM   Options: -Quick, -Integration, -Benchmark, -Coverage, -Verbose

echo AI Memory Service - Running Tests...
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found! Please install PowerShell to run tests.
    exit /b 1
)

REM Check if test script exists
if not exist scripts\run_tests.ps1 (
    echo ERROR: Test script not found at scripts\run_tests.ps1
    echo Please ensure you're running this from the project root directory.
    exit /b 1
)

REM Run the PowerShell test script
REM Note: ExecutionPolicy Bypass is used for local development only
REM In production environments, consider signing the script
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1 %*

REM Capture exit code
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% equ 0 (
    echo.
    echo SUCCESS: All tests passed!
) else (
    echo.
    echo FAILURE: Some tests failed. Exit code: %EXIT_CODE%
)

exit /b %EXIT_CODE%