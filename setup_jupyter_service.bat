@echo off
REM Setup JupyterLab as Windows Service
REM Using NSSM (Non-Sucking Service Manager)

setlocal EnableDelayedExpansion

echo Setting up JupyterLab as Windows Service...
echo ========================================

REM Set paths
set "NSSM_PATH=%CD%\nssm-2.24\win64\nssm.exe"
set "PYTHON_PATH=C:\Users\1\AppData\Local\Programs\Python\Python313\python.exe"
set "JUPYTER_CONFIG=C:\ProgramData\jupyter\jupyter_server_config_secure.py"
set "SERVICE_NAME=JupyterLab"
set "STARTUP_SCRIPT=%CD%\start_jupyter_global.bat"

REM Check NSSM exists
if not exist "%NSSM_PATH%" (
    echo ERROR: NSSM not found at %NSSM_PATH%
    exit /b 1
)

REM Stop existing service if exists
echo Checking for existing service...
"%NSSM_PATH%" status "%SERVICE_NAME%" >nul 2>&1
if !errorlevel! equ 0 (
    echo Stopping existing service...
    "%NSSM_PATH%" stop "%SERVICE_NAME%"
    timeout /t 3 /nobreak >nul
    echo Removing existing service...
    "%NSSM_PATH%" remove "%SERVICE_NAME%" confirm
)

REM Install service
echo Installing JupyterLab service...
"%NSSM_PATH%" install "%SERVICE_NAME%" "%STARTUP_SCRIPT%"

REM Configure service parameters
echo Configuring service parameters...

REM Set display name and description
"%NSSM_PATH%" set "%SERVICE_NAME%" DisplayName "JupyterLab Multi-Kernel Server"
"%NSSM_PATH%" set "%SERVICE_NAME%" Description "JupyterLab server with Python, Rust, JavaScript/TypeScript, R, and Bash kernels"

REM Set startup directory
"%NSSM_PATH%" set "%SERVICE_NAME%" AppDirectory "C:\srv\jupyter"

REM Set environment variables
"%NSSM_PATH%" set "%SERVICE_NAME%" AppEnvironmentExtra "JUPYTER_CONFIG_DIR=C:\ProgramData\jupyter" "JUPYTER_DATA_DIR=C:\ProgramData\jupyter\share" "JUPYTER_ROOT_DIR=C:\srv\jupyter"

REM Set restart policy
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRestartDelay 5000
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStopMethodSkip 0
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStopMethodConsole 1500
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStopMethodWindow 1500
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStopMethodThreads 1500

REM Set logging
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStdout "C:\srv\jupyter\logs\service_stdout.log"
"%NSSM_PATH%" set "%SERVICE_NAME%" AppStderr "C:\srv\jupyter\logs\service_stderr.log"
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRotateFiles 1
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRotateOnline 1
"%NSSM_PATH%" set "%SERVICE_NAME%" AppRotateBytes 10485760

REM Set service to start automatically
"%NSSM_PATH%" set "%SERVICE_NAME%" Start SERVICE_AUTO_START

echo.
echo Service configuration complete!
echo ========================================
echo.
echo To manage the service:
echo   Start:   %NSSM_PATH% start %SERVICE_NAME%
echo   Stop:    %NSSM_PATH% stop %SERVICE_NAME%
echo   Restart: %NSSM_PATH% restart %SERVICE_NAME%
echo   Status:  %NSSM_PATH% status %SERVICE_NAME%
echo   Remove:  %NSSM_PATH% remove %SERVICE_NAME% confirm
echo.
echo To start the service now, run:
echo   %NSSM_PATH% start %SERVICE_NAME%
echo.

pause