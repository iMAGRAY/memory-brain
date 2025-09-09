@echo off
REM JupyterLab Global Startup Script
REM Version: 1.0.0

setlocal EnableDelayedExpansion

REM Set paths
set "JUPYTER_CONFIG_DIR=C:\ProgramData\jupyter"
set "JUPYTER_LOG_DIR=C:\srv\jupyter\logs"

REM Create directories if needed
if not exist "C:\srv\jupyter" mkdir "C:\srv\jupyter"
if not exist "%JUPYTER_LOG_DIR%" mkdir "%JUPYTER_LOG_DIR%"

echo Starting JupyterLab...
echo ========================================
echo Configuration: %JUPYTER_CONFIG_DIR%
echo Logs: %JUPYTER_LOG_DIR%
echo.

REM Check for existing processes
tasklist /FI "IMAGENAME eq jupyter-lab.exe" | find /i "jupyter-lab.exe" >nul
if !errorlevel! equ 0 (
    echo Stopping existing JupyterLab process...
    taskkill /IM jupyter-lab.exe /F >nul 2>&1
    timeout /t 2 /nobreak >nul
)

REM Set environment variables
set "JUPYTER_CONFIG_DIR=%JUPYTER_CONFIG_DIR%"
set "JUPYTER_DATA_DIR=C:\ProgramData\jupyter\share"
set "JUPYTER_RUNTIME_DIR=C:\srv\jupyter\runtime"
set "JUPYTER_ROOT_DIR=C:\srv\jupyter"

REM Start JupyterLab with the secure config (but no auth as requested)
echo Launching JupyterLab...
jupyter lab --config="%JUPYTER_CONFIG_DIR%\jupyter_server_config.py" --no-browser

exit /b 0