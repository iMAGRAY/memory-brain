@echo off
echo Starting AI Memory Service...

REM Set environment variables
set RUST_LOG=info
set ORT_DYLIB_PATH=%~dp0onnxruntime\lib\onnxruntime.dll
set PATH=%~dp0onnxruntime\lib;%PATH%

REM Run the service
echo Running with ONNX Runtime from: %ORT_DYLIB_PATH%
target\release\memory-server.exe