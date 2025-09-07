# ONNX Runtime Setup Guide

## Version Requirements

The AI Memory Service uses `ort` crate version 2.0.0-rc.10, which requires ONNX Runtime version 1.22.x.

### Current Issue
- **ort crate version**: 2.0.0-rc.10 (requires ONNX Runtime 1.22.x)
- **Installed ONNX Runtime**: 1.19.2 (located in `./onnxruntime/lib/`)
- **Status**: Version mismatch - tests will fail until resolved

## Solution

You need to update the ONNX Runtime DLLs to version 1.22.x:

1. Download ONNX Runtime 1.22.x from: https://github.com/microsoft/onnxruntime/releases
2. Replace the files in `./onnxruntime/lib/`:
   - `onnxruntime.dll`
   - `onnxruntime_providers_shared.dll`

## Alternative Solution

If you need to keep ONNX Runtime 1.19.2, you would need to use an older version of the ort crate, but versions compatible with 1.19.x are no longer available on crates.io.

## Verifying Installation

After updating the DLLs, run:
```bash
cargo test --test onnx_runtime_test
```

The tests should pass without version mismatch errors.

## EmbeddingGemma Model Requirements

When using the EmbeddingGemma-300m model:
- Model format: ONNX
- Expected location: `./models/` directory
- Model repository: https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX

Note: The model requires accepting the Gemma license terms on Hugging Face before downloading.