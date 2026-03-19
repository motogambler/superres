GPU-Accelerated Image Super-Resolution Demo

Overview

This project provides a small, runnable image super-resolution demo with a bicubic baseline and optional paths for a PyTorch model, ONNX runtime, and TensorRT.

What's included
- `app.py` — Gradio web UI (upload image, choose method, run)
- `model_stub.py` — loader/run wrapper for a PyTorch model (if present)
- `models/` — model helpers and demo scripted model target
- `scripts/` — helpers: `save_demo_model.py`, `export_onnx.py`, `benchmark.py`, `convert_trt.sh`
- `requirements.txt` — reproducible Python deps
- `Dockerfile` — sample CPU container and notes for GPU images

Prerequisites
- Python 3.8+ (we tested with 3.8)
- Git Bash, WSL, or a POSIX-like shell on Windows is recommended for the included scripts.

Quick start (fresh clone)

1) Create and activate a project venv (Git Bash on Windows):

```bash
python -m venv .venv
source .venv/Scripts/activate    # Git Bash
# or on PowerShell: .venv\\Scripts\\Activate.ps1
# or on Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2) Run the demo locally:

```bash
python app.py
```

3) Open the Gradio link printed in the terminal and upload an image.

Model / export / benchmark
- Create a small demo scripted model (builds `models/model.pt`):

```bash
python scripts/save_demo_model.py
```

- Export the scripted model to ONNX:

```bash
python scripts/export_onnx.py --scale 4 --out models/model.onnx
```

- Run a benchmark (requires an example image, e.g. `examples/limes.jpg`):

```bash
python scripts/benchmark.py --image examples/limes.jpg --runs 20
```

Benchmark results are appended to `benchmarks/results.csv`.

Notes on GPU
- To run ONNX on GPU, install the matching `onnxruntime-gpu` package and ensure CUDA drivers are present. For TensorRT-based production builds, use `trtexec` to convert `models/model.onnx` to a TRT engine; see `scripts/convert_trt.sh`.

Support
- If you run into dependency issues on Windows, prefer creating the project `.venv` and installing there (do not install packages system-wide).

License
- MIT