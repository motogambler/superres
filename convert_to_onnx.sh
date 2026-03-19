#!/usr/bin/env bash
# Example: convert a PyTorch model to ONNX (adjust input shape and model loading)
# Assumes a scripted/traced model that accepts a tensor [1,3,H,W]

set -e
MODEL=models/model.pt
OUT=models/model.onnx
python - <<'PY'
import torch
from pathlib import Path
MODEL='models/model.pt'
OUT='models/model.onnx'
if not Path(MODEL).exists():
    print('Model file not found at', MODEL)
    raise SystemExit(1)
model = torch.jit.load(MODEL) if MODEL.endswith('.jit') else torch.load(MODEL)
model.eval()
# Dummy input — replace H,W with expected size or use dynamic axes
dummy = torch.randn(1,3,64,64)
torch.onnx.export(model, dummy, OUT, opset_version=12, input_names=['input'], output_names=['output'], dynamic_axes={'input':{2:'h',3:'w'}, 'output':{2:'h',3:'w'}})
print('Exported ONNX to', OUT)
PY
