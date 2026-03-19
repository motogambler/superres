#!/usr/bin/env bash
# Convert ONNX model to TensorRT engine using trtexec (if available).
# Requires NVIDIA TensorRT/trtexec installed and accessible in PATH.

set -e
ONNX=models/model.onnx
OUT=models/model.trt

if [ ! -f "$ONNX" ]; then
  echo "ONNX model not found at $ONNX. Run: python scripts/export_onnx.py --out $ONNX"
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found in PATH. Install TensorRT and ensure trtexec is available."
  exit 1
fi

echo "Converting $ONNX -> $OUT using trtexec"
trtexec --onnx=$ONNX --saveEngine=$OUT --explicitBatch --workspace=2048 --fp16 || {
  echo "trtexec failed. Try without --fp16 or increase --workspace"; exit 1
}
echo "Saved TensorRT engine to $OUT"
