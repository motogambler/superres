#!/usr/bin/env python3
"""Export the demo model to ONNX.

Usage:
  python scripts/export_onnx.py --scale 4 --out models/model.onnx
If `models/model.pt` is missing the script will attempt to instantiate the model
and export an untrained version so you can verify the pipeline.
"""
import argparse
from pathlib import Path
import sys
import torch

# Ensure project root (superres/) is on sys.path so `models` imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.espcn import make_model


def load_torch_model(path: Path, scale: int):
    if path.exists():
        try:
            m = torch.jit.load(str(path)) if str(path).endswith('.jit') else torch.load(str(path))
            m.eval()
            print("Loaded model from", path)
            return m
        except Exception as e:
            print("Failed to load model, will instantiate fresh model:", e)
    print("Instantiating fresh model (untrained)")
    return make_model(scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='models/model.pt')
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--out', type=str, default='models/model.onnx')
    args = p.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out)

    model = load_torch_model(model_path, args.scale)
    model.eval()

    dummy = torch.randn(1, 3, 64, 64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out_path), opset_version=12,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {2: 'h', 3: 'w'}, 'output': {2: 'h', 3: 'w'}})
    print('Exported ONNX to', out_path)


if __name__ == '__main__':
    main()
