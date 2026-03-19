#!/usr/bin/env python3
"""Create and save a demo TorchScript model to `models/model.pt`.

This produces a usable model file (scripted) so ONNX export and benchmarks
can run even without a pretrained checkpoint.
"""
from pathlib import Path
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.espcn import make_model


def main():
    out = Path('models/model.pt')
    out.parent.mkdir(parents=True, exist_ok=True)
    model = make_model(scale=4)
    model.eval()
    # Script the model to create a portable artifact
    dummy = torch.randn(1, 3, 64, 64)
    try:
        scripted = torch.jit.trace(model, dummy)
    except Exception:
        scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(out))
    print('Saved demo scripted model to', out)


if __name__ == '__main__':
    main()
