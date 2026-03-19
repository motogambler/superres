#!/usr/bin/env python3
"""Benchmark inference paths: torch CPU, torch CUDA (if available), ONNX Runtime CPU/GPU (if installed).

Usage:
  python scripts/benchmark.py --image examples/test.jpg --runs 20

If `models/model.pt` is present it'll be loaded; otherwise an untrained model is used.
"""
import time
from pathlib import Path
import sys
import argparse
import numpy as np
from PIL import Image

# Ensure project root (superres/) is on sys.path so `models` imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from models.espcn import make_model


def load_model(scale=4):
    p = Path('models/model.pt')
    if p.exists():
        try:
            m = torch.jit.load(str(p)) if str(p).endswith('.jit') else torch.load(str(p))
            m.eval()
            return m
        except Exception:
            pass
    return make_model(scale)


def prepare(img_path: Path, scale=4):
    img = Image.open(img_path).convert('RGB')
    # use a small crop for benchmarking
    img = img.resize((64, 64))
    arr = np.array(img).astype('float32') / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def time_torch(model, tensor, device, runs=20):
    model = model.to(device)
    inp = tensor.to(device)
    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(inp)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(inp)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / runs


def time_onnx(onnx_path: Path, tensor, use_gpu=False, runs=20):
    try:
        import onnxruntime as ort
    except Exception:
        print('onnxruntime not installed')
        return None
    providers = ['CPUExecutionProvider']
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp = tensor.numpy()
    ort_input = {sess.get_inputs()[0].name: inp}
    # warmup
    for _ in range(3):
        sess.run(None, ort_input)
    t0 = time.time()
    for _ in range(runs):
        sess.run(None, ort_input)
    return (time.time() - t0) / runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=str, default='examples/test.jpg')
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--scale', type=int, default=4)
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print('Image not found at', img_path, 'create examples/test.jpg or pass --image')
        return

    tensor = prepare(img_path, args.scale)
    model = load_model(args.scale)

    print('Benchmarking torch CPU...')
    t_cpu = time_torch(model, tensor, torch.device('cpu'), runs=args.runs)
    print(f'torch CPU avg time (s): {t_cpu:.6f}')

    if torch.cuda.is_available():
        print('Benchmarking torch CUDA...')
        t_gpu = time_torch(model, tensor, torch.device('cuda'), runs=args.runs)
        print(f'torch CUDA avg time (s): {t_gpu:.6f}')
    else:
        print('CUDA not available for torch')

    onnx_path = Path('models/model.onnx')
    if onnx_path.exists():
        print('Benchmarking ONNX Runtime CPU...')
        t_ort_cpu = time_onnx(onnx_path, tensor, use_gpu=False, runs=args.runs)
        if t_ort_cpu is not None:
            print(f'onnxruntime CPU avg time (s): {t_ort_cpu:.6f}')
        print('Benchmarking ONNX Runtime GPU (if available)...')
        t_ort_gpu = time_onnx(onnx_path, tensor, use_gpu=True, runs=args.runs)
        if t_ort_gpu is not None:
            print(f'onnxruntime GPU avg time (s): {t_ort_gpu:.6f}')
    else:
        print('ONNX model not found at models/model.onnx — run scripts/export_onnx.py first')

    # Save benchmark results to CSV for quick reporting
    out_dir = Path('benchmarks')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / 'results.csv'
    headers = ['name','torch_cpu_s','torch_cuda_s','onnx_cpu_s','onnx_gpu_s']
    row = {
        'name': args.image,
        'torch_cpu_s': f"{t_cpu:.6f}",
        'torch_cuda_s': f"{t_gpu:.6f}" if 't_gpu' in locals() else '',
        'onnx_cpu_s': f"{t_ort_cpu:.6f}" if 't_ort_cpu' in locals() and t_ort_cpu is not None else '',
        'onnx_gpu_s': f"{t_ort_gpu:.6f}" if 't_ort_gpu' in locals() and t_ort_gpu is not None else ''
    }
    write_header = not csv.exists()
    with open(csv, 'a') as f:
        if write_header:
            f.write(','.join(headers) + '\n')
        f.write(','.join([row[h] for h in headers]) + '\n')
    print('Wrote benchmark results to', csv)


if __name__ == '__main__':
    main()
