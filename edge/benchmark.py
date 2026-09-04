"""
Benchmarks every exported ONNX model on THIS machine: size (MB), ms/tile, FPS.

Run on the laptop CPU for the Tier 0 number; run again on a Jetson (with
onnxruntime-gpu) for the Tier 1 number. Both go straight into the deck instead of
an unverified "lightweight" claim.

    python edge/benchmark.py                 # all *.onnx in ml/models/exported
    python edge/benchmark.py --runs 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1] / "ml" / "models" / "exported"


def bench(onnx_path: Path, imgsz: int, warmup: int, runs: int, provider: str) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {
        "model": onnx_path.name,
        "provider": provider,
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "mean_ms": round(float(ts.mean()), 1),
        "p95_ms": round(float(np.percentile(ts, 95)), 1),
        "fps": round(1000 / float(ts.mean()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=EXP)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    args = ap.parse_args()

    models = sorted(args.dir.glob("*.onnx"))
    if not models:
        raise SystemExit(f"no .onnx files in {args.dir}")

    out = [bench(m, args.imgsz, args.warmup, args.runs, args.provider) for m in models]
    print(f"\n  {'model':32s} {'MB':>7} {'mean ms':>9} {'p95 ms':>8} {'FPS':>6}")
    print("  " + "-" * 66)
    for r in out:
        print(f"  {r['model']:32s} {r['size_mb']:7.1f} {r['mean_ms']:9.1f} {r['p95_ms']:8.1f} {r['fps']:6.1f}")
    (args.dir / "benchmark_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  -> {args.dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
