"""
Resume an interrupted YOLO training run from its last.pt.

Ultralytics writes weights/last.pt after every epoch. resume=True restores the
optimizer state, LR schedule and every hyperparameter from that checkpoint and
continues from the next epoch - do NOT pass other train args with it.

    python ml/scripts/resume_training.py                 # newest run
    python ml/scripts/resume_training.py --run <dir>     # a specific run dir
"""

import os
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

CKPT = Path(__file__).resolve().parent.parent / "models" / "checkpoints"


def newest_run() -> Path:
    runs = sorted(CKPT.glob("yolov8_seg_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if (r / "weights" / "last.pt").exists():
            return r
    raise SystemExit("no run with weights/last.pt found under models/checkpoints/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None, help="run directory (default: newest)")
    args = ap.parse_args()

    run = args.run or newest_run()
    last = run / "weights" / "last.pt"
    if not last.exists():
        raise SystemExit(f"no checkpoint: {last}")

    from ultralytics import YOLO
    print(f"resuming: {last}")
    YOLO(str(last)).train(resume=True)

    best = run / "weights" / "best.pt"
    if best.exists():
        import shutil
        dest = CKPT / "best_detector.pt"
        shutil.copy2(best, dest)
        print(f"best copied to {dest}")


if __name__ == "__main__":
    main()
