"""
DRISHTI SSS — YOLOv8 Detector Training Script  (box detection, post-pivot)

Fine-tunes a COCO-pretrained YOLOv8 detector on pure SSS (Side-Scan Sonar)
data with 5 anomaly classes: crab_pot, submarine_pipeline, shipwreck,
ghost_net, mine_cylinder.

Key SSS-specific adaptations:
  - Colour aug off (single-channel acoustic data)
  - Vertical flip off (would invert the highlight->shadow geometry the model
    and shadow_verification.py depend on)
  - Brightness / scale / erase jitter mapped to real SSS nuisances
    (acoustic gain, pixel-resolution spread, motion dropouts)
  - Class-balanced mosaic for rare-class (shipwreck / mine) exposure
  - Classification-loss gain auto-raised from the training label distribution
  - Early stopping (--patience) picks the epoch; epochs is just a ceiling

Usage:
    python train_yolo_seg.py                        # yolov8s.pt, 100 epochs
    python train_yolo_seg.py --model yolov8n.pt     # lighter edge variant
    python train_yolo_seg.py --epochs 150 --batch 8 # longer / smaller VRAM
"""

import os

# Set BEFORE torch is imported. This box has 15 GB RAM and a finite Windows
# commit limit; the CUDA libs + dataloader worker subprocesses can blow past it
# (WinError 1455 "paging file too small"). LAZY loading defers CUDA kernel load.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = ML_DIR / "configs"
CHECKPOINTS_DIR = ML_DIR / "models" / "checkpoints"


def check_gpu():
    """Check CUDA availability and warn if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            logger.warning(
                "No CUDA GPU detected! Training will be extremely slow on CPU.\n"
                "Recommendation: Use Google Colab (free T4) or Kaggle Notebooks "
                "(free T4/P100, 30 GPU-hrs/week)."
            )
            return False
    except ImportError:
        logger.warning("PyTorch not installed — cannot check GPU")
        return False


def train_yolo_seg(
    data_config: str = "drishti.yaml",
    model: str = "yolov8s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    workers: int = 4,
    device: str = "",
    project: str = "",
    name: str = "",
    resume: bool = False,
    extra_args: dict = None,
):
    """
    Launch YOLOv8-seg training with the specified configuration.

    Args:
        data_config: Path to dataset YAML (relative to configs/ or absolute).
        model: Pretrained model checkpoint name or path.
        epochs: Maximum training epochs.
        imgsz: Input image size.
        batch: Batch size (reduce if OOM).
        patience: Early stopping patience (epochs without improvement).
        device: Device string ("", "0", "cpu", "0,1").
        project: Output project directory.
        name: Experiment name.
        resume: Resume training from last checkpoint.
        extra_args: Additional YOLO training arguments.
    """
    from ultralytics import YOLO

    # Resolve data config path
    data_path = Path(data_config)
    if not data_path.is_absolute():
        data_path = CONFIGS_DIR / data_config
    if not data_path.exists():
        logger.error(f"Data config not found: {data_path}")
        sys.exit(1)

    logger.info(f"Data config: {data_path}")
    logger.info(f"Model: {model}")
    logger.info(f"Epochs: {epochs}, Image size: {imgsz}, Batch: {batch}")
    logger.info(f"Early stopping patience: {patience}")

    # Check GPU
    has_gpu = check_gpu()
    if not device:
        device = "0" if has_gpu else "cpu"

    # Set up output directories
    if not project:
        project = str(CHECKPOINTS_DIR)
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"yolov8_seg_{timestamp}"

    # Load model (COCO-pretrained for transfer learning)
    logger.info(f"Loading pretrained model: {model}")
    yolo_model = YOLO(model)

    # Build training arguments
    train_args = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "workers": workers,   # Windows: each worker re-imports torch (~GBs commit). Keep low.
        "cache": False,       # do not cache the dataset in RAM (only 15 GB, tight commit)
        "device": device,
        "project": project,
        "name": name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,         # final LR = lr0 * lrf
        "warmup_epochs": 3,
        "warmup_bias_lr": 0.1,
        "weight_decay": 0.0005,
        # ---- SSS-specific augmentation (physically motivated) ----------------
        # Sonar is single-channel acoustic data, and the highlight->shadow
        # geometry is the main discriminative cue (and what Module 2 verifies),
        # so colour aug is off and vertical flip is disabled on purpose.
        "hsv_h": 0.0,          # grayscale - no hue
        "hsv_s": 0.0,          # grayscale - no saturation
        "hsv_v": 0.4,          # +/-40% brightness == varying acoustic gain / insonification
        "degrees": 10.0,       # small rotation only - big rotations break shadow direction
        "translate": 0.15,
        "scale": 0.5,          # +/-50% zoom simulates the PS's "varying pixel resolutions"
        "shear": 2.0,          # mild - towfish yaw
        "perspective": 0.0,
        "fliplr": 0.5,         # port/starboard symmetry - always valid for SSS
        "flipud": 0.0,         # DISABLED - a vertical flip inverts the highlight->shadow
                               #   relationship, i.e. the exact feature we want the model
                               #   (and shadow_verification.py) to rely on
        "mosaic": 0.8,         # rare-class exposure; <1.0 so large pipelines / wrecks
                               #   aren't always sliced by a mosaic seam
        "close_mosaic": 15,    # last 15 epochs train on un-mosaicked tiles -> clean finish
        "mixup": 0.1,          # light - heavy mixup paints phantom objects into noisy sonar
        "copy_paste": 0.0,     # no-op for task=detect (needs seg masks); mosaic covers this
        "erasing": 0.4,        # random erase == proxy for motion-induced data dropouts (PS req)
        "bgr": 0.0,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "resume": resume,
    }

    # Compute class weights from training label distribution
    # This penalizes misses on rare classes like shipwreck/ghost_net
    try:
        import sys
        sys.path.insert(0, str(SCRIPT_DIR))
        from build_synthetic_data import tally_training_classes, CLASSES
        splits_dir = data_path.parent.parent / "data" / "splits"
        if not splits_dir.exists():
            # Fallback: resolve from the data config
            import yaml
            with open(data_path) as f:
                cfg = yaml.safe_load(f)
            splits_dir = Path(cfg.get("path", splits_dir))

        counts = tally_training_classes(splits_dir / "train" / "labels")
        total = sum(counts.values())
        if total > 0:
            max_count = max(counts.values())
            # Inverse frequency weighting (capped at 5x)
            cls_weights = []
            for cls_id in sorted(counts):
                weight = min(5.0, max_count / max(counts[cls_id], 1))
                cls_weights.append(weight)
            logger.info(f"Auto-computed class weights: {cls_weights}")
            logger.info(f"Class distribution: {counts}")
            # Note: YOLOv8 doesn't directly support per-class weights via train(),
            # but cls_pw (classification positive weight) can be used to
            # bias the focal loss towards rare classes
            avg_weight = sum(cls_weights) / len(cls_weights)
            train_args["cls"] = max(1.0, avg_weight)  # classification loss gain
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}")

    # Merge extra args
    if extra_args:
        train_args.update(extra_args)

    # Launch training
    logger.info("=" * 60)
    logger.info("Starting YOLOv8-seg SSS training...")
    logger.info("=" * 60)

    results = yolo_model.train(**train_args)

    # Post-training: copy best model to a known location
    best_pt = Path(project) / name / "weights" / "best.pt"
    if best_pt.exists():
        dest = CHECKPOINTS_DIR / "best_detector.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, dest)
        logger.info(f"Best model saved to: {dest}")

    logger.info("Training complete!")
    logger.info(f"Results directory: {Path(project) / name}")
    logger.info("Next steps:")
    logger.info("  1. python evaluate.py              (honest eval on held-out real set)")
    logger.info("  2. python calibrate_confidence.py   (Platt scaling)")
    logger.info("  3. python export_onnx.py            (ONNX + INT8 export)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg on DRISHTI sonar debris dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="drishti.yaml",
        help="Dataset config YAML name or path (default: drishti.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Pretrained detector (default: yolov8s.pt -- box detection, post-pivot). "
        "Options: yolov8n.pt (nano/edge), yolov8s.pt (small), yolov8m.pt (medium)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4,
                        help="dataloader workers (default 4). Drop to 2 or 0 on WinError 1455")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    train_yolo_seg(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
