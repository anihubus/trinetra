"""
DRISHTI SSS Evaluation — YOLOv8-seg on held-out REAL SSS test set.

Reports mAP50 / mAP50-95 (detection), mIoU (segmentation), and per-class
metrics for the 5 SSS anomaly classes:
  0: crab_pot, 1: submarine_pipeline, 2: shipwreck, 3: ghost_net, 4: mine_cylinder

Evaluates ONLY on held-out real images — never on synthetic-augmented data.
This is the number that goes in the pitch deck.

Usage:
    python evaluate.py                                  # uses default best checkpoint
    python evaluate.py --model path/to/best.pt          # specify checkpoint
    python evaluate.py --model path/to/best.pt --save   # save results to disk
"""

import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = ML_DIR / "configs"
CHECKPOINTS_DIR = ML_DIR / "models" / "checkpoints"
EVAL_RESULTS_DIR = CHECKPOINTS_DIR / "eval_results"


def evaluate_yolo_seg(
    model_path: Optional[Path] = None,
    data_config: str = "drishti.yaml",
    imgsz: int = 640,
    batch: int = 16,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.6,
    device: str = "",
    save_results: bool = True,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Run YOLOv8-seg validation on held-out test set and report metrics.

    Args:
        model_path: Path to trained model checkpoint.
        data_config: Dataset config YAML.
        imgsz: Input image size.
        batch: Batch size for validation.
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        device: Device string.
        save_results: Save results to disk.
        split: Which split to evaluate on ("test" for honest eval).

    Returns:
        Dictionary of evaluation metrics.
    """
    from ultralytics import YOLO

    # Resolve model path
    if model_path is None:
        model_path = CHECKPOINTS_DIR / "best_yolo_seg.pt"
        if not model_path.exists():
            # Look for best.pt in any training run
            candidates = list(CHECKPOINTS_DIR.rglob("best.pt"))
            if candidates:
                model_path = candidates[0]
            else:
                logger.error(
                    "No trained model found. Run train_yolo_seg.py first."
                )
                return {}

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return {}

    # Resolve data config
    data_path = Path(data_config)
    if not data_path.is_absolute():
        data_path = CONFIGS_DIR / data_config
    if not data_path.exists():
        logger.error(f"Data config not found: {data_path}")
        return {}

    logger.info(f"Model: {model_path}")
    logger.info(f"Data config: {data_path}")
    logger.info(f"Evaluating on: {split} split")
    logger.info(f"Conf threshold: {conf_threshold}, IoU threshold: {iou_threshold}")

    # Load model
    model = YOLO(str(model_path))

    # Run validation
    results = model.val(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device or None,
        split=split,
        plots=False,         # generate confusion matrix, PR curves, etc.
        save_json=True,     # COCO-format results JSON
        verbose=True,
    )

    # Extract metrics
    metrics = {}

    # Box detection metrics
    if hasattr(results, "box"):
        box = results.box
        metrics["detection"] = {
            "mAP50": float(box.map50) if hasattr(box, "map50") else None,
            "mAP50_95": float(box.map) if hasattr(box, "map") else None,
            "precision": float(box.mp) if hasattr(box, "mp") else None,
            "recall": float(box.mr) if hasattr(box, "mr") else None,
        }

        # Per-class metrics
        if hasattr(box, "ap_class_index") and hasattr(box, "ap50"):
            per_class = {}
            class_names = results.names if hasattr(results, "names") else {}
            for i, cls_idx in enumerate(box.ap_class_index):
                cls_name = class_names.get(int(cls_idx), f"class_{cls_idx}")
                per_class[cls_name] = {
                    "AP50": float(box.ap50[i]) if i < len(box.ap50) else None,
                    "precision": float(box.p[i]) if hasattr(box, "p") and i < len(box.p) else None,
                    "recall": float(box.r[i]) if hasattr(box, "r") and i < len(box.r) else None,
                }
            metrics["per_class_detection"] = per_class

    # Segmentation metrics
    if hasattr(results, "seg"):
        seg = results.seg
        metrics["segmentation"] = {
            "mAP50": float(seg.map50) if hasattr(seg, "map50") else None,
            "mAP50_95": float(seg.map) if hasattr(seg, "map") else None,
            "precision": float(seg.mp) if hasattr(seg, "mp") else None,
            "recall": float(seg.mr) if hasattr(seg, "mr") else None,
        }

    # Speed
    if hasattr(results, "speed"):
        metrics["speed"] = {
            "preprocess_ms": results.speed.get("preprocess", None),
            "inference_ms": results.speed.get("inference", None),
            "postprocess_ms": results.speed.get("postprocess", None),
        }

    # Metadata
    metrics["metadata"] = {
        "model_path": str(model_path),
        "data_config": str(data_path),
        "split": split,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "imgsz": imgsz,
        "timestamp": datetime.now().isoformat(),
    }

    # DRISHTI SSS class names for reporting
    sss_classes = ["crab_pot", "submarine_pipeline", "shipwreck", "ghost_net", "mine_cylinder"]

    # Print results
    logger.info("=" * 60)
    logger.info("DRISHTI SSS EVALUATION — HELD-OUT REAL TEST SET")
    logger.info("=" * 60)

    if "detection" in metrics:
        det = metrics["detection"]
        logger.info(f"Detection mAP50:     {det.get('mAP50', 'N/A')}")
        logger.info(f"Detection mAP50-95:  {det.get('mAP50_95', 'N/A')}")
        logger.info(f"Detection Precision:  {det.get('precision', 'N/A')}")
        logger.info(f"Detection Recall:     {det.get('recall', 'N/A')}")

    if "segmentation" in metrics:
        seg_m = metrics["segmentation"]
        logger.info(f"Segment mAP50:       {seg_m.get('mAP50', 'N/A')}")
        logger.info(f"Segment mAP50-95:    {seg_m.get('mAP50_95', 'N/A')}")

    if "per_class_detection" in metrics:
        logger.info("\nPer-class AP50 (SSS anomaly classes):")
        for cls_name, cls_metrics in metrics["per_class_detection"].items():
            ap50 = cls_metrics.get("AP50", "N/A")
            prec = cls_metrics.get("precision", "N/A")
            rec = cls_metrics.get("recall", "N/A")
            # Flag rare classes that might have low recall
            flag = " ⚠️ LOW" if isinstance(rec, float) and rec < 0.5 else ""
            logger.info(f"  {cls_name:25s}  AP50={ap50}  P={prec}  R={rec}{flag}")

    # Estimate false-positive density (FP per km² — requires survey metadata)
    if "detection" in metrics:
        prec = metrics["detection"].get("precision")
        if prec is not None and prec > 0:
            fp_rate = 1.0 - prec
            metrics["fp_analysis"] = {
                "estimated_fp_rate": float(fp_rate),
                "note": "FP/km² requires survey area metadata (not available in test set)",
            }
            logger.info(f"\n  Estimated FP rate: {fp_rate:.4f}")

    # Save results
    if save_results:
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = EVAL_RESULTS_DIR / f"eval_{split}_{timestamp}.json"

        # Convert numpy values to Python types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_metrics = json.loads(
            json.dumps(metrics, default=_convert)
        )

        with open(results_path, "w") as f:
            json.dump(clean_metrics, f, indent=2)

        logger.info(f"\nResults saved to: {results_path}")

    logger.info("=" * 60)
    logger.info(
        "NOTE: These results are from the HELD-OUT REAL SSS test set.\n"
        "Classes trained with synthetic data (ghost_net, shipwreck, pipeline,\n"
        "mine_cylinder) should be evaluated with extra scrutiny.\n"
        "This is the number that goes in the pitch deck."
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8-seg on held-out real test set"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="drishti.yaml",
        help="Dataset config YAML",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )
    args = parser.parse_args()

    evaluate_yolo_seg(
        model_path=args.model,
        data_config=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        save_results=not args.no_save,
        split=args.split,
    )


if __name__ == "__main__":
    main()
