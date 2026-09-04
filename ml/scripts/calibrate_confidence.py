"""
Fits sklearn.linear_model.LogisticRegression (Platt scaling) on held-out
(raw_confidence -> correct/incorrect) pairs, so the dashboard's 0-100% score is an
honest calibrated probability, not a raw softmax value. Saves calibrator.pkl,
loaded by ml/inference/confidence_filter.py at serve time.

Usage:
    python calibrate_confidence.py                         # uses defaults
    python calibrate_confidence.py --model best.pt --split val
"""

import logging
import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = ML_DIR / "configs"
CHECKPOINTS_DIR = ML_DIR / "models" / "checkpoints"
EXPORTED_DIR = ML_DIR / "models" / "exported"
SPLITS_DIR = ML_DIR / "data" / "splits"


def collect_predictions(
    model_path: Path,
    data_config: Path,
    split: str = "val",
    imgsz: int = 640,
    conf_threshold: float = 0.01,  # low threshold to get full range of scores
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on the validation set and collect (raw_confidence, is_correct) pairs.

    A prediction is 'correct' if its IoU with a ground-truth box of the same class
    exceeds the iou_threshold.

    Returns:
        (confidences, correctness) — both 1D numpy arrays.
    """
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(model_path))

    # Get image and label paths
    images_dir = SPLITS_DIR / split / "images"
    labels_dir = SPLITS_DIR / split / "labels"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return np.array([]), np.array([])

    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in extensions
    )

    if not image_paths:
        logger.error(f"No images found in {images_dir}")
        return np.array([]), np.array([])

    all_confidences = []
    all_correctness = []
    all_classes = []

    logger.info(f"Running inference on {len(image_paths)} {split} images...")

    for img_path in image_paths:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"

        # Load ground truth bboxes from YOLO labels.
        # DRISHTI is task=detect: each line is "cls xc yc w h" (normalized box),
        # NOT a polygon. Handle both just in case (>4 coords -> polygon).
        gt_boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(coords) == 4:
                        xc, yc, bw, bh = coords
                        x_min, x_max = xc - bw / 2, xc + bw / 2
                        y_min, y_max = yc - bh / 2, yc + bh / 2
                    else:  # polygon fallback
                        xs, ys = coords[0::2], coords[1::2]
                        if not (xs and ys):
                            continue
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                    gt_boxes.append({
                        "class_id": cls_id,
                        "x_min": x_min, "y_min": y_min,
                        "x_max": x_max, "y_max": y_max,
                    })

        # Run inference
        results = model.predict(
            str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False,
        )

        if not results or len(results) == 0:
            continue

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        # Load image dimensions for denormalization
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Match predictions to ground truth
        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i])
            pred_cls = int(result.boxes.cls[i])

            # Get predicted box (xyxy format, pixel coords)
            pred_box = result.boxes.xyxy[i].cpu().numpy()
            pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
            # Normalize
            pred_norm = {
                "x_min": pred_x1 / img_w,
                "y_min": pred_y1 / img_h,
                "x_max": pred_x2 / img_w,
                "y_max": pred_y2 / img_h,
            }

            # Check if this prediction matches any GT box
            is_correct = 0
            best_iou = 0.0

            for gt in gt_boxes:
                if gt["class_id"] != pred_cls:
                    continue

                # Compute IoU
                inter_x1 = max(pred_norm["x_min"], gt["x_min"])
                inter_y1 = max(pred_norm["y_min"], gt["y_min"])
                inter_x2 = min(pred_norm["x_max"], gt["x_max"])
                inter_y2 = min(pred_norm["y_max"], gt["y_max"])

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    pred_area = (
                        (pred_norm["x_max"] - pred_norm["x_min"])
                        * (pred_norm["y_max"] - pred_norm["y_min"])
                    )
                    gt_area = (
                        (gt["x_max"] - gt["x_min"])
                        * (gt["y_max"] - gt["y_min"])
                    )
                    union_area = pred_area + gt_area - inter_area
                    iou = inter_area / (union_area + 1e-10)

                    if iou > best_iou:
                        best_iou = iou

            if best_iou >= iou_threshold:
                is_correct = 1

            all_confidences.append(conf)
            all_correctness.append(is_correct)
            all_classes.append(pred_cls)

    confidences = np.array(all_confidences)
    correctness = np.array(all_correctness)
    classes = np.array(all_classes)

    logger.info(
        f"Collected {len(confidences)} predictions: "
        f"{correctness.sum()} correct, {len(correctness) - correctness.sum()} incorrect"
    )

    return confidences, correctness, classes


CLASS_NAMES = {0: "crab_pot", 1: "submarine_pipeline", 2: "shipwreck", 3: "ghost_net", 4: "mine_cylinder"}


def fit_per_class(confidences, correctness, classes, min_n: int = 25):
    """
    One Platt-scaling logistic per class + a global fallback. Module 2's step-1
    finding: score meaning differs sharply per class, and a single global fit
    doesn't transfer. Classes with too few samples reuse the global calibrator.
    """
    from sklearn.linear_model import LogisticRegression

    def _fit(x, y):
        if len(x) < 10 or len(set(y.tolist())) < 2:
            return None
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        m.fit(np.asarray(x).reshape(-1, 1), y)
        return m

    global_cal = _fit(confidences, correctness)
    per_class = {}
    for c in sorted(set(classes.tolist())):
        mask = classes == c
        cal = _fit(confidences[mask], correctness[mask]) if mask.sum() >= min_n else None
        if cal is not None:
            ece_c = compute_ece(confidences[mask], correctness[mask], cal)
            logger.info(f"  class {c} {CLASS_NAMES.get(c,'?'):18s} n={mask.sum():4d}  ECE={ece_c:.4f}")
            per_class[int(c)] = cal
        else:
            logger.info(f"  class {c} {CLASS_NAMES.get(c,'?'):18s} n={mask.sum():4d}  -> global fallback")
    return {"kind": "per_class", "models": per_class, "fallback": global_cal}


def fit_platt_scaling(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> Any:
    """
    Fit Platt scaling (LogisticRegression) on (confidence -> correct/incorrect) pairs.

    This transforms raw YOLO confidence scores into calibrated probabilities.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if len(confidences) < 10:
        logger.error("Not enough predictions for calibration (need >= 10)")
        return None

    # Reshape for sklearn
    X = confidences.reshape(-1, 1)
    y = correctness

    # Fit logistic regression (Platt scaling)
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    calibrator.fit(X, y)

    # Cross-validated accuracy
    cv_scores = cross_val_score(calibrator, X, y, cv=min(5, len(y)), scoring="accuracy")
    logger.info(f"Calibrator CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Calibration quality: Expected Calibration Error (ECE)
    ece = compute_ece(confidences, correctness, calibrator)
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")

    return calibrator


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    calibrator: Any,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error after Platt scaling.

    ECE measures the gap between predicted probabilities and actual correctness
    rates across confidence bins.
    """
    # Get calibrated probabilities
    cal_probs = calibrator.predict_proba(confidences.reshape(-1, 1))[:, 1]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(cal_probs)

    for i in range(n_bins):
        mask = (cal_probs >= bin_edges[i]) & (cal_probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue

        bin_conf = cal_probs[mask].mean()
        bin_acc = correctness[mask].mean()
        bin_weight = mask.sum() / total

        ece += bin_weight * abs(bin_acc - bin_conf)

    return ece


def save_calibrator(
    calibrator: Any,
    output_path: Path,
    metadata: Dict = None,
):
    """Save calibrator and its metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save calibrator
    with open(output_path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info(f"Calibrator saved to: {output_path}")

    # Save metadata
    if metadata:
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Platt scaling calibrator on validation-set predictions"
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
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split to use for calibration (default: val)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORTED_DIR / "calibrator.pkl",
        help="Output path for calibrator.pkl",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--global-only", action="store_true",
                        help="fit one global calibrator (default: per-class + global fallback)")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if model_path is None:
        model_path = CHECKPOINTS_DIR / "best_yolo_seg.pt"
        if not model_path.exists():
            candidates = list(CHECKPOINTS_DIR.rglob("best.pt"))
            if candidates:
                model_path = candidates[0]
            else:
                logger.error("No trained model found. Run train_yolo_seg.py first.")
                return

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = CONFIGS_DIR / args.data

    # Step 1: Collect predictions
    logger.info("Step 1: Collecting predictions on validation set...")
    confidences, correctness, classes = collect_predictions(
        model_path, data_path, split=args.split, imgsz=args.imgsz
    )

    if len(confidences) == 0:
        logger.error("No predictions collected — cannot calibrate")
        return

    # Step 2: Fit calibrator
    if args.global_only:
        logger.info("Step 2: Fitting ONE global Platt-scaling calibrator...")
        calibrator = fit_platt_scaling(confidences, correctness)
        meta_extra = {"kind": "global"}
    else:
        logger.info("Step 2: Fitting PER-CLASS calibrators (+ global fallback)...")
        calibrator = fit_per_class(confidences, correctness, classes)
        meta_extra = {"kind": "per_class",
                      "classes_fitted": sorted(calibrator["models"].keys())}

    if calibrator is None:
        return

    # Step 3: Save
    logger.info("Step 3: Saving calibrator...")
    save_calibrator(
        calibrator, args.output,
        metadata={
            "model_path": str(model_path), "split": args.split,
            "n_predictions": len(confidences), "n_correct": int(correctness.sum()),
            "timestamp": datetime.now().isoformat(), **meta_extra,
        },
    )

    # Demo: show the mapping (per class if available)
    test_confs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    if isinstance(calibrator, dict):
        for c, m in sorted(calibrator["models"].items()):
            row = "  ".join(f"{r:.2f}->{m.predict_proba([[r]])[0][1]:.2f}" for r in (0.2, 0.4, 0.6, 0.8))
            logger.info(f"  {CLASS_NAMES.get(c,'?'):18s}  {row}")
    else:
        logger.info("\nCalibration mapping (raw -> calibrated):")
        for raw in test_confs:
            logger.info(f"  Raw {raw:.2f} -> {calibrator.predict_proba([[raw]])[0][1]:.2f}")

    logger.info("\nDone. calibrator.pkl loaded by ml/inference/confidence_filter.py")


if __name__ == "__main__":
    main()
