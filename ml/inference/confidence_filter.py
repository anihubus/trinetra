"""
Post-detection pipeline, in order:
  1. NMS (remove duplicate overlapping boxes on the same object)
  2. Platt-scaling calibration (calibrator.pkl from calibrate_confidence.py)
  3. shadow_verification.py geometric check (demotes rock-cluster false positives)
Outputs the final calibrated 0-100% confidence score per surviving detection.

Usage:
    from ml.inference.confidence_filter import ConfidenceFilter
    cf = ConfidenceFilter(calibrator_path="models/exported/calibrator.pkl")
    filtered = cf.filter(raw_detections, sonar_metadata, image)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .shadow_verification import ShadowVerifier

logger = logging.getLogger(__name__)


class ConfidenceFilter:
    """
    Post-detection confidence filtering pipeline.

    Chains three stages:
    1. NMS — removes duplicate/overlapping boxes
    2. Platt scaling — transforms raw confidence into calibrated probability
    3. Shadow verification — geometric check demotes rock-cluster false positives

    The output is the final 0–100% confidence score ready for the API response.
    """

    # Per-class RAW-confidence keep thresholds. From the Module 2 measurement run:
    # false positives are overwhelmingly low-scoring, and each class needs its own
    # cut-off (pipeline separates cleanly, shipwreck/cylinder need a higher bar,
    # crab_pot is dropped in the nc=4 model so its threshold is moot).
    DEFAULT_PER_CLASS_CONF = {
        "submarine_pipeline": 0.25,
        "shipwreck": 0.40,
        "mine_cylinder": 0.40,
        "ghost_net": 0.30,
        "crab_pot": 0.99,          # effectively off
    }

    def __init__(
        self,
        calibrator_path: Optional[str] = None,
        nms_iou_threshold: float = 0.45,
        min_confidence: float = 0.15,
        per_class_conf: Optional[Dict] = None,
        use_shadow_check: bool = True,
        shadow_verifier: Optional[ShadowVerifier] = None,
    ):
        """
        Args:
            calibrator_path: Path to calibrator.pkl (from calibrate_confidence.py).
                If None, raw confidence is used without Platt scaling.
            nms_iou_threshold: IoU threshold for NMS deduplication.
            min_confidence: Fallback minimum calibrated confidence (used when a
                class has no per-class threshold).
            per_class_conf: {class_label: raw-confidence keep threshold}. Applied
                to the RAW score before calibration. Defaults to DEFAULT_PER_CLASS_CONF.
            shadow_verifier: Optional ShadowVerifier instance. If None, creates one.
        """
        self.nms_iou_threshold = nms_iou_threshold
        self.min_confidence = min_confidence
        self.per_class_conf = dict(self.DEFAULT_PER_CLASS_CONF)
        if per_class_conf:
            self.per_class_conf.update(per_class_conf)
        # The geometric shadow check only helps on FULL transect images with real
        # per-ping altitude/slant range (via Module 3). On cropped 640 px tiles,
        # or with a single assumed geometry, it removes about as many true
        # positives as false ones - keep it off for tile inference.
        self.use_shadow_check = use_shadow_check
        self.shadow_verifier = shadow_verifier or ShadowVerifier()

        # Load Platt-scaling calibrator
        self._calibrator = None
        if calibrator_path:
            self._load_calibrator(calibrator_path)

    def _load_calibrator(self, path: str):
        """Load the sklearn LogisticRegression calibrator from pickle."""
        cal_path = Path(path)
        if not cal_path.exists():
            logger.warning(
                f"Calibrator not found: {cal_path}. "
                f"Using raw confidence scores (not calibrated)."
            )
            return

        try:
            with open(cal_path, "rb") as f:
                self._calibrator = pickle.load(f)
            logger.info(f"Loaded confidence calibrator: {cal_path}")
        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            self._calibrator = None

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-10)

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        YOLO already applies NMS internally, but we run a second pass here
        in case detections come from multiple tiles or model runs.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (descending)
        sorted_dets = sorted(
            detections, key=lambda d: d.get("confidence_raw", 0), reverse=True
        )

        keep = []
        suppressed = set()

        for i, det_i in enumerate(sorted_dets):
            if i in suppressed:
                continue

            keep.append(det_i)
            bbox_i = det_i.get("bbox", [])
            if len(bbox_i) != 4:
                continue

            for j in range(i + 1, len(sorted_dets)):
                if j in suppressed:
                    continue

                bbox_j = sorted_dets[j].get("bbox", [])
                if len(bbox_j) != 4:
                    continue

                # Only suppress same-class overlaps
                if det_i.get("class_label") != sorted_dets[j].get("class_label"):
                    continue

                iou = self._compute_iou(bbox_i, bbox_j)
                if iou > self.nms_iou_threshold:
                    suppressed.add(j)

        logger.debug(f"NMS: {len(detections)} -> {len(keep)} detections")
        return keep

    # class_label -> id, for the per-class calibrator lookup
    _CLASS_ID = {"crab_pot": 0, "submarine_pipeline": 1, "shipwreck": 2,
                 "ghost_net": 3, "mine_cylinder": 4}

    def calibrate_confidence(self, raw_confidence: float, class_label: Optional[str] = None) -> float:
        """
        Map a raw score to a calibrated probability. Supports both calibrator
        formats: a single sklearn model, or {"kind":"per_class", "models":{id:model},
        "fallback":model}. No calibrator -> raw score unchanged.
        """
        cal = self._calibrator
        if cal is None:
            return raw_confidence
        try:
            if isinstance(cal, dict) and cal.get("kind") == "per_class":
                cid = self._CLASS_ID.get(class_label)
                model = cal["models"].get(cid) or cal.get("fallback")
                if model is None:
                    return raw_confidence
                return float(model.predict_proba([[raw_confidence]])[0][1])
            return float(cal.predict_proba([[raw_confidence]])[0][1])
        except Exception as e:
            logger.warning(f"Calibration failed: {e}, using raw confidence")
            return raw_confidence

    def filter(
        self,
        detections: List[Dict],
        sonar_metadata: Optional[Dict] = None,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Full post-detection filtering pipeline.

        1. NMS deduplication
        2. Platt-scaling calibration
        3. Shadow geometry verification
        4. Minimum confidence threshold

        Args:
            detections: Raw detections from SonarDetector.detect().
            sonar_metadata: Dict with sonar geometry info:
                - altitude: float (meters)
                - max_range: float (meters)
                - image_height: int (pixels)
                - image_width: int (pixels)
            image: Optional grayscale sonar image for shadow verification.

        Returns:
            Filtered detections with calibrated 0–100% confidence_score.
        """
        if not detections:
            return []

        # Stage 0: per-class raw-confidence gate (the main false-positive filter -
        # FPs sit well below these thresholds, see the Module 2 measurement run)
        gated = []
        for det in detections:
            thr = self.per_class_conf.get(det.get("class_label"), 0.0)
            if det.get("confidence_raw", 0.0) >= thr:
                gated.append(det)
        detections = gated
        if not detections:
            return []

        # Stage 1: NMS
        nms_filtered = self.apply_nms(detections)

        # Stage 2: Platt-scaling calibration (per-class where available)
        for det in nms_filtered:
            raw_conf = det.get("confidence_raw", 0)
            cal_conf = self.calibrate_confidence(raw_conf, det.get("class_label"))
            det["confidence_calibrated"] = round(cal_conf, 4)

        # Stage 3: Shadow geometry verification (only with real transect geometry)
        if sonar_metadata and self.use_shadow_check:
            nms_filtered = self.shadow_verifier.verify_detections(
                nms_filtered, sonar_metadata, image
            )

            # Recalibrate after shadow penalty
            for det in nms_filtered:
                if det.get("shadow_penalty", 0) > 0:
                    # Re-run calibration on the penalized raw confidence
                    adjusted_raw = det["confidence_raw"]
                    det["confidence_calibrated"] = round(
                        self.calibrate_confidence(adjusted_raw, det.get("class_label")), 4
                    )

        # Stage 4: Convert to 0-100% score and filter by threshold
        results = []
        for det in nms_filtered:
            cal_conf = det.get("confidence_calibrated", det.get("confidence_raw", 0))

            if cal_conf < self.min_confidence:
                continue

            # Final API-ready detection
            det["confidence_score"] = round(cal_conf * 100, 1)  # 0-100 scale
            results.append(det)

        logger.debug(
            f"Filter pipeline: {len(detections)} raw -> "
            f"{len(nms_filtered)} after NMS -> {len(results)} after threshold"
        )

        return results


# ---- Module-level convenience function ------------------------------------

_default_filter = None


def get_default_filter(calibrator_path: Optional[str] = None) -> ConfidenceFilter:
    """Get or create the default ConfidenceFilter singleton."""
    global _default_filter
    if _default_filter is None:
        if calibrator_path is None:
            default_cal = Path(__file__).parent.parent / "models" / "exported" / "calibrator.pkl"
            calibrator_path = str(default_cal) if default_cal.exists() else None
        _default_filter = ConfidenceFilter(calibrator_path=calibrator_path)
    return _default_filter
