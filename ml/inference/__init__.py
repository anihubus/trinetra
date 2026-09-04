"""
DRISHTI ML Inference Package

Public API: run_inference(image, sonar_metadata) -> List[Detection]

This chains:
  preprocess -> detect -> confidence_filter (NMS + Platt + shadow check)

Imported by:
  - backend/detections/tasks.py (server-side Celery worker)
  - edge/onnx_runtime_server.py (edge-side microservice)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .preprocess import preprocess_tile
from .detector import SonarDetector, Detection
from .confidence_filter import ConfidenceFilter
from .shadow_verification import ShadowVerifier

logger = logging.getLogger(__name__)

# Package-level defaults
_ML_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _ML_DIR / "models" / "exported" / "best_yolo_seg.onnx"
_DEFAULT_MODEL_PT = _ML_DIR / "models" / "checkpoints" / "best_yolo_seg.pt"
_DEFAULT_CALIBRATOR = _ML_DIR / "models" / "exported" / "calibrator.pkl"

# Singleton instances (lazy-initialized)
_detector: Optional[SonarDetector] = None
_filter: Optional[ConfidenceFilter] = None


def _get_detector(model_path: Optional[str] = None) -> SonarDetector:
    """Get or create the detector singleton."""
    global _detector
    if _detector is None:
        if model_path is None:
            # Prefer ONNX, fall back to .pt
            if _DEFAULT_MODEL.exists():
                model_path = str(_DEFAULT_MODEL)
            elif _DEFAULT_MODEL_PT.exists():
                model_path = str(_DEFAULT_MODEL_PT)
            else:
                raise FileNotFoundError(
                    "No model found. Run train_yolo_seg.py + export_onnx.py first.\n"
                    f"Expected: {_DEFAULT_MODEL} or {_DEFAULT_MODEL_PT}"
                )
        _detector = SonarDetector(model_path)
    return _detector


def _get_filter(calibrator_path: Optional[str] = None) -> ConfidenceFilter:
    """Get or create the confidence filter singleton."""
    global _filter
    if _filter is None:
        if calibrator_path is None and _DEFAULT_CALIBRATOR.exists():
            calibrator_path = str(_DEFAULT_CALIBRATOR)
        _filter = ConfidenceFilter(calibrator_path=calibrator_path)
    return _filter


def run_inference(
    image: Union[str, Path, np.ndarray],
    sonar_metadata: Optional[Dict] = None,
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    preprocess: bool = True,
) -> List[Detection]:
    """
    Run the full DRISHTI inference pipeline on a single sonar image.

    Pipeline: preprocess → detect → NMS → Platt calibration → shadow verification

    Args:
        image: Input sonar image. Can be:
            - A file path (str or Path)
            - A numpy array (grayscale or BGR)
        sonar_metadata: Optional dict with sonar geometry for shadow verification:
            {
                "altitude": float,    # sonar altitude in meters
                "max_range": float,   # max slant range in meters
                "image_height": int,  # image height in pixels
                "image_width": int,   # image width in pixels
            }
        model_path: Override default model path.
        calibrator_path: Override default calibrator path.
        conf_threshold: Minimum confidence threshold for raw detections.
        preprocess: Whether to apply sonar preprocessing. Set False if
            the image is already preprocessed.

    Returns:
        List of Detection dicts matching the API contract:
        [
            {
                "class_label": "net",
                "confidence_raw": 0.87,
                "confidence_score": 85.0,  # calibrated, 0-100 scale
                "bbox": [x_min, y_min, x_max, y_max],
                "mask_polygon": [[x1,y1], [x2,y2], ...],
            },
            ...
        ]
    """
    # Load image if path
    raw_image = None
    if isinstance(image, (str, Path)):
        import cv2

        raw_image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if raw_image is None:
            raise ValueError(f"Could not read image: {image}")
        input_for_model = str(image)
    else:
        raw_image = image.copy()
        input_for_model = image

    # Preprocess if requested
    if preprocess and isinstance(input_for_model, np.ndarray):
        altitude = sonar_metadata.get("altitude") if sonar_metadata else None
        max_range = sonar_metadata.get("max_range") if sonar_metadata else None
        input_for_model = preprocess_tile(
            input_for_model,
            sonar_altitude=altitude,
            max_range=max_range,
        )

    # Detect
    detector = _get_detector(model_path)
    raw_detections = detector.detect(
        input_for_model, conf_threshold=conf_threshold
    )

    if not raw_detections:
        return []

    # Filter (NMS + calibration + shadow check)
    filter_pipeline = _get_filter(calibrator_path)
    filtered = filter_pipeline.filter(
        raw_detections,
        sonar_metadata=sonar_metadata,
        image=raw_image,
    )

    return filtered


def run_inference_batch(
    images: List[Union[str, Path, np.ndarray]],
    sonar_metadata: Optional[Dict] = None,
    **kwargs,
) -> List[List[Detection]]:
    """
    Run inference on a batch of images.

    Args:
        images: List of input images.
        sonar_metadata: Shared sonar metadata (same for all images in batch).
        **kwargs: Additional arguments passed to run_inference.

    Returns:
        List of detection lists (one per image).
    """
    return [run_inference(img, sonar_metadata, **kwargs) for img in images]


# ---- Initialization helpers -----------------------------------------------

def initialize(
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
) -> None:
    """
    Pre-initialize detector and filter (warm up).

    Call this at application startup to avoid lazy-init latency
    on the first inference request.
    """
    _get_detector(model_path)
    _get_filter(calibrator_path)
    logger.info("DRISHTI inference pipeline initialized")


def reset() -> None:
    """Reset singletons (for testing or model reloading)."""
    global _detector, _filter
    _detector = None
    _filter = None


__all__ = [
    "run_inference",
    "run_inference_batch",
    "initialize",
    "reset",
    "SonarDetector",
    "ConfidenceFilter",
    "ShadowVerifier",
    "preprocess_tile",
    "Detection",
]
