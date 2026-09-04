"""
Shared preprocessing pipeline for inference (thin wrapper around
ml/scripts/preprocess_sonar.py to ensure train/serve never drift apart).

Usage:
    from ml.inference.preprocess import preprocess_tile
    processed = preprocess_tile(raw_image, target_size=(640, 640))
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Add ml/scripts to path so we can import the shared pipeline
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from preprocess_sonar import (
    preprocess_tile as _preprocess_tile,
    lee_filter,
    slant_range_correction,
    apply_clahe,
    resize_and_normalize,
    despeckle_clahe,          # the exact filter the preprocessed model was trained with
    DEFAULT_TARGET_SIZE,
)


def preprocess_tile(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    sonar_altitude: Optional[float] = None,
    max_range: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess a single sonar tile for model inference.

    This is the SAME pipeline used during training (via preprocess_sonar.py),
    ensuring train/serve consistency. Any change to preprocessing must go
    through preprocess_sonar.py and will automatically propagate here.

    Args:
        image: Raw sonar image (grayscale or BGR, any dtype).
        target_size: Output (width, height) — must match training (640, 640).
        sonar_altitude: Sonar altitude in meters (for TVG correction).
        max_range: Maximum slant range in meters.
        normalize: If True, output is float32 in [0, 1].

    Returns:
        Preprocessed image ready for model input.
    """
    return _preprocess_tile(
        image,
        target_size=target_size,
        sonar_altitude=sonar_altitude,
        max_range=max_range,
        normalize=normalize,
    )


def preprocess_for_yolo(
    image: np.ndarray,
    target_size: int = 640,
    sonar_altitude: Optional[float] = None,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """
    Preprocess and format for YOLO inference input.

    Returns a (1, 3, H, W) float32 tensor suitable for ONNX or YOLO.predict().
    YOLO expects 3-channel input even for grayscale — we replicate the channel.

    Args:
        image: Raw sonar image.
        target_size: Square image size.
        sonar_altitude: Sonar altitude in meters.
        max_range: Max slant range in meters.

    Returns:
        (1, 3, H, W) float32 numpy array in [0, 1].
    """
    # Preprocess to (H, W) float32 [0, 1]
    processed = preprocess_tile(
        image,
        target_size=(target_size, target_size),
        sonar_altitude=sonar_altitude,
        max_range=max_range,
        normalize=True,
    )

    # Replicate grayscale to 3 channels for YOLO
    if processed.ndim == 2:
        processed = np.stack([processed] * 3, axis=0)  # (3, H, W)
    elif processed.ndim == 3 and processed.shape[2] == 1:
        processed = np.concatenate([processed] * 3, axis=2).transpose(2, 0, 1)
    elif processed.ndim == 3 and processed.shape[2] == 3:
        processed = processed.transpose(2, 0, 1)  # HWC -> CHW

    # Add batch dimension
    return np.expand_dims(processed, axis=0).astype(np.float32)
