"""
Shared preprocessing pipeline (used identically at train time and inference time
via ml/inference/preprocess.py, so train/serve never drift apart):
  slant-range correction -> motion compensation -> speckle denoise -> resample/normalize

This module is imported by:
  - ml/scripts/convert_to_yolo_format.py (batch preprocessing before training)
  - ml/inference/preprocess.py (single-tile inference preprocessing)
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- Constants -----------------------------------------------------------

DEFAULT_TARGET_SIZE = (640, 640)   # YOLOv8 input size
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
LEE_FILTER_SIZE = 7                # kernel size for Lee speckle filter


# ---- Core preprocessing functions ----------------------------------------

def lee_filter(image: np.ndarray, size: int = LEE_FILTER_SIZE) -> np.ndarray:
    """
    Lee speckle filter for sonar imagery.

    Reduces multiplicative speckle noise while preserving edges by adapting
    the filter response based on local statistics (mean, variance).

    Args:
        image: Input grayscale image (float32 or uint8).
        size: Filter kernel size (must be odd).

    Returns:
        Filtered image (same dtype as input).
    """
    img = image.astype(np.float64)

    # Local statistics
    local_mean = cv2.blur(img, (size, size))
    local_sq_mean = cv2.blur(img ** 2, (size, size))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # numerical safety

    # Estimate noise variance from the whole image
    overall_var = np.var(img)
    if overall_var == 0:
        return image  # flat image, nothing to filter

    # Weighting factor
    weight = local_var / (local_var + overall_var + 1e-10)

    # Filtered result
    result = local_mean + weight * (img - local_mean)

    if image.dtype == np.uint8:
        return np.clip(result, 0, 255).astype(np.uint8)
    return result.astype(image.dtype)


def slant_range_correction(
    image: np.ndarray,
    sonar_altitude: Optional[float] = None,
    max_range: Optional[float] = None,
) -> np.ndarray:
    """
    Time-varying gain (TVG) correction for slant-range geometry.

    Side-scan sonar images have decreasing intensity with range due to
    spreading loss and absorption. This applies a range-dependent gain
    to compensate.

    For FLS tank data (Watertank), this is a no-op since the geometry
    is controlled. For real SSS data, applies R^2 spreading correction.

    Args:
        image: Grayscale sonar image.
        sonar_altitude: Sonar altitude in meters (optional).
        max_range: Maximum slant range in meters (optional).

    Returns:
        Range-corrected image.
    """
    if sonar_altitude is None or max_range is None:
        # No geometry metadata — apply a mild linear ramp as fallback
        h, w = image.shape[:2]
        img_float = image.astype(np.float64)

        # Create a linear gain ramp across columns (range axis for SSS)
        # Columns near nadir (center) get less gain, far range gets more
        ramp = np.linspace(1.0, 2.0, w).reshape(1, -1)
        corrected = img_float * ramp

        if image.dtype == np.uint8:
            return np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected.astype(image.dtype)

    # Full TVG correction with known geometry
    h, w = image.shape[:2]
    img_float = image.astype(np.float64)

    # Slant range per column
    col_indices = np.arange(w, dtype=np.float64)
    ground_range = (col_indices / w) * max_range
    slant_range = np.sqrt(ground_range ** 2 + sonar_altitude ** 2)

    # Spreading loss compensation: gain proportional to R^2
    ref_range = sonar_altitude  # normalize to nadir
    gain = (slant_range / ref_range) ** 2
    gain = gain.reshape(1, -1)

    # Cap gain to avoid blowing up noise at far range
    gain = np.clip(gain, 1.0, 10.0)

    corrected = img_float * gain

    if image.dtype == np.uint8:
        return np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected.astype(image.dtype)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: Tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Contrast-limited adaptive histogram equalization.

    Enhances local contrast in sonar images without amplifying noise
    in uniform regions (unlike standard histogram equalization).

    Args:
        image: Grayscale uint8 image.
        clip_limit: Contrast limiting threshold.
        tile_grid: Grid size for local histograms.

    Returns:
        CLAHE-enhanced image (uint8).
    """
    if image.dtype != np.uint8:
        img = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def resize_and_normalize(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    normalize: bool = True,
) -> np.ndarray:
    """
    Resize to target dimensions and optionally normalize to [0, 1] float32.

    Uses INTER_AREA for downsampling (anti-aliased) and INTER_LINEAR for
    upsampling, matching YOLO's default behavior.

    Args:
        image: Input grayscale image.
        target_size: (width, height) target dimensions.
        normalize: If True, output is float32 in [0, 1].

    Returns:
        Resized (and optionally normalized) image.
    """
    h, w = image.shape[:2]
    tw, th = target_size

    interp = cv2.INTER_AREA if (h > th or w > tw) else cv2.INTER_LINEAR
    resized = cv2.resize(image, (tw, th), interpolation=interp)

    if normalize:
        return resized.astype(np.float32) / 255.0

    return resized


def despeckle_clahe(image: np.ndarray) -> np.ndarray:
    """
    The tile-level preprocessing used for BOTH training data and serve-time
    inference: Lee speckle filter + CLAHE, at the image's native resolution.

    Slant-range correction is deliberately NOT applied here - a cropped tile has
    no reliable swath geometry. That correction runs on full transects where the
    ping altitude/range are known (see coordinate_projection / the survey runner).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8) if image.max() > 1.0 \
            else (image * 255).astype(np.uint8)
    out = lee_filter(image, size=LEE_FILTER_SIZE)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return apply_clahe(out)


def preprocess_tile(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    sonar_altitude: Optional[float] = None,
    max_range: Optional[float] = None,
    normalize: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single sonar tile.

    Order: grayscale -> slant-range correction -> speckle denoise ->
           CLAHE contrast enhancement -> resize + normalize

    Args:
        image: Input image (grayscale or BGR).
        target_size: Output (width, height).
        sonar_altitude: Altitude in meters (for TVG correction).
        max_range: Max slant range in meters.
        normalize: Output float32 [0,1] if True, else uint8.
        denoise: Apply Lee speckle filter.
        enhance_contrast: Apply CLAHE.

    Returns:
        Preprocessed image ready for model input.
    """
    # Ensure grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        gray = image.squeeze(axis=2)
    else:
        gray = image.copy()

    # Ensure uint8 for processing
    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 1. Slant-range correction (TVG)
    corrected = slant_range_correction(gray, sonar_altitude, max_range)

    # 2. Speckle denoise (Lee filter)
    if denoise:
        corrected = lee_filter(corrected, size=LEE_FILTER_SIZE)
        if corrected.dtype != np.uint8:
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # 3. CLAHE contrast enhancement
    if enhance_contrast:
        corrected = apply_clahe(corrected)

    # 4. Resize + normalize
    result = resize_and_normalize(corrected, target_size, normalize=normalize)

    return result


def preprocess_batch(
    input_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    normalize: bool = False,  # batch output stays uint8 for compatibility
) -> int:
    """
    Batch-preprocess all images in a directory.

    Args:
        input_dir: Directory containing raw sonar images.
        output_dir: Directory for preprocessed output.
        target_size: Output dimensions.
        extensions: Accepted file extensions.
        normalize: If False, output as uint8 PNG.

    Returns:
        Number of images processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in extensions
    )

    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return 0

    logger.info(f"Preprocessing {len(image_paths)} images from {input_dir}")
    count = 0

    for img_path in image_paths:
        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Could not read: {img_path}")
                continue

            processed = preprocess_tile(
                image, target_size=target_size, normalize=normalize
            )

            # Save as PNG (lossless)
            out_path = output_dir / img_path.with_suffix(".png").name
            if normalize:
                # Convert back to uint8 for saving
                save_img = (processed * 255).astype(np.uint8)
            else:
                save_img = processed

            cv2.imwrite(str(out_path), save_img)
            count += 1

        except Exception as e:
            logger.error(f"Failed to preprocess {img_path}: {e}")

    logger.info(f"Preprocessed {count}/{len(image_paths)} images -> {output_dir}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw sonar imagery for DRISHTI training"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw sonar images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for preprocessed output",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("W", "H"),
        help="Target width and height (default: 640 640)",
    )
    args = parser.parse_args()

    count = preprocess_batch(
        args.input_dir,
        args.output_dir,
        target_size=tuple(args.size),
    )
    logger.info(f"Done — {count} images preprocessed.")


if __name__ == "__main__":
    main()
