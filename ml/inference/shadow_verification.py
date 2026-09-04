"""
Deterministic geometric false-positive filter (no second trained network needed):
compares a candidate's predicted acoustic-shadow length against the shadow length
geometry alone would predict for an object of that apparent height, given sonar
altitude and slant range at that ping. A geometrically inconsistent shadow demotes
the detection's confidence regardless of raw visual response strength.

This is the "highlight-shadow geometry check" from the Skills doc's confidence
scoring & noise-filtering module. It targets rock-cluster false positives specifically:
a rock cluster gives a strong sonar return but its shadow won't match what geometry
predicts for its apparent height, while a real raised debris object's shadow will.

Usage:
    from ml.inference.shadow_verification import ShadowVerifier
    verifier = ShadowVerifier()
    penalty = verifier.compute_penalty(detection, sonar_metadata)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ShadowVerifier:
    """
    Geometric shadow-consistency verifier for sonar detections.

    In side-scan sonar, objects on the seabed cast acoustic shadows.
    The shadow length is determined by:
    - Object height (h_obj)
    - Sonar altitude (h_sonar)
    - Slant range (R) to the object

    Shadow length = h_obj * R / (h_sonar - h_obj)

    A detection whose observed shadow is inconsistent with this geometry
    is likely a false positive (e.g., a rock cluster that produces a
    strong but geometrically incorrect response).
    """

    def __init__(
        self,
        shadow_tolerance: float = 0.4,
        min_confidence_penalty: float = 0.1,
        max_confidence_penalty: float = 0.5,
        min_shadow_pixels: int = 10,
    ):
        """
        Args:
            shadow_tolerance: Fractional tolerance for shadow length mismatch
                (0.4 = ±40% is acceptable for sonar noise/uncertainty).
            min_confidence_penalty: Minimum penalty for shadow inconsistency.
            max_confidence_penalty: Maximum penalty for severe inconsistency.
            min_shadow_pixels: Minimum shadow region size to attempt verification.
        """
        self.shadow_tolerance = shadow_tolerance
        self.min_penalty = min_confidence_penalty
        self.max_penalty = max_confidence_penalty
        self.min_shadow_pixels = min_shadow_pixels

    def compute_expected_shadow_length(
        self,
        object_height: float,
        sonar_altitude: float,
        slant_range: float,
    ) -> float:
        """
        Compute expected acoustic shadow length using sonar geometry.

        Args:
            object_height: Estimated object height in meters.
            sonar_altitude: Sonar altitude above seabed in meters.
            slant_range: Slant range from sonar to object in meters.

        Returns:
            Expected shadow length in meters.
        """
        if sonar_altitude <= object_height:
            # Object taller than sonar altitude — shadow extends to infinity
            return float("inf")

        if object_height <= 0 or sonar_altitude <= 0:
            return 0.0

        # Ground range from sonar nadir to object
        ground_range = math.sqrt(max(slant_range ** 2 - sonar_altitude ** 2, 0))

        # Shadow length from similar triangles
        # shadow_length / (ground_range + shadow_length) = object_height / sonar_altitude
        shadow_length = (object_height * ground_range) / (sonar_altitude - object_height)

        return max(shadow_length, 0.0)

    def estimate_object_height_from_highlight(
        self,
        bbox: List[float],
        image_height: int,
        sonar_altitude: float,
        max_range: float,
    ) -> float:
        """
        Estimate object height from its sonar highlight extent.

        The across-track extent of the bright highlight region is related
        to the object's height via the sonar geometry.

        Args:
            bbox: [x_min, y_min, x_max, y_max] in pixel coordinates.
            image_height: Total image height in pixels.
            sonar_altitude: Sonar altitude in meters.
            max_range: Maximum slant range in meters.

        Returns:
            Estimated object height in meters.
        """
        # The highlight extent in the range direction
        highlight_extent_px = bbox[3] - bbox[1]  # y_max - y_min
        highlight_fraction = highlight_extent_px / max(image_height, 1)

        # Convert to range extent in meters
        range_extent = highlight_fraction * max_range

        # Rough height estimate from range extent
        # For raised objects, highlight extent ≈ 2 * h_obj * sin(grazing_angle)
        y_center = (bbox[1] + bbox[3]) / 2
        range_fraction = y_center / max(image_height, 1)
        slant_range = range_fraction * max_range

        if slant_range <= sonar_altitude:
            grazing_angle = math.pi / 2
        else:
            grazing_angle = math.asin(sonar_altitude / max(slant_range, 0.001))

        sin_graze = math.sin(grazing_angle)
        if sin_graze > 0.01:
            estimated_height = range_extent / (2 * sin_graze)
        else:
            estimated_height = range_extent * 0.5

        # Clamp to reasonable range (0-5m for seabed debris)
        return min(max(estimated_height, 0.01), 5.0)

    def extract_shadow_region(
        self,
        image: np.ndarray,
        bbox: List[float],
        mask_polygon: Optional[List] = None,
        shadow_search_factor: float = 2.0,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract the acoustic shadow region below/behind a detected object.

        In side-scan sonar, the shadow is cast in the far-range direction
        (away from the sonar transducer, which is typically the bottom
        of the image or the side away from nadir).

        Args:
            image: Grayscale sonar image (uint8).
            bbox: [x_min, y_min, x_max, y_max] in pixel coordinates.
            mask_polygon: Optional mask polygon for more precise extraction.
            shadow_search_factor: How far below the object to search.

        Returns:
            (shadow_mask, shadow_length_pixels) or (None, 0) if no shadow found.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        obj_width = x2 - x1
        obj_height = y2 - y1

        # Shadow search region: below the object in the range direction
        shadow_y_start = y2
        shadow_y_end = min(y2 + int(obj_height * shadow_search_factor), h)
        shadow_x_start = max(x1 - 5, 0)
        shadow_x_end = min(x2 + 5, w)

        if shadow_y_end <= shadow_y_start or shadow_x_end <= shadow_x_start:
            return None, 0

        # Extract shadow search region
        shadow_roi = image[shadow_y_start:shadow_y_end, shadow_x_start:shadow_x_end]

        if shadow_roi.size == 0:
            return None, 0

        # Shadow detection: shadow pixels are significantly darker than surroundings
        # Use Otsu's thresholding on the ROI
        roi_mean = shadow_roi.mean()
        shadow_threshold = max(roi_mean * 0.4, 10)  # shadow is < 40% of mean intensity

        shadow_mask = (shadow_roi < shadow_threshold).astype(np.uint8)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

        # Measure shadow extent (length in range direction)
        shadow_pixels = shadow_mask.sum()

        if shadow_pixels < self.min_shadow_pixels:
            return None, 0

        # Shadow length = extent in the y-direction (range direction)
        shadow_cols = shadow_mask.any(axis=1)
        if shadow_cols.any():
            shadow_rows = np.where(shadow_cols)[0]
            shadow_length = shadow_rows[-1] - shadow_rows[0] + 1
        else:
            shadow_length = 0

        return shadow_mask, shadow_length

    def compute_penalty(
        self,
        detection: Dict,
        sonar_metadata: Dict,
        image: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute confidence penalty based on shadow geometry consistency.

        Args:
            detection: Detection dict with 'bbox', 'mask_polygon', 'confidence_raw'.
            sonar_metadata: Dict with 'altitude', 'max_range', 'image_height', 'image_width'.
            image: Optional grayscale sonar image for shadow extraction.

        Returns:
            Confidence penalty in [0, max_penalty]. 0 = no penalty (consistent shadow).
        """
        altitude = sonar_metadata.get("altitude")
        max_range = sonar_metadata.get("max_range")
        img_h = sonar_metadata.get("image_height", 640)
        img_w = sonar_metadata.get("image_width", 640)

        # Cannot verify without geometry metadata
        if altitude is None or max_range is None:
            return 0.0

        bbox = detection.get("bbox", [])
        if len(bbox) != 4:
            return 0.0

        # Step 1: Estimate object height from highlight
        estimated_height = self.estimate_object_height_from_highlight(
            bbox, img_h, altitude, max_range
        )

        # Step 2: Compute expected shadow length
        y_center = (bbox[1] + bbox[3]) / 2
        range_fraction = y_center / max(img_h, 1)
        slant_range = range_fraction * max_range

        expected_shadow_m = self.compute_expected_shadow_length(
            estimated_height, altitude, slant_range
        )

        # Convert expected shadow to pixels
        meters_per_pixel = max_range / max(img_h, 1)
        expected_shadow_px = expected_shadow_m / max(meters_per_pixel, 0.001)

        # Step 3: Measure actual shadow (if image provided)
        if image is not None:
            _, observed_shadow_px = self.extract_shadow_region(image, bbox)

            if observed_shadow_px < self.min_shadow_pixels:
                # No shadow detected — suspicious for a raised object
                # Small penalty: could be a flat debris item
                return self.min_penalty

            # Step 4: Compare expected vs. observed
            ratio = observed_shadow_px / max(expected_shadow_px, 1.0)

            # Consistent shadow: ratio near 1.0
            deviation = abs(ratio - 1.0)

            if deviation <= self.shadow_tolerance:
                # Shadow is geometrically consistent — no penalty
                return 0.0
            else:
                # Shadow is inconsistent — penalty scales with deviation
                excess = deviation - self.shadow_tolerance
                penalty = min(
                    self.min_penalty + excess * (self.max_penalty - self.min_penalty),
                    self.max_penalty,
                )
                return penalty

        # No image available — cannot verify, no penalty
        return 0.0

    def verify_detections(
        self,
        detections: List[Dict],
        sonar_metadata: Dict,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Apply shadow verification to a list of detections.

        Adds 'shadow_penalty' and adjusts 'confidence_raw' for each detection.

        Args:
            detections: List of detection dicts.
            sonar_metadata: Sonar geometry metadata.
            image: Optional grayscale sonar image.

        Returns:
            Detections with updated confidence and shadow_penalty fields.
        """
        verified = []

        for det in detections:
            penalty = self.compute_penalty(det, sonar_metadata, image)

            det_copy = det.copy()
            det_copy["shadow_penalty"] = round(penalty, 4)
            det_copy["confidence_raw"] = round(
                max(det["confidence_raw"] - penalty, 0.0), 4
            )

            verified.append(det_copy)

        # Log summary
        if detections:
            n_penalized = sum(1 for d in verified if d["shadow_penalty"] > 0)
            if n_penalized > 0:
                logger.debug(
                    f"Shadow verification: {n_penalized}/{len(detections)} "
                    f"detections penalized"
                )

        return verified
