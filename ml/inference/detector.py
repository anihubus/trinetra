"""
Shared inference wrapper -- imported by BOTH backend/detections/tasks.py (server-side)
and edge/onnx_runtime_server.py (edge-side), so serving logic exists in exactly one place.

Loads the exported ONNX model (or .pt during early development) and runs inference on
a single preprocessed tile, returning raw boxes/masks before confidence filtering.

Usage:
    from ml.inference.detector import SonarDetector
    detector = SonarDetector("path/to/model.onnx")  # or .pt
    detections = detector.detect(preprocessed_image)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Detection output schema (matches docs/api_contract.md)
Detection = Dict


class SonarDetector:
    """
    Unified inference wrapper for YOLOv8-seg models.

    Supports both:
    - `.pt` files (via Ultralytics YOLO, for development)
    - `.onnx` files (via onnxruntime, for production/edge)

    The output format matches the API contract exactly:
    {
        "class_label": "net",
        "confidence_raw": 0.87,
        "bbox": [x_min, y_min, x_max, y_max],
        "mask_polygon": [[x1,y1], [x2,y2], ...]
    }
    """

    # Fallback only - the real names are read from the loaded model (model.names).
    # DRISHTI post-pivot taxonomy (configs/drishti.yaml).
    CLASS_NAMES = [
        "crab_pot", "submarine_pipeline", "shipwreck", "ghost_net", "mine_cylinder",
    ]

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to .pt or .onnx model file.
            conf_threshold: Minimum confidence for detections.
            iou_threshold: IoU threshold for NMS.
            imgsz: Input image size.
            device: Device string ("", "cpu", "cuda", "0").
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device

        self._model = None
        self._is_onnx = self.model_path.suffix == ".onnx"
        self._ort_session = None

        self._load_model()

    def _load_model(self):
        """Load the model (lazy, once)."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self._is_onnx:
            self._load_onnx()
        else:
            self._load_pt()

        logger.info(f"Loaded model: {self.model_path}")

    def _load_pt(self):
        """Load PyTorch model via Ultralytics."""
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))

    def _load_onnx(self):
        """
        Load ONNX via onnxruntime only — no torch, no Ultralytics.

        Delegates to edge.edge_infer.EdgeDetector (ORT session + numpy YOLOv8 decode
        + numpy NMS), the same path the AUV runs. Do NOT wrap an .onnx in
        Ultralytics' YOLO(): it attempts GPU IO-binding against a CPU session and
        raises "no data transfer registered for copying tensors from Device...".
        """
        import sys

        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from edge.edge_infer import EdgeDetector

        providers = ["CPUExecutionProvider"]
        if self.device and self.device != "cpu":
            providers = ["CUDAExecutionProvider"] + providers

        self._edge = EdgeDetector(self.model_path, imgsz=self.imgsz, providers=providers)
        self._ort_session = self._edge.sess

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """
        Run inference on a single image.

        Args:
            image: Input image (raw or preprocessed). Can be:
                - A file path string
                - A numpy array (HWC or grayscale)
            conf_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            List of Detection dicts matching API contract format.
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        # ONNX: torch-free path (onnxruntime + numpy), already API-contract shaped
        if self._is_onnx:
            import cv2

            img = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else image
            if img is None:
                raise FileNotFoundError(f"could not read image: {image}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return self._edge.detect(img, conf=conf, iou=iou)

        # Run prediction
        results = self._model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device or None,
            verbose=False,
            retina_masks=True,  # high-res masks
        )

        if not results or len(results) == 0:
            return []

        return self._parse_results(results[0])

    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[List[Detection]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images.
            conf_threshold: Override default confidence threshold.
            iou_threshold: Override default IoU threshold.

        Returns:
            List of detection lists (one per image).
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        results = self._model.predict(
            images,
            conf=conf,
            iou=iou,
            imgsz=self.imgsz,
            device=self.device or None,
            verbose=False,
            retina_masks=True,
        )

        return [self._parse_results(r) for r in results]

    def _parse_results(self, result) -> List[Detection]:
        """
        Parse Ultralytics results into API contract format.
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        masks = result.masks if hasattr(result, "masks") and result.masks is not None else None

        for i in range(len(boxes)):
            # Bounding box (xyxy format, pixel coords)
            bbox = boxes.xyxy[i].cpu().numpy().tolist()

            # Class - prefer the names baked into the loaded model
            cls_id = int(boxes.cls[i])
            model_names = getattr(getattr(self, "_model", None), "names", None)
            if isinstance(model_names, dict) and cls_id in model_names:
                cls_name = model_names[cls_id]
            elif cls_id < len(self.CLASS_NAMES):
                cls_name = self.CLASS_NAMES[cls_id]
            else:
                cls_name = f"class_{cls_id}"

            # Confidence
            conf_raw = float(boxes.conf[i])

            # Mask polygon
            mask_polygon = []
            if masks is not None and i < len(masks):
                mask_data = masks[i]

                # Extract polygon from mask
                if hasattr(mask_data, "xy") and mask_data.xy is not None:
                    # Ultralytics provides xy coordinates directly
                    for segment in mask_data.xy:
                        polygon = [[float(x), float(y)] for x, y in segment]
                        mask_polygon = polygon
                        break
                elif hasattr(mask_data, "data"):
                    # Fall back to extracting contours from binary mask
                    import cv2

                    mask_np = mask_data.data.cpu().numpy().squeeze()
                    if mask_np.ndim == 2:
                        mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(
                            mask_uint8,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            epsilon = 0.005 * cv2.arcLength(largest, True)
                            approx = cv2.approxPolyDP(largest, epsilon, True)
                            mask_polygon = approx.squeeze().tolist()

            detection: Detection = {
                "class_label": cls_name,
                "confidence_raw": round(conf_raw, 4),
                "bbox": [round(v, 2) for v in bbox],
                "mask_polygon": mask_polygon,
            }

            detections.append(detection)

        return detections

    @property
    def class_names(self) -> List[str]:
        """Return the list of class names."""
        return self.CLASS_NAMES.copy()
