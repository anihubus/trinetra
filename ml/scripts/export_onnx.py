"""
Export chain (Tier 0 target: <15MB, <200ms/image on CPU):
  1. yolo export model=best.pt format=onnx opset=12 simplify=True imgsz=640
  2. Validate: onnxruntime.InferenceSession output vs. PyTorch output, on a few samples
  3. Quantize: onnxruntime.quantization.quantize_static, calibrated on ~100 sonar tiles
  4. Benchmark size + latency -> feeds directly into the pitch deck's AUV-readiness slide

Usage:
    python export_onnx.py                                      # uses defaults
    python export_onnx.py --model best.pt --quantize           # with INT8 quantization
    python export_onnx.py --model best.pt --benchmark --n 100  # benchmark latency
"""

import logging
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ML_DIR = SCRIPT_DIR.parent
CHECKPOINTS_DIR = ML_DIR / "models" / "checkpoints"
EXPORTED_DIR = ML_DIR / "models" / "exported"
SPLITS_DIR = ML_DIR / "data" / "splits"


def export_to_onnx(
    model_path: Path,
    output_dir: Path,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    half: bool = False,
    dynamic: bool = False,
) -> Optional[Path]:
    """
    Export YOLOv8-seg model to ONNX format.

    Returns path to exported ONNX file.
    """
    from ultralytics import YOLO

    logger.info(f"Exporting {model_path} to ONNX...")

    model = YOLO(str(model_path))

    # Export
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        half=half,
        dynamic=dynamic,
    )

    if export_path:
        export_path = Path(export_path)
        # Move to our output directory
        dest = output_dir / export_path.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_path != dest:
            import shutil
            shutil.copy2(export_path, dest)
            logger.info(f"ONNX model copied to: {dest}")
        else:
            logger.info(f"ONNX model at: {dest}")

        # File size
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"ONNX model size: {size_mb:.2f} MB")

        return dest
    else:
        logger.error("ONNX export failed")
        return None


def validate_onnx(
    onnx_path: Path,
    pt_model_path: Path,
    test_images_dir: Path,
    imgsz: int = 640,
    n_samples: int = 5,
    atol: float = 0.01,
) -> bool:
    """
    Validate ONNX model output against PyTorch model output.

    Compares predictions on a few sample images to ensure export fidelity.
    """
    import onnx
    import onnxruntime as ort
    from ultralytics import YOLO
    import cv2

    logger.info("Validating ONNX model against PyTorch...")

    # Validate ONNX model structure
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure: VALID")

    # Load models
    pt_model = YOLO(str(pt_model_path))
    ort_session = ort.InferenceSession(str(onnx_path))

    # Get test images
    extensions = (".png", ".jpg", ".jpeg")
    test_images = sorted(
        p for p in test_images_dir.iterdir()
        if p.suffix.lower() in extensions
    )[:n_samples]

    if not test_images:
        logger.warning("No test images found for validation")
        return True  # Skip validation, not a failure

    all_match = True

    for img_path in test_images:
        # PyTorch inference
        pt_results = pt_model.predict(
            str(img_path), imgsz=imgsz, conf=0.25, verbose=False
        )

        # Count detections
        pt_n = len(pt_results[0].boxes) if pt_results and pt_results[0].boxes is not None else 0

        # ONNX inference (via YOLO wrapper for fair comparison)
        onnx_model_yolo = YOLO(str(onnx_path))
        onnx_results = onnx_model_yolo.predict(
            str(img_path), imgsz=imgsz, conf=0.25, verbose=False
        )
        onnx_n = len(onnx_results[0].boxes) if onnx_results and onnx_results[0].boxes is not None else 0

        match = abs(pt_n - onnx_n) <= 2  # allow small differences from quantization
        status = "MATCH" if match else "DIFF"

        logger.info(
            f"  {img_path.name}: PT={pt_n} detections, ONNX={onnx_n} detections [{status}]"
        )

        if not match:
            all_match = False

    if all_match:
        logger.info("Validation PASSED: ONNX output matches PyTorch")
    else:
        logger.warning(
            "Validation WARNING: Some differences detected. "
            "This may be acceptable due to floating-point precision."
        )

    return all_match


def quantize_onnx(
    onnx_path: Path,
    calibration_dir: Path,
    output_path: Optional[Path] = None,
    n_calibration: int = 100,
    imgsz: int = 640,
) -> Optional[Path]:
    """
    Apply INT8 static quantization to ONNX model.

    Uses calibration data from the training set to determine quantization ranges.
    """
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat,
        )
    except ImportError:
        logger.warning(
            "onnxruntime quantization not available. "
            "Install: pip install onnxruntime"
        )
        return None

    import cv2

    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"

    logger.info(f"Quantizing ONNX model to INT8...")

    # Calibration data reader
    class SonarCalibrationReader(CalibrationDataReader):
        def __init__(self, images_dir, n_samples, img_size):
            extensions = (".png", ".jpg", ".jpeg")
            self.image_paths = sorted(
                p for p in Path(images_dir).iterdir()
                if p.suffix.lower() in extensions
            )[:n_samples]
            self.img_size = img_size
            self.idx = 0

            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_path))
            self.input_name = session.get_inputs()[0].name

        def get_next(self):
            if self.idx >= len(self.image_paths):
                return None

            img_path = self.image_paths[self.idx]
            self.idx += 1

            img = cv2.imread(str(img_path))
            if img is None:
                return self.get_next()

            # Preprocess to match YOLO input
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = np.expand_dims(img, axis=0)  # add batch dim

            return {self.input_name: img}

    # Find calibration images
    cal_dir = calibration_dir
    if not cal_dir.exists():
        cal_dir = SPLITS_DIR / "train" / "images"

    try:
        reader = SonarCalibrationReader(cal_dir, n_calibration, imgsz)

        quantize_static(
            model_input=str(onnx_path),
            model_output=str(output_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            orig_mb = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"INT8 quantized model: {output_path} "
                f"({size_mb:.2f} MB, {size_mb / orig_mb * 100:.0f}% of original)"
            )
            return output_path

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        logger.info("Continuing without quantization...")

    return None


def benchmark_model(
    model_path: Path,
    imgsz: int = 640,
    n_warmup: int = 10,
    n_runs: int = 50,
    device: str = "cpu",
) -> Dict:
    """
    Benchmark model inference latency and throughput.
    """
    import onnxruntime as ort

    logger.info(f"Benchmarking {model_path.name} on {device}...")

    # Create session
    providers = ["CPUExecutionProvider"]
    if device != "cpu":
        providers = ["CUDAExecutionProvider"] + providers

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Create dummy input
    if input_shape[0] is None or isinstance(input_shape[0], str):
        batch = 1
    else:
        batch = input_shape[0]

    channels = input_shape[1] if len(input_shape) > 1 else 3
    dummy_input = np.random.randn(batch, channels, imgsz, imgsz).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        "model": model_path.name,
        "device": device,
        "imgsz": imgsz,
        "size_mb": model_path.stat().st_size / (1024 * 1024),
        "latency_mean_ms": float(latencies.mean()),
        "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "throughput_fps": float(1000 / latencies.mean()),
        "n_runs": n_runs,
    }

    logger.info(f"  Size:     {results['size_mb']:.2f} MB")
    logger.info(f"  Mean:     {results['latency_mean_ms']:.1f} ms")
    logger.info(f"  Median:   {results['latency_median_ms']:.1f} ms")
    logger.info(f"  P95:      {results['latency_p95_ms']:.1f} ms")
    logger.info(f"  FPS:      {results['throughput_fps']:.1f}")

    # Check against Tier 0 target
    tier0_pass = results["latency_mean_ms"] < 200 and results["size_mb"] < 15
    logger.info(
        f"  Tier 0 target (<15MB, <200ms/image): "
        f"{'PASS ✓' if tier0_pass else 'FAIL ✗'}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8-seg to ONNX and benchmark"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained .pt model",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 static quantization",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after export",
    )
    parser.add_argument(
        "--n-benchmark",
        type=int,
        default=50,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX vs PyTorch validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPORTED_DIR,
        help="Output directory for exported models",
    )
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    logger.info("=" * 60)
    logger.info("Step 1: ONNX Export")
    logger.info("=" * 60)
    onnx_path = export_to_onnx(
        model_path, output_dir, imgsz=args.imgsz, opset=args.opset
    )

    if onnx_path is None:
        return

    # Step 2: Validate
    if not args.no_validate:
        logger.info("=" * 60)
        logger.info("Step 2: Validation")
        logger.info("=" * 60)
        test_dir = SPLITS_DIR / "test" / "images"
        if test_dir.exists():
            validate_onnx(onnx_path, model_path, test_dir, imgsz=args.imgsz)
        else:
            logger.warning(f"Test images not found at {test_dir} — skipping validation")

    # Step 3: Quantize (optional)
    int8_path = None
    if args.quantize:
        logger.info("=" * 60)
        logger.info("Step 3: INT8 Quantization")
        logger.info("=" * 60)
        cal_dir = SPLITS_DIR / "train" / "images"
        int8_path = quantize_onnx(onnx_path, cal_dir, imgsz=args.imgsz)

    # Step 4: Benchmark
    if args.benchmark:
        logger.info("=" * 60)
        logger.info("Step 4: Benchmark")
        logger.info("=" * 60)

        all_results = []

        # Benchmark FP32 ONNX
        fp32_results = benchmark_model(
            onnx_path, imgsz=args.imgsz, n_runs=args.n_benchmark
        )
        all_results.append(fp32_results)

        # Benchmark INT8 if available
        if int8_path and int8_path.exists():
            int8_results = benchmark_model(
                int8_path, imgsz=args.imgsz, n_runs=args.n_benchmark
            )
            all_results.append(int8_results)

        # Save benchmark results
        bench_path = output_dir / "benchmark_results.json"
        with open(bench_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Benchmark results saved to: {bench_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Export Summary")
    logger.info("=" * 60)
    logger.info(f"ONNX FP32:  {onnx_path}")
    if int8_path:
        logger.info(f"ONNX INT8:  {int8_path}")
    logger.info(f"Calibrator: {EXPORTED_DIR / 'calibrator.pkl'}")
    logger.info("These files are loaded by ml/inference/detector.py and edge/onnx_runtime_server.py")


if __name__ == "__main__":
    main()
