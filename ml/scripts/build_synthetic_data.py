"""
DRISHTI ML Pipeline — Procedural SSS Synthetic Dataset Generator
-----------------------------------------------------------------
Generates synthetic Side-Scan Sonar (SSS) tiles for underrepresented classes
to balance the training set against the crab_pot baseline (~5700 images).

DRISHTI SSS Classes:
  0: crab_pot            — Dominant class (real data, no synthetic needed)
  1: submarine_pipeline  — Elongated linear structure with highlight/shadow
  2: shipwreck           — Irregular hull fragment with complex shadow
  3: ghost_net           — Non-rigid fibrous mesh (tangled fishing nets)
  4: mine_cylinder       — Cylindrical object with end-cap highlight

Deficit Balancing Strategy:
  1. Tally per-class counts in splits/train/labels/
  2. Calculate deficit relative to the max class count
  3. Generate synthetic tiles ONLY for underrepresented classes
  4. Inject STRICTLY into train/images + train/labels (never val/test)

Input:   ml/data/raw/ (background textures from SSS datasets)
Output:  ml/data/splits/train/images/ + ml/data/splits/train/labels/
"""

import math
import random
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- Path Configuration ---------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# ---- DRISHTI SSS Class Map ------------------------------------------------

CLASSES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}


def bbox_from_polygon(polygon):
    """Normalized polygon [(x,y),...] -> YOLO detect box (xc, yc, w, h), all in [0,1]."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


# ---- Procedural Object Generators -----------------------------------------

class GhostNetGenerator:
    """Generates non-rigid fibrous mesh structures representing abandoned fishing nets."""

    @staticmethod
    def generate(img_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        net_w = random.randint(80, 220)
        net_h = random.randint(80, 220)

        # Irregular perimeter blob
        num_points = random.randint(7, 12)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
        pts = []
        for angle in angles:
            rx = (net_w / 2) * random.uniform(0.5, 1.3)
            ry = (net_h / 2) * random.uniform(0.5, 1.3)
            px = int(np.clip(cx + rx * np.cos(angle), 5, w - 5))
            py = int(np.clip(cy + ry * np.sin(angle), 5, h - 5))
            pts.append([px, py])

        poly_pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 180)

        # Internal mesh grid lines
        mesh_spacing = random.randint(8, 18)
        min_x, min_y = np.min(poly_pts, axis=0)
        max_x, max_y = np.max(poly_pts, axis=0)

        for x in range(min_x, max_x, mesh_spacing):
            offset = random.uniform(-12, 12)
            cv2.line(mask, (x, min_y), (int(x + offset), max_y), 255, random.choice([1, 2]))
        for y in range(min_y, max_y, mesh_spacing):
            offset = random.uniform(-12, 12)
            cv2.line(mask, (min_x, y), (max_x, int(y + offset)), 255, random.choice([1, 2]))

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours((mask > 50).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.012 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx:
                polygon.append((round(pt[0][0] / w, 5), round(pt[0][1] / h, 5)))

        return mask, polygon


class PipelineGenerator:
    """Generates elongated submarine pipeline structures."""

    @staticmethod
    def generate(img_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Long, narrow rectangle at random angle
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        length = random.randint(150, 400)
        thickness = random.randint(12, 30)
        angle_deg = random.uniform(0, 180)

        rect = ((cx, cy), (length, thickness), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(mask, [box], 255)

        # Add slight waviness along the pipeline
        if random.random() > 0.4:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)

        polygon = [(round(pt[0] / w, 5), round(pt[1] / h, 5)) for pt in box]
        return mask, polygon


class ShipwreckGenerator:
    """Generates irregular shipwreck hull fragments."""

    @staticmethod
    def generate(img_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        hull_w = random.randint(100, 280)
        hull_h = random.randint(60, 180)

        # Irregular hull shape with more vertices
        num_pts = random.randint(8, 16)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
        pts = []
        for angle in angles:
            rx = (hull_w / 2) * random.uniform(0.4, 1.1)
            ry = (hull_h / 2) * random.uniform(0.4, 1.1)
            px = int(np.clip(cx + rx * np.cos(angle), 5, w - 5))
            py = int(np.clip(cy + ry * np.sin(angle), 5, h - 5))
            pts.append([px, py])

        poly_pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly_pts], 255)

        # Add internal structural lines (ribs / bulkheads)
        min_x, min_y = np.min(poly_pts, axis=0)
        max_x, max_y = np.max(poly_pts, axis=0)
        for _ in range(random.randint(3, 8)):
            if random.random() > 0.5:
                y = random.randint(min_y, max_y)
                cv2.line(mask, (min_x, y), (max_x, y + random.randint(-10, 10)), 200, 2)
            else:
                x = random.randint(min_x, max_x)
                cv2.line(mask, (x, min_y), (x + random.randint(-10, 10), max_y), 200, 2)

        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        contours, _ = cv2.findContours((mask > 50).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx:
                polygon.append((round(pt[0][0] / w, 5), round(pt[0][1] / h, 5)))

        return mask, polygon


class MineCylinderGenerator:
    """Generates cylindrical mine-like objects."""

    @staticmethod
    def generate(img_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        length = random.randint(30, 90)
        thickness = random.randint(15, 35)
        angle_deg = random.uniform(0, 360)

        # Main cylinder body
        rect = ((cx, cy), (length, thickness), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(mask, [box], 255)

        # End caps (ellipses)
        angle_rad = math.radians(angle_deg)
        for sign in [-1, 1]:
            ecx = int(cx + sign * (length / 2) * math.cos(angle_rad))
            ecy = int(cy + sign * (length / 2) * math.sin(angle_rad))
            ecx = max(5, min(w - 5, ecx))
            ecy = max(5, min(h - 5, ecy))
            cv2.ellipse(mask, (ecx, ecy), (thickness // 2, thickness // 3),
                        angle_deg, 0, 360, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx:
                polygon.append((round(pt[0][0] / w, 5), round(pt[0][1] / h, 5)))

        return mask, polygon


# Generators registry
GENERATORS = {
    1: ("submarine_pipeline", PipelineGenerator.generate),
    2: ("shipwreck", ShipwreckGenerator.generate),
    3: ("ghost_net", GhostNetGenerator.generate),
    4: ("mine_cylinder", MineCylinderGenerator.generate),
}


# ---- Acoustic Physics & Compositing Engine --------------------------------

class SonarPhysicsCompositer:
    """Applies SSS acoustic highlights, trailing shadows, and Rayleigh noise."""

    def __init__(self, raw_data_dir: Path):
        self.bg_images = self._load_backgrounds(raw_data_dir)

    def _load_backgrounds(self, raw_dir: Path) -> List[np.ndarray]:
        """Load real SSS background textures from available datasets."""
        bg_list = []
        bg_dirs = [
            raw_dir / "roboflow_sss",
            raw_dir / "side-scan-sonar-object-detection-challenge",
            raw_dir / "sss_crab_pot" / "train" / "images",
            raw_dir / "SubPipeMini2",
        ]

        for bdir in bg_dirs:
            if not bdir.exists():
                continue
            for ext in ["*.png", "*.jpg", "*.bmp", "*.tif"]:
                for img_path in bdir.rglob(ext):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.shape[0] >= 256 and img.shape[1] >= 256:
                        bg_list.append(img)
                        if len(bg_list) >= 500:  # Cap to avoid excessive memory
                            break
                if len(bg_list) >= 500:
                    break
            if len(bg_list) >= 500:
                break

        logger.info(f"Loaded {len(bg_list)} real SSS background textures")
        return bg_list

    def _generate_procedural_background(self, shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """Fallback procedural SSS background."""
        h, w = shape
        base = np.full((h, w), random.randint(35, 80), dtype=np.uint8)

        # Sand ripple patterns
        y_coords = np.arange(h)[:, None]
        freq = random.uniform(0.015, 0.06)
        ripple = (12 * np.sin(y_coords * freq + random.uniform(0, 6.28))).astype(np.int16)
        base = np.clip(base.astype(np.int16) + ripple, 0, 255).astype(np.uint8)

        # Rayleigh speckle
        noise = np.random.rayleigh(scale=0.2, size=(h, w))
        bg = np.clip(base * noise, 10, 245).astype(np.uint8)
        return bg

    def get_background(self, shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
        if self.bg_images and random.random() > 0.2:
            bg = random.choice(self.bg_images)
            bh, bw = bg.shape
            if bh >= shape[0] and bw >= shape[1]:
                sy = random.randint(0, bh - shape[0])
                sx = random.randint(0, bw - shape[1])
                return bg[sy:sy + shape[0], sx:sx + shape[1]].copy()
            return cv2.resize(bg, (shape[1], shape[0]))
        return self._generate_procedural_background(shape)

    def apply_acoustic_physics(
        self, bg: np.ndarray, obj_mask: np.ndarray
    ) -> np.ndarray:
        """Apply SSS highlight, shadow, and speckle to composite tile."""
        h, w = bg.shape
        composite = bg.astype(np.float32)

        shadow_len = random.randint(30, 100)
        shadow_dir = random.choice([-1, 1])

        # Shadow mask (shifted object mask)
        M = np.float32([[1, 0, shadow_dir * (shadow_len // 2)], [0, 1, 0]])
        shifted = cv2.warpAffine(obj_mask, M, (w, h))
        shadow = np.clip(shifted.astype(np.int16) - obj_mask.astype(np.int16), 0, 255).astype(np.uint8)
        shadow = (shadow > 30).astype(np.uint8) * 255

        # Highlight (object face)
        highlight = (obj_mask > 30).astype(np.uint8) * 255

        # Apply shadow (near-zero intensity)
        composite[shadow > 0] *= random.uniform(0.03, 0.12)

        # Apply highlight (high intensity)
        bright = np.random.uniform(185, 250, size=(h, w))
        composite[highlight > 0] = bright[highlight > 0]

        # Speckle noise
        speckle = np.random.gamma(shape=5.0, scale=0.2, size=(h, w))
        composite = np.clip(composite * speckle, 0, 255).astype(np.uint8)

        return composite


# ---- Deficit Calculator & Synthetic Builder --------------------------------

def tally_training_classes(labels_dir: Path) -> Dict[int, int]:
    """Count per-class instances in the training label directory."""
    counts = {cls_id: 0 for cls_id in CLASSES}

    if not labels_dir.exists():
        return counts

    for lbl_path in labels_dir.glob("*.txt"):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in counts:
                        counts[cls_id] += 1

    return counts


def calculate_deficit(counts: Dict[int, int]) -> Dict[int, int]:
    """Calculate how many synthetic samples each class needs."""
    max_count = max(counts.values()) if counts.values() else 0
    deficit = {}

    for cls_id, count in counts.items():
        if cls_id == 0:  # crab_pot — dominant, no synthetic needed
            deficit[cls_id] = 0
        else:
            # Target ~80% of the dominant class count
            target = int(max_count * 0.8)
            deficit[cls_id] = max(0, target - count)

    return deficit


class SyntheticDatasetBuilder:
    """Generates synthetic SSS tiles targeting underrepresented classes."""

    def __init__(self, raw_dir: Path, splits_dir: Path, split: str = "train"):
        self.compositer = SonarPhysicsCompositer(raw_dir)
        self.split = split
        self.img_dir = splits_dir / split / "images"
        self.lbl_dir = splits_dir / split / "labels"

    def build_for_class(
        self, cls_id: int, num_samples: int, seed: int = 42, tile_size: int = 640
    ) -> int:
        """Generate synthetic tiles for a single class."""
        if cls_id not in GENERATORS:
            logger.warning(f"No generator for class {cls_id}")
            return 0

        cls_name, generator_fn = GENERATORS[cls_id]
        random.seed(seed + cls_id * 10000)
        np.random.seed(seed + cls_id * 10000)

        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for i in range(1, num_samples + 1):
            sample_id = f"synth_{cls_name}_{i:05d}"

            # Generate object mask + polygon
            mask, polygon = generator_fn((tile_size, tile_size))
            if not polygon or len(polygon) < 3:
                continue

            # Get background and apply acoustic physics
            bg = self.compositer.get_background((tile_size, tile_size))
            tile = self.compositer.apply_acoustic_physics(bg, mask)

            # Save image
            cv2.imwrite(str(self.img_dir / f"{sample_id}.png"), tile)

            # Save YOLO detection label (box, not polygon - task: detect)
            xc, yc, bw, bh = bbox_from_polygon(polygon)
            if bw <= 0 or bh <= 0:
                continue
            with open(self.lbl_dir / f"{sample_id}.txt", "w") as f:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            generated += 1
            if i % 500 == 0 or i == num_samples:
                logger.info(f"  {cls_name}: {i}/{num_samples} tiles generated")

        return generated

    def build_balanced(self, deficit: Dict[int, int], seed: int = 42) -> Dict[str, int]:
        """Generate synthetic tiles for all deficit classes."""
        results = {}
        for cls_id, num_needed in deficit.items():
            if num_needed <= 0:
                continue
            cls_name = CLASSES[cls_id]
            logger.info(f"\nGenerating {num_needed} synthetic {cls_name} tiles...")
            count = self.build_for_class(cls_id, num_needed, seed)
            results[cls_name] = count
        return results


# ---- Entry Point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DRISHTI Procedural Synthetic SSS Generator — Deficit Balancing"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help="Raw dataset root (for background textures)",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=SPLITS_DIR,
        help="Splits directory (reads train labels, writes train images)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Override: generate this many per deficit class (ignores auto-deficit)",
    )
    parser.add_argument(
        "--deficit-mode", action="store_true",
        help="Auto-calculate deficit and balance to ~80%% of dominant class",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report deficit only, don't generate",
    )
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Targeted mode: comma-separated class ids to generate, e.g. '3' or '3,4'",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"],
        help="Targeted mode: which split to write into (default: train)",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Targeted mode: how many tiles to generate for each --classes id",
    )
    args = parser.parse_args()

    # ---- Targeted mode: explicit classes + split + count (used for ghost_net) ----
    if args.classes is not None:
        if args.count is None:
            parser.error("--classes requires --count")
        target_ids = [int(x) for x in args.classes.split(",") if x.strip() != ""]
        builder = SyntheticDatasetBuilder(args.raw_dir, args.splits_dir, split=args.split)
        logger.info(f"Targeted synthesis -> split='{args.split}', "
                    f"classes={target_ids}, count={args.count} each")
        if args.dry_run:
            logger.info("  [DRY RUN] No files written.")
            return
        for cid in target_ids:
            made = builder.build_for_class(cid, args.count, seed=args.seed + hash(args.split) % 1000)
            logger.info(f"  class {cid} {CLASSES.get(cid, '?'):20s}: {made} tiles -> {args.split}")
        return

    # Tally current training distribution
    train_labels = args.splits_dir / "train" / "labels"
    counts = tally_training_classes(train_labels)

    logger.info("=" * 60)
    logger.info("  CURRENT TRAINING CLASS DISTRIBUTION")
    logger.info("=" * 60)
    for cls_id, count in sorted(counts.items()):
        logger.info(f"  {cls_id} {CLASSES[cls_id]:25s}: {count:6d}")

    if args.deficit_mode or args.num_samples is None:
        deficit = calculate_deficit(counts)
    else:
        # Manual mode: generate fixed count for non-dominant classes
        deficit = {
            cls_id: args.num_samples
            for cls_id in CLASSES
            if cls_id != 0  # skip crab_pot
        }

    if args.num_samples is not None and not args.deficit_mode:
        deficit = {cls_id: args.num_samples for cls_id in CLASSES if cls_id != 0}

    logger.info("\n  SYNTHETIC DEFICIT TO GENERATE:")
    for cls_id, needed in sorted(deficit.items()):
        logger.info(f"  {cls_id} {CLASSES[cls_id]:25s}: {needed:6d} tiles needed")

    total_needed = sum(deficit.values())
    logger.info(f"\n  Total synthetic tiles to generate: {total_needed}")

    if args.dry_run:
        logger.info("\n  [DRY RUN] No files written.")
        return

    if total_needed == 0:
        logger.info("\n  No synthetic data needed — classes already balanced!")
        return

    builder = SyntheticDatasetBuilder(args.raw_dir, args.splits_dir)
    results = builder.build_balanced(deficit, seed=args.seed)

    logger.info("\n" + "=" * 60)
    logger.info("  SYNTHETIC GENERATION COMPLETE")
    logger.info("=" * 60)
    for cls_name, count in results.items():
        logger.info(f"  {cls_name:25s}: {count:6d} tiles generated")
    logger.info("=" * 60)

    # Report updated distribution
    updated_counts = tally_training_classes(train_labels)
    logger.info("\n  UPDATED TRAINING DISTRIBUTION:")
    max_c = max(updated_counts.values()) if updated_counts.values() else 1
    for cls_id, count in sorted(updated_counts.items()):
        bar = "█" * int(40 * count / max(max_c, 1))
        logger.info(f"  {cls_id} {CLASSES[cls_id]:25s}: {count:6d} {bar}")


if __name__ == "__main__":
    main()