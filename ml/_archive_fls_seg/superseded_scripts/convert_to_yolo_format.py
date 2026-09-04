"""
DRISHTI ML Pipeline — SSS YOLO-seg Format Converter & Orchestrator
-------------------------------------------------------------------
Orchestrates the full SSS data preparation pipeline:

  1. Extract real SSS datasets → splits/ (preserving original splits)
  2. Generate deficit-balanced synthetic tiles → splits/train/
  3. Validate all labels have correct class IDs (0–4)
  4. Report final class distribution

This replaces the old Watertank-specific conversion script.

DRISHTI SSS Classes:
  0: crab_pot
  1: submarine_pipeline
  2: shipwreck
  3: ghost_net
  4: mine_cylinder

Usage:
    python convert_to_yolo_format.py                    # full pipeline
    python convert_to_yolo_format.py --no-sam           # skip SAM
    python convert_to_yolo_format.py --no-synthetic     # real data only
    python convert_to_yolo_format.py --validate-only    # just validate
"""

import logging
import argparse
import sys
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---- Configuration --------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

VALID_CLASSES = {0, 1, 2, 3, 4}
CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}


# ---- Validation ------------------------------------------------------------

def validate_labels(splits_dir: Path) -> bool:
    """
    Validate all YOLO-seg label files have correct format and class IDs.

    Returns True if all labels are valid.
    """
    logger.info("Validating label integrity...")
    errors = 0
    total_files = 0
    total_instances = 0
    class_counts: Dict[int, int] = {c: 0 for c in VALID_CLASSES}

    for split in ["train", "val", "test"]:
        lbl_dir = splits_dir / split / "labels"
        img_dir = splits_dir / split / "images"

        if not lbl_dir.exists():
            logger.warning(f"  {split}/labels/ does not exist")
            continue

        label_files = sorted(lbl_dir.glob("*.txt"))
        image_files = set(p.stem for p in img_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))

        for lbl_path in label_files:
            total_files += 1
            stem = lbl_path.stem

            # Check matching image exists
            if stem not in image_files:
                logger.error(f"  Orphan label (no image): {split}/labels/{lbl_path.name}")
                errors += 1

            with open(lbl_path) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:  # Need at least cls + 2 points (4 coords)
                        logger.error(
                            f"  Malformed label in {split}/{lbl_path.name}:{line_num} "
                            f"— only {len(parts)} values"
                        )
                        errors += 1
                        continue

                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        logger.error(f"  Invalid class ID in {split}/{lbl_path.name}:{line_num}")
                        errors += 1
                        continue

                    if cls_id not in VALID_CLASSES:
                        logger.error(
                            f"  Invalid class {cls_id} in {split}/{lbl_path.name}:{line_num} "
                            f"— valid range: 0–4"
                        )
                        errors += 1
                        continue

                    # Validate coordinates are in [0, 1]
                    coords = parts[1:]
                    for i, coord in enumerate(coords):
                        try:
                            val = float(coord)
                            if val < -0.01 or val > 1.01:
                                logger.warning(
                                    f"  Coordinate out of range ({val}) in "
                                    f"{split}/{lbl_path.name}:{line_num}"
                                )
                        except ValueError:
                            logger.error(
                                f"  Invalid coordinate in {split}/{lbl_path.name}:{line_num}"
                            )
                            errors += 1

                    class_counts[cls_id] += 1
                    total_instances += 1

        # Check for images without labels
        label_stems = set(p.stem for p in label_files)
        orphan_images = image_files - label_stems
        if orphan_images:
            logger.info(
                f"  {split}: {len(orphan_images)} images without labels "
                "(background/negative images — this is OK)"
            )

        logger.info(
            f"  {split}: {len(label_files)} labels, {len(image_files)} images"
        )

    # Report
    logger.info(f"\n  Total: {total_files} label files, {total_instances} instances")
    logger.info(f"  Errors: {errors}")

    if errors > 0:
        logger.error(f"\n  VALIDATION FAILED — {errors} errors found!")
        return False

    logger.info("\n  ✓ All labels valid!")

    # Class distribution
    logger.info("\n  Class Distribution (all splits):")
    max_count = max(class_counts.values()) if class_counts.values() else 1
    for cls_id in sorted(class_counts):
        count = class_counts[cls_id]
        bar = "█" * int(40 * count / max(max_count, 1))
        logger.info(f"    {cls_id} {CLASS_NAMES[cls_id]:25s}: {count:6d} {bar}")

    return True


# ---- Orchestrator -----------------------------------------------------------

def run_pipeline(
    raw_dir: Path,
    splits_dir: Path,
    use_sam: bool = True,
    sam_checkpoint: str = None,
    preprocess: bool = True,
    generate_synthetic: bool = True,
    synthetic_seed: int = 42,
    validate_only: bool = False,
):
    """Run the full SSS data preparation pipeline."""

    if validate_only:
        valid = validate_labels(splits_dir)
        sys.exit(0 if valid else 1)

    # Step 1: Extract real SSS datasets
    logger.info("=" * 60)
    logger.info("  STEP 1: Extract Real SSS Datasets")
    logger.info("=" * 60)

    from extract_and_annotate_sss import extract_all
    extract_stats = extract_all(
        raw_dir=raw_dir,
        splits_dir=splits_dir,
        use_sam=use_sam,
        sam_checkpoint=sam_checkpoint,
        preprocess=preprocess,
    )

    # Step 2: Generate synthetic data (train-only)
    if generate_synthetic:
        logger.info("\n" + "=" * 60)
        logger.info("  STEP 2: Deficit-Balanced Synthetic Generation")
        logger.info("=" * 60)

        from build_synthetic_data import (
            tally_training_classes,
            calculate_deficit,
            SyntheticDatasetBuilder,
        )

        counts = tally_training_classes(splits_dir / "train" / "labels")
        deficit = calculate_deficit(counts)

        total_deficit = sum(deficit.values())
        if total_deficit > 0:
            builder = SyntheticDatasetBuilder(raw_dir, splits_dir)
            builder.build_balanced(deficit, seed=synthetic_seed)
        else:
            logger.info("  No synthetic data needed — classes already balanced!")
    else:
        logger.info("\n  Synthetic generation skipped (--no-synthetic)")

    # Step 3: Validate
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 3: Label Validation")
    logger.info("=" * 60)

    valid = validate_labels(splits_dir)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  SSS DATA PREPARATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Output: {splits_dir}")
    logger.info(f"  Validation: {'PASSED ✓' if valid else 'FAILED ✗'}")
    logger.info("\n  Next step: python train_yolo_seg.py")

    return valid


# ---- Entry Point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DRISHTI SSS → YOLO-seg Format Conversion & Data Preparation"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help="Raw dataset root directory",
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=SPLITS_DIR,
        help="Output splits directory",
    )
    parser.add_argument(
        "--no-sam", action="store_true",
        help="Skip SAM auto-annotation",
    )
    parser.add_argument(
        "--sam-checkpoint", type=str, default=None,
        help="Path to SAM checkpoint",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip acoustic preprocessing",
    )
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Skip synthetic data generation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing labels, don't process",
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        splits_dir=args.splits_dir,
        use_sam=not args.no_sam,
        sam_checkpoint=args.sam_checkpoint,
        preprocess=not args.no_preprocess,
        generate_synthetic=not args.no_synthetic,
        synthetic_seed=args.seed,
        validate_only=args.validate_only,
    )


if __name__ == "__main__":
    main()
