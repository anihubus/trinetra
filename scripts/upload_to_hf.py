"""
Publish the DRISHTI model + dataset to the Hugging Face Hub.

Model repo  -> PUBLIC by default  (our own trained weights)
Dataset repo-> PRIVATE by default (contains third-party-derived tiles; see docs/hf/dataset_card.md)

    hf auth login                 # once, needs a WRITE token
    python scripts/upload_to_hf.py --user Rehan9599 --what model
    python scripts/upload_to_hf.py --user Rehan9599 --what dataset
    python scripts/upload_to_hf.py --user Rehan9599 --what both --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "ml" / "models" / "checkpoints"
EXP = ROOT / "ml" / "models" / "exported"
SPLITS = ROOT / "ml" / "data" / "splits"
CARDS = ROOT / "docs" / "hf"

# shipped run — training curves / confusion matrix / exact hyper-parameters
RUN = CKPT / "yolov8_seg_20260831_231345"
RUN_ARTIFACTS = [
    "results.csv", "args.yaml", "results.png", "confusion_matrix.png",
    "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
    "val_batch0_pred.jpg", "val_batch0_labels.jpg",
]

MODEL_FILES = [
    (CKPT / "best_detector.pt", "best_detector.pt"),
    (CKPT / "best_detector_prep.pt", "best_detector_prep.pt"),
    (CKPT / "best_detector_raw.pt", "best_detector_raw.pt"),
    (EXP / "best_detector.onnx", "best_detector.onnx"),
    (EXP / "best_detector_fp16.onnx", "best_detector_fp16.onnx"),
    (EXP / "best_detector_int8.onnx", "best_detector_int8.onnx"),
    (EXP / "calibrator.pkl", "calibrator.pkl"),
]


def _mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def stage_model(dst: Path) -> None:
    shutil.copy2(CARDS / "model_card.md", dst / "README.md")
    for src, name in MODEL_FILES:
        if src.exists():
            shutil.copy2(src, dst / name)
            print(f"    {_mb(src):8.1f} MB  {name}")
        else:
            print(f"    {'MISSING':>8}   {name}  ({src})")
    train_dir = dst / "training"
    train_dir.mkdir(exist_ok=True)
    for name in RUN_ARTIFACTS:
        src = RUN / name
        if src.exists():
            shutil.copy2(src, train_dir / name)
    n = len(list(train_dir.iterdir()))
    print(f"    {'':8}     training/  ({n} provenance artifacts)")


def stage_dataset(dst: Path, dry: bool = False) -> None:
    """Copy splits into the staging dir. On --dry-run just report, don't copy 2.1 GB."""
    if not dry:
        shutil.copy2(CARDS / "dataset_card.md", dst / "README.md")
        yaml = ROOT / "ml" / "configs" / "drishti.yaml"
        if yaml.exists():
            shutil.copy2(yaml, dst / "drishti.yaml")
    for split in ("train", "val", "test"):
        s = SPLITS / split
        if not s.exists():
            print(f"    MISSING split: {s}")
            continue
        n = len(list((s / "images").glob("*")))
        if not dry:
            shutil.copytree(s, dst / split, dirs_exist_ok=True)
        print(f"    {split:5s}  {n} images")


def upload(repo_id: str, repo_type: str, folder: Path, private: bool, dry: bool,
           size_mb: float | None = None) -> None:
    total = size_mb if size_mb is not None else \
        sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6
    print(f"\n  -> {repo_type}: https://huggingface.co/"
          f"{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
          f"  [{'private' if private else 'PUBLIC'}]  {total:.1f} MB")
    if dry:
        print("     (dry run — nothing uploaded)")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="DRISHTI v1.0 — SIH 2026 PS 26057",
    )
    print("     done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your Hugging Face username or org")
    ap.add_argument("--what", choices=["model", "dataset", "both"], default="both")
    ap.add_argument("--model-repo", default="drishti-detector")
    ap.add_argument("--dataset-repo", default="drishti-sss")
    ap.add_argument("--public-dataset", action="store_true",
                    help="publish the dataset PUBLICLY — only after verifying every source licence")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (CARDS / "model_card.md").exists():
        sys.exit(f"missing cards in {CARDS}")

    if args.what in ("model", "both"):
        print("\nStaging model ...")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stage_model(d)
            upload(f"{args.user}/{args.model_repo}", "model", d, private=False, dry=args.dry_run)

    if args.what in ("dataset", "both"):
        print("\nStaging dataset (2.1 GB — this takes a while) ...")
        src_mb = sum(f.stat().st_size for f in SPLITS.rglob("*") if f.is_file()) / 1e6
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            stage_dataset(d, dry=args.dry_run)
            upload(f"{args.user}/{args.dataset_repo}", "dataset", d,
                   private=not args.public_dataset, dry=args.dry_run, size_mb=src_mb)

    print("\nNext: fill the two *(link TBD)* placeholders in README.md and ml/models/README.md.")


if __name__ == "__main__":
    main()
