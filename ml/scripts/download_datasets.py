"""
DRISHTI ML Pipeline — SSS Dataset Downloader & Ingestion Script
----------------------------------------------------------------
Downloads Side-Scan Sonar (SSS) datasets for DRISHTI (SIH 2026 PS #26057).
All FLS (Forward-Looking Sonar) datasets have been removed — this pipeline
is now pure SSS.

Datasets:
  1. SSS Crab Pot Detection (HuggingFace: Crab-Pot/crab_pot_detection_dataset)
  2. SubPipeMini2 (SSS pipeline survey imagery)
  3. AI4Shipwrecks (Zenodo: 286 labelled real-shipwreck SSS images)
  4. Kaggle Sonar Mine Detection (MILCO / NonMILCO)
  5. Roboflow SSS (Side-Scan Sonar Object Detection)
  6. SSS Object Detection Challenge (Kaggle)
  7. SeabedObjects-KLSG (GitHub: SSS ship & airplane dataset)
  8. SSL Sonar (Pre-trained SSL weights — optional)

Outputs raw files to: ml/data/raw/
"""

import os
import sys
import shutil
import logging
import argparse
import urllib.request
import zipfile
import tarfile
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Paths & Configuration ------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_RAW = SCRIPT_DIR.parent / "data" / "raw"

# SSS Dataset Registry
SSS_DATASETS = {
    "sss_crab_pot": {
        "description": "HuggingFace Crab-Pot SSS Detection Dataset (~5700 train images)",
        "url": None,  # Downloaded via HuggingFace datasets library
        "hf_repo": "Crab-Pot/crab_pot_detection_dataset",
        "dest": "sss_crab_pot",
        "optional": False,
    },
    "SubPipeMini2": {
        "description": "SubPipeMini2: SSS Pipeline Survey (raw SSS strips + telemetry)",
        "url": "https://zenodo.org/records/PLACEHOLDER/files/SubPipeMini2.zip",
        "dest": "SubPipeMini2",
        "optional": True,
    },
    "ai4shipwrecks": {
        "description": "AI4Shipwrecks: 286 Labelled Real-Shipwreck SSS Images",
        "url": "https://zenodo.org/records/10578378/files/AI4Shipwrecks.zip",
        "dest": "ai4shipwrecks",
        "optional": False,
    },
    "kaggle_sonar_mine": {
        "description": "Kaggle Sonar Mine Detection: MILCO & NonMILCO SSS Objects",
        "url": None,  # Downloaded via Kaggle CLI
        "dest": "kaggle_sonar_mine",
        "kaggle_target": "sierra022/sonar-imaging-mine-detection",
        "optional": False,
    },
    "roboflow_sss": {
        "description": "Roboflow SSS: Side-Scan Sonar Object Detection",
        "url": "https://universe.roboflow.com/ds/mg4j8?key=drishti_sss_export",
        "dest": "roboflow_sss",
        "optional": True,
    },
    "side_scan_challenge": {
        "description": "SSS Object Detection Challenge (Kaggle competition data)",
        "url": None,
        "dest": "side-scan-sonar-object-detection-challenge",
        "kaggle_target": "side-scan-sonar-object-detection-challenge",
        "optional": True,
    },
    "seabed_objects_klsg": {
        "description": "SeabedObjects-KLSG: Real Side-Scan Seabed Objects (Ship/Plane)",
        "url": "https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset/archive/refs/heads/master.zip",
        "dest": "seabed_objects_klsg",
        "optional": True,
    },
    "ssl_sonar": {
        "description": "GitHub SSL Sonar: Pre-trained Acoustic SSL Weights (no labelled images)",
        "url": None,
        "git_repo": "https://github.com/agrija9/ssl-sonar-images.git",
        "dest": "ssl_sonar",
        "optional": True,
    },
}


# ---- Download & Utility Helpers ------------------------------------------

def _download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress reporting. Returns True on success."""
    if dest_path.exists():
        logger.info(f"Already downloaded: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {desc or url} -> {dest_path.name}")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DRISHTI-ML-Downloader/2.0"}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 8192
            block_num = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                block_num += 1
                out_file.write(buffer)
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    print(f"\r  Progress: {pct}%", end="", flush=True)
        print()  # Newline after progress
        return True
    except Exception as e:
        logger.warning(f"Download failed for {desc or url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def _extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract zip or tar archive safely."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive_path.name} -> {dest_dir}")

    try:
        if archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest_dir)
        else:
            logger.warning(f"Unsupported archive extension: {archive_path.name}")
            return False
        return True
    except Exception as e:
        logger.error(f"Extraction failed for {archive_path.name}: {e}")
        return False


def _try_git_clone(repo_url: str, dest_dir: Path) -> bool:
    """Attempt to clone a git repository."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info(f"Already cloned: {dest_dir.name}")
        return True

    try:
        logger.info(f"Cloning {repo_url} -> {dest_dir.name}")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except FileNotFoundError:
        logger.warning("git executable not found on PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"git clone failed: {e.stderr}")
        return False


def _try_kaggle_download(kaggle_dataset: str, dest_dir: Path) -> bool:
    """Attempt download using Kaggle CLI API."""
    try:
        logger.info(f"Attempting Kaggle API download for {kaggle_dataset}...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(dest_dir), "--unzip"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully downloaded Kaggle dataset {kaggle_dataset}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.info(f"Kaggle CLI not available or dataset locked ({kaggle_dataset}).")
        return False


def _try_hf_download(hf_repo: str, dest_dir: Path) -> bool:
    """Attempt download using HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        logger.info(f"Downloading HuggingFace dataset: {hf_repo}...")
        ds = load_dataset(hf_repo, cache_dir=str(dest_dir))
        logger.info(f"HuggingFace dataset {hf_repo} loaded successfully.")
        # The dataset will be cached in dest_dir by HuggingFace
        return True
    except ImportError:
        logger.warning("HuggingFace `datasets` library not installed.")
        return False
    except Exception as e:
        logger.warning(f"HuggingFace download failed for {hf_repo}: {e}")
        return False


# ---- Main Download Pipeline -----------------------------------------------

def download_sss_datasets(data_dir: Path) -> dict:
    """Download all SSS datasets for DRISHTI pipeline."""
    results = {}

    for name, info in SSS_DATASETS.items():
        dest = data_dir / info["dest"]

        if dest.exists() and any(dest.iterdir()):
            logger.info(f"{info['description']} — already present")
            results[name] = True
            continue

        # Try HuggingFace download
        if "hf_repo" in info:
            if _try_hf_download(info["hf_repo"], dest):
                results[name] = True
                continue

        # Try Git Clone
        if "git_repo" in info:
            if _try_git_clone(info["git_repo"], dest):
                results[name] = True
                continue

        # Try Kaggle CLI
        if "kaggle_target" in info:
            if _try_kaggle_download(info["kaggle_target"], dest):
                results[name] = True
                continue

        # Direct HTTP Download
        if info.get("url") and "PLACEHOLDER" not in info["url"]:
            zip_path = data_dir / f"{name}.zip"
            if _download_file(info["url"], zip_path, info["description"]):
                if _extract_archive(zip_path, dest):
                    zip_path.unlink(missing_ok=True)
                    results[name] = True
                    continue

        # Manual download notification
        logger.info(
            f"Note: {info['description']} requires manual file placement.\n"
            f"  Target path: {dest}"
        )
        results[name] = False

    return results


# ---- Main Entry Point ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DRISHTI ML Pipeline — SSS Dataset Download & Ingestion Manager"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_RAW,
        help=f"Target directory for raw dataset output (default: {DATA_RAW})",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Download only core SSS datasets (crab_pot, ai4shipwrecks, kaggle_mine)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  DRISHTI ML Engine — SSS Dataset Download & Ingestion Pipeline")
    logger.info("=" * 65)
    logger.info(f"Raw Output Folder: {data_dir.resolve()}\n")

    # Download SSS datasets
    logger.info("--- SSS (Side-Scan Sonar) Datasets ---")
    results = download_sss_datasets(data_dir)

    # Summary Output
    logger.info("\n" + "=" * 65)
    logger.info("   SSS DATASET INGESTION SUMMARY")
    logger.info("=" * 65)

    core_datasets = ["sss_crab_pot", "ai4shipwrecks", "kaggle_sonar_mine"]
    all_ok = True

    for name, status in results.items():
        desc = SSS_DATASETS[name]["description"].split(":")[0]
        is_core = name in core_datasets
        if status:
            status_str = "[OK]"
        elif is_core:
            status_str = "[FAIL - REQUIRED]"
            all_ok = False
        else:
            status_str = "[SKIPPED/OPTIONAL]"
        logger.info(f"  {desc:45s}: {status_str}")

    logger.info("=" * 65)

    if not all_ok:
        logger.warning(
            "\nSome core SSS datasets are missing. Please download manually.\n"
            "The pipeline can still proceed with available data."
        )

    logger.info(
        "\nData ingestion complete! Next step: Run SSS extraction & annotation:\n"
        "  python ml/scripts/extract_and_annotate_sss.py"
    )


if __name__ == "__main__":
    main()