"""Generate the Colab notebooks in notebooks/ (keeps the .ipynb JSON valid by construction)."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)

REPO = "https://github.com/Rehan9599/Sonar-Drishti.git"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": "\n".join(lines)}


def notebook(cells, name):
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    p = OUT / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {p.relative_to(OUT.parent)}  ({len(cells)} cells)")


# ───────────────────────── 01 — quickstart inference ─────────────────────────
cells = [
    md("# DRISHTI — run the detector (Colab quickstart)",
       "",
       "Detect man-made seabed hazards in side-scan sonar. **~2 minutes, CPU only, no GPU needed.**",
       "",
       "| | |",
       "|---|---|",
       "| Repo | https://github.com/Rehan9599/Sonar-Drishti |",
       "| Model | https://huggingface.co/rehan9599/drishti-detector |",
       "",
       "### What you need",
       "**Nothing.** The model *and* sample tiles are committed in the repo — one `git clone` and you're running.",
       "",
       "### What you do NOT need",
       "- No PyTorch (this notebook uses `onnxruntime`, ~50 MB instead of ~2 GB)",
       "- No Hugging Face account",
       "- No dataset download — that's only for **re-training**",
       "",
       "> ⚠️ **The one thing people get wrong:** this model was trained on **Lee-filter + CLAHE**",
       "> preprocessed tiles. Feeding it a raw sonar image without that filter silently degrades every",
       "> prediction. The pipeline applies it for you (`preprocess=True`) — just don't disable it on raw input."),

    md("## 1. Setup"),
    code("!pip -q install onnxruntime opencv-python-headless numpy scikit-learn",
        "!git clone -q " + REPO,
        "%cd Sonar-Drishti",
        "",
        "import sys; sys.path.insert(0, '.')",
        "print('ready')"),

    code("# sanity: the model ships in the repo",
        "!ls -lh ml/models/exported/best_detector.onnx ml/models/exported/calibrator.pkl demo/tiles/"),

    md("## 2. What actually happens",
       "",
       "Running the detector alone gives you raw boxes. The *product* is three stages:",
       "",
       "```",
       "tile ─► M0 preprocess ─► M1 detect (YOLOv8s) ─► M2 confidence ─► box + class + 0-100 %",
       "        Lee + CLAHE      raw boxes + scores    per-class gate",
       "                                               per-class calibration",
       "```",
       "",
       "**Module 2 is not optional.** It applies a per-class score gate (most false positives sit far",
       "below true positives) and per-class Platt calibration that turns a raw score into an honest",
       "probability — Expected Calibration Error 0.052 → 0.037. Skipping it gives you numbers that",
       "look like confidences but aren't."),

    md("## 3. Detect"),
    code("from edge.edge_infer import run",
        "import json",
        "",
        "# --no-preprocess equivalent: these demo tiles are ALREADY Lee+CLAHE filtered.",
        "# For your own RAW sonar image use preprocess=True (the default).",
        "result = run('demo/tiles/synth_ghost_net_00002.jpg', preprocess=False)",
        "print(json.dumps(result, indent=2))"),

    md("### Reading the output",
      "",
      "| field | meaning |",
      "|---|---|",
      "| `class_label` | `submarine_pipeline` · `shipwreck` · `mine_cylinder` · `ghost_net` |",
      "| `confidence_score` | **calibrated** 0–100 %, not the raw sigmoid |",
      "| `bbox` | `[x1, y1, x2, y2]` in pixels of the input image |",
      "| `review_status` | ≥80 % `auto_confirmed` · 30–80 % `pending_review` · <30 % dropped |",
      "",
      "`crab_pot` is trained but **filtered out** of the product — it never reached usable accuracy."),

    md("## 4. See it"),
    code("import cv2, matplotlib.pyplot as plt",
        "",
        "def show(path, dets):",
        "    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)",
        "    for d in dets:",
        "        x1, y1, x2, y2 = [int(v) for v in d['bbox']]",
        "        ok = d['confidence_score'] >= 80",
        "        col = (0, 200, 0) if ok else (255, 165, 0)",
        "        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)",
        "        cv2.putText(img, f\"{d['class_label']} {d['confidence_score']:.0f}%\",",
        "                    (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)",
        "    plt.figure(figsize=(11, 9)); plt.imshow(img); plt.axis('off')",
        "    plt.title(f\"{len(dets)} detection(s)\"); plt.show()",
        "",
        "show('demo/tiles/synth_ghost_net_00002.jpg', result['detections'])"),

    code("# all three demo tiles",
        "import glob",
        "for p in sorted(glob.glob('demo/tiles/*.jpg')):",
        "    r = run(p, preprocess=False)",
        "    print(f\"{p.split('/')[-1]:34s} {r['kept']} kept, {r['inference_ms']} ms\")",
        "    for d in r['detections']:",
        "        print(f\"     {d['class_label']:20s} {d['confidence_score']:5.1f}%  {d['review_status']}\")"),

    md("> **Note on these tiles.** They are *synthetic* `ghost_net` samples we generated — there is no",
      "> public real ghost-net-in-SSS dataset. High scores here are proof-of-capability, **not** field",
      "> performance. Real-data numbers: `submarine_pipeline` AP50 0.98, `shipwreck` ~0.30–0.45,",
      "> `mine_cylinder` 0.42."),

    md("## 5. Your own image"),
    code("# from google.colab import files",
        "# up = files.upload()",
        "# path = list(up.keys())[0]",
        "",
        "# RAW sonar image → keep preprocess=True (default) so Lee+CLAHE is applied",
        "# result = run(path)                 # <- correct for raw input",
        "# show(path, result['detections'])"),

    md("## 6. Coordinates (Module 3)",
       "",
       "Detections above are **pixels**. To get latitude/longitude you need the sonar's navigation —",
       "an XTF log and/or a navigation CSV. Pixels alone cannot become coordinates.",
       "",
       "```python",
       "from ml.inference.pipeline import run_pipeline",
       "",
       "report = run_pipeline(",
       "    'tile.png', source_file='DATA0000106.H-PU',",
       "    model_path='ml/models/exported/best_detector.onnx',",
       "    xtf='DATA0000106.H-PU.xtf',      # from xtf/xtf-navigation/, NOT xtf/xtf-data/",
       "    nav='navigation.csv',",
       ")",
       "report['detections'][0]['latitude'], report['detections'][0]['longitude']",
       "```",
       "",
       "Each record then carries `latitude`, `longitude`, `ping_id`, `across_track_m`, `side` and a",
       "Leaflet-ready GeoJSON is written alongside.",
       "",
       "**The nav files are not in the repo** (survey data, not ours to redistribute). Point it at your",
       "own sonar log.",
       "",
       "> **Accuracy caveat:** the tow-fish position recovers to <1 m, but detection position also",
       "> depends on across-track scale — two navigation paths can disagree by ~100 m on the same target.",
       "> Treat a pin as a **search area, not a survey fix**."),

    md("## 7. If you want to re-train",
       "",
       "Only then do you need the dataset (2.1 GB, **access required** — it contains third-party-derived",
       "tiles and is private until every source licence is verified):",
       "",
       "```python",
       "from huggingface_hub import snapshot_download",
       "snapshot_download('rehan9599/drishti-sss', repo_type='dataset',",
       "                  local_dir='ml/data/splits', token='hf_...')   # your OWN read token",
       "```",
       "",
       "```bash",
       "!yolo detect train data=ml/configs/drishti.yaml model=yolov8s.pt imgsz=640 epochs=120 batch=16",
       "```",
       "",
       "Ask Rehan for dataset access. **Never share a write token** — generate your own read token at",
       "https://huggingface.co/settings/tokens.",
       "",
       "---",
       "",
       "### Gotchas, collected",
       "1. Raw sonar input → `preprocess=True`. Tiles from `ml/data/splits/` → `preprocess=False` (already filtered).",
       "2. Use `conf=0.10` at the detector; Module 2's per-class gate does the real cut.",
       "3. Coordinates need XTF/nav — pixels alone can't produce them.",
       "4. The model **does not transfer** to an unseen survey without fine-tuning.",
       "5. Don't use the INT8 ONNX — its accuracy collapsed to zero. FP32 or FP16 only."),
]
notebook(cells, "01_quickstart_inference.ipynb")

print("\nOpen in Colab:")
print("  https://colab.research.google.com/github/Rehan9599/Sonar-Drishti/blob/main/"
      "notebooks/01_quickstart_inference.ipynb")
