# Pre-final-training change list

Everything to fold in **if** we do one last training run (Run 4). Frozen fallback:
`best_detector_prep.pt` (Run 3, shipped). Only ship Run 4 if it beats Run 3 on a
matched protocol (same test set, precision + FP-rate + per-class AP50).

Source for each item: the 12-source literature review — see `PROJECT_RECORD.html` §12.

---

## Tier A — do these (low risk, high expected value)

### A1. 50 px proximity box-merge on AI4Shipwrecks tiles
- **File:** `ml/scripts/tile_ai4shipwrecks.py`
- **Change:** after `connectedComponents`, before writing boxes, merge any two boxes whose
  minimum edge-to-edge distance `< 50 px` (~1–2 m) into their union. Keep the existing
  `MASK_CLOSE_PX` close as a first pass or drop it — test both.
- **Why:** a single wreck fragmented by acoustic shadow currently becomes several boxes,
  which inflates the shipwreck test set and confuses training. DFSE-YOLO (2026) cross-validated
  this merge on 14 wrecks at F1 0.87.
- **Re-run:** `tile_ai4shipwrecks.py` → `build_dataset.py` → `preprocess_splits.py`.
- **Check:** shipwreck instance count per tile drops; shipwreck AP50 should rise toward 0.5–0.7.

### A2. WIoU v3 box loss (replaces CIoU)
- **File:** Ultralytics install — 1-file patch to `ultralytics/utils/loss.py` (or `metrics.py`),
  a known community snippet; wire a `--iou-type wiou` flag through `train_yolo_seg.py`.
- **Why:** CIoU degrades on extreme-aspect boxes (our pipelines) and low-quality labels
  (our mask-derived wreck boxes). WIoU v3's non-monotonic focusing down-weights those.
  Zero inference cost, zero size cost. +1.2 mAP alone in Jiang et al. 2024's ablation.
- **Risk:** low — it is a loss swap; if Run 4 is worse, revert the flag.

### A3. Confirm + record the site-disjoint AI4Shipwrecks split
- **File:** `ml/scripts/build_dataset.py` (or wherever the shipwreck split is drawn).
- **Change:** ensure no wreck *site* appears in both train and test; if it does, redraw by site.
- **Why:** SW-Net (14 train sites / 15 test sites), TR-YOLOv5 and GhostVision all split
  geographically. It is a rigor checkbox judges look for, and it makes the number honest.

---

## Tier B — do if a training slot is free

### B1. YOLOv8n 4-class variant (separate export, not a replacement)
- **Change:** train a `yolov8n.pt`-seeded model on the same 4-class data; export ONNX.
- **Why:** the SSS shipwreck literature runs ~5 MB nano models. A size/mAP point next to our
  's' model pre-empts "why is your model 45 MB?" and may itself be the better edge model.
- **Keep:** the 's' model stays the default unless nano is within ~2 mAP.

### B2. Staged transfer (COCO → generic-sonar → 4-class)
- **Change:** a short warm-up training pass on KLSG-as-generic-sonar (or raw AI4Shipwrecks /
  SubPipe strips, unlabelled for our classes), then continue on the 4-class set.
- **Why:** Yu et al. and Qin et al. — an in-domain intermediate adapts the backbone to
  speckle statistics before the small task-specific set. Caveat (Du et al. 2023): KLSG's
  object types help shipwreck-like classes only, not pipeline/cylinder/net.
- **Risk:** medium-uncertain gain on an already-COCO-pretrained model. Time-box it.

---

## Tier C — data / eval, not training itself

### C1. Nadir-gap mask
- **File:** `ml/scripts/run_aurora_survey.py` (transect path) + optionally Module 2.
- **Change:** suppress or flag detections in the fixed centre column band where `G → 0`.
- **Why:** SW-Net — a fixed-geometry false-positive region, cheap to remove.

### C2. Temporal-persistence gate (transect inference only)
- **File:** `ml/scripts/run_aurora_survey.py`; new option in `ml/inference/confidence_filter.py`.
- **Change:** require a detection to recur across ≥ K overlapping tile windows before accepting.
  Set K by a coarse-graining sweep (bin detections at doubling scales, find where real
  objects cluster) — GhostVision + Cuff et al. Combined score
  `S = α·conf_avg + (1−α)·count/max_count`, α ≈ 0.85–0.95.
- **Why:** kills one-off false positives on full transects. Does not touch tile-level training.

### C3. Synthetic-vs-real feature-distribution check
- **Change:** for ghost_net and mine_cylinder, compare synthetic vs real feature
  distributions (GLCM stats or an embedding overlap plot), or train-real/test-synth +
  train-synth/test-real.
- **Why:** Qin et al. — synthetic data helps only when it matches the real distribution.
  Pre-empts "your ghost_net is fake" with a measured answer.

---

## Deck / eval artefacts (no retrain needed — do these regardless)

| Artefact | Source | File |
|---|---|---|
| Processing-time vs survey-time number (~10× real-time) | GhostVision | measure on an AURORA transect |
| Scenario table: low-contrast / reverberation / complex-seabed bins × mAP | DFSE-YOLO T3, TR-YOLOv5 | new `ml/scripts/eval_scenarios.py` |
| Confusion matrix + class-balance ("mutual influence") analysis | Du et al. 2025 | `ml/scripts/evaluate.py` extension |
| Grad-CAM heatmaps over sonar tiles (highlight + shadow) | Jiang et al. 2024 | Ultralytics explainability util |

---

## Do NOT do before the deadline

- CloFormer / heavy attention modules — ~30 % FPS hit (DFSE-YOLO, Jiang), fails the edge budget.
- INT8 — measured: faint-target recall collapsed. FP32 ONNX ships.
- GAN-based synthetic data — resource-heavy, "imitates, doesn't manufacture" (Qin et al.).
- Full DFSE selective-enhancement architecture — v2 roadmap, not a hackathon task.
