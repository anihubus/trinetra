---
license: other
license_name: mixed-source-see-provenance
task_categories:
  - object-detection
tags:
  - sonar
  - side-scan-sonar
  - marine-debris
  - underwater
size_categories:
  - 1K<n<10K
---

# DRISHTI — side-scan sonar training splits

The assembled, preprocessed train / val / test tiles behind the
[DRISHTI detector](https://github.com/Rehan9599/Sonar-Drishti). YOLO format.

> ## ⚠️ Access and licence
>
> **This repo is private/gated.** It contains tiles **derived from six third-party datasets**
> whose licences were not distributed with the source archives. It is shared for
> project-team reproduction only and must not be redistributed publicly until each
> source licence has been verified. The **synthetic** portion (~26 % of train) is original
> work and is freely reusable; everything else is governed by its upstream licence.
>
> If you want a fully unencumbered path: use `ml/scripts/` in the GitHub repo to rebuild the
> splits from the original sources yourself.

## Contents

| Split | Images | Labels |
|---|---|---|
| train | 4,775 | 4,775 |
| val | 780 | 780 |
| test | 850 | 850 |
| **total** | **6,405** | **6,405** |

~2.1 GB. Layout:

```
train/images/*.jpg   train/labels/*.txt
val/images/*.jpg     val/labels/*.txt
test/images/*.jpg    test/labels/*.txt
drishti.yaml         # Ultralytics data config
```

Labels are YOLO boxes: `class_id x_center y_center width height` (normalised).

## Classes

| id | class |
|---|---|
| 0 | `crab_pot` |
| 1 | `submarine_pipeline` |
| 2 | `shipwreck` |
| 3 | `ghost_net` |
| 4 | `mine_cylinder` |

The shipped product uses 4 — `crab_pot` is trained as a hard negative and filtered downstream.

## Preprocessing — already applied

Every tile has been through **Lee speckle filter + CLAHE** (`despeckle_clahe()` in
`ml/scripts/preprocess_sonar.py`). Do **not** apply it again. If you train on these tiles,
apply the same filter to your inference inputs.

Speckle in sonar is *multiplicative* (`I_obs = I_true · n`), so a plain blur destroys the object
and shadow edges that carry the signal. The Lee filter is a local MMSE estimate that smooths flat
seabed and preserves edges; CLAHE lifts faint contrast with a clip limit so flat-sand speckle
isn't amplified.

Note: an ablation found this preprocessing gave **no accuracy gain** over raw tiles once
training-time augmentation was strong — it is retained because CLAHE'd input makes acoustic
shadows more detectable for the downstream geometry check.

## Provenance — per prefix

Filenames carry their source. Train-split composition:

| Prefix | Count | Source | Nature |
|---|---|---|---|
| `synth` | 1,250 | procedural generator, modelled acoustic physics | **original work** |
| `pipe` | 1,000 | SubPipeMini2 survey strips, tiled | real |
| `cp` | 900 | HuggingFace side-scan crab-pot set | real |
| `wreckA` | 546 | AI4Shipwrecks transects (pixel masks → boxes) | real |
| `bg` | 500 | object-free seabed tiles (hard negatives), mixed sources | real |
| `wreckR` | 354 | Roboflow side-scan-sonar (Ship + Plane) | real |
| `mine` | 225 | Kaggle sonar-mine (MILCO + NonMILCO contacts) | real |

`ghost_net` is **100 % synthetic** — there is no public real ghost-net-in-SSS dataset. A
Microsoft AI for Good / WWF effort had 412 real segments in total and described it as a
feasibility study.

Full source detail: `docs/PROJECT_RECORD.html` §03 and §15 in the GitHub repo.

## Known caveats

- **Not site-disjoint everywhere.** The shipwreck split is being audited to guarantee no wreck
  *site* appears in both train and test. Until confirmed, treat shipwreck metrics as optimistic.
- **The test set is deliberately hard.** A 50 %-overlap re-tile tripled shipwreck test instances
  with partial and near-duplicate tiles. Numbers on this split are not comparable to papers using
  a non-overlapping tiling of the same source.
- **`ghost_net` evaluation is synthetic-on-synthetic.** Not a field number.
- **Class imbalance is deliberate.** Per-class caps per split; a dominant class suppresses accuracy
  on feature-dissimilar classes.

## Usage

```python
from huggingface_hub import snapshot_download
snapshot_download("rehan9599/drishti-sss", repo_type="dataset",
                  local_dir="ml/data/splits", token="hf_...")
```

```bash
yolo detect train data=drishti.yaml model=yolov8s.pt imgsz=640 epochs=120 batch=16
```

## Citation

```bibtex
@software{drishti2026,
  title  = {DRISHTI: AI-Powered Marine Debris Detection from Side-Scan Sonar},
  author = {Fazal, Rehan and others},
  year   = {2026},
  note   = {Smart India Hackathon 2026, Problem Statement 26057},
  url    = {https://github.com/Rehan9599/Sonar-Drishti}
}
```

Please also cite the upstream datasets listed under Provenance.
