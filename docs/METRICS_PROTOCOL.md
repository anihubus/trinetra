# Reporting protocol — read before quoting any number

## The rule

| Metric | Protocol | Why |
|---|---|---|
| **mAP@50, mAP@50-95, per-class AP** | `conf=0.001` (Ultralytics default) | mAP is the area under the *whole* precision–recall curve. A high confidence floor truncates the low-confidence tail and mechanically depresses AP. This is the COCO convention and what every paper we compare against uses. |
| **Precision, Recall, F1, FP-rate** | `conf=0.25, iou=0.6` | Operating-point numbers. Only meaningful at the threshold actually deployed (ahead of Module 2's per-class gate). |

## The error we made

`ml/scripts/evaluate.py` defaults to `conf=0.25, iou=0.6` and we reported its mAP as "mAP".
That number is not comparable to published mAP. Same model, same test set:

| Metric | conf=0.25 (was reported) | conf=0.001 (correct for AP) |
|---|---|---|
| mAP@50 | 0.5796 | **0.6412** |
| mAP@50-95 | 0.4341 | **0.4590** |
| submarine_pipeline AP50 | 0.9844 | **0.9944** |
| ghost_net AP50 | 0.9950 | 0.9950 |
| mine_cylinder AP50 | 0.4240 | **0.5174** |
| shipwreck AP50 | 0.3020 | **0.3696** |
| crab_pot AP50 | 0.1927 | **0.3294** |

We were under-reporting our own model, and the §08 literature comparison was apples-to-oranges.

## Reproduce

```bash
# AP / mAP — standard protocol
python -c "from ultralytics import YOLO; \
YOLO('ml/models/checkpoints/best_detector.pt').val(data='ml/configs/drishti.yaml', split='test', imgsz=640)"

# Precision / recall at the operating point
python ml/scripts/evaluate.py --model ml/models/checkpoints/best_detector.pt
```

## Still to fix

- `docs/PROJECT_RECORD.html` §06/§08 still carry the conf=0.25 mAP figures.
- The Hugging Face model card results table carries them too.
- Run-to-run comparisons (Run 1 / 2 / 3) were all measured at conf=0.25, so the *progression*
  is internally valid — but re-measure all three at standard protocol before quoting them
  against outside work.
