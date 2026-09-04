# demo/ — runnable samples

Three **synthetic** `ghost_net` tiles (567 KB) so a fresh `git clone` can run the detector
immediately, in Colab or locally, with nothing else to download.

These are **our own procedurally-generated tiles** — freely reusable, no third-party licence.
They are not field data: `ghost_net` metrics are synthetic-on-synthetic and are a
proof-of-capability, not a field number.

For **real** sonar tiles (shipwreck / pipeline / cylinder), pull the dataset repo — access
required, see the top-level README:

```bash
hf download rehan9599/drishti-sss --repo-type dataset --local-dir ml/data/splits
```

## Run

```bash
python edge/edge_infer.py --image demo/tiles/synth_ghost_net_00001.jpg --no-preprocess
```

`--no-preprocess` because these tiles are already Lee+CLAHE filtered (they came from the
preprocessed splits). For a **raw** sonar image, drop the flag.

Labels are YOLO boxes: `class_id x_center y_center width height`, normalised.
Class ids: 0 crab_pot · 1 submarine_pipeline · 2 shipwreck · 3 ghost_net · 4 mine_cylinder.
