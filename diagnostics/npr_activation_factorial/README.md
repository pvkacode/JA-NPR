# Activation-space factorial

Frozen released NPR. The four Q95 JPEG arms are the files already written by `experiments/npr_chroma_factorial`. This directory does not retrain, does not edit the checkpoint, and does not change the split.

## Forward stages

Names are the modules in `ResNet.forward`:

| Stage | Tensor |
|---|---|
| `npr` | `x - interpolate(x, 0.5)` before the 2/3 scale |
| `npr_scaled` | that residual times 2/3, the input of `conv1` |
| `conv1` | first convolution, 64 channels, stride 2 |
| `bn1` | batch-norm, cloned before the inplace ReLU |
| `relu` | ReLU |
| `maxpool` | 3×3 max-pool, stride 2 |
| `layer1` | three bottleneck blocks, 256 channels |
| `layer2` | four bottleneck blocks, stride 2, 512 channels |
| `pre_fc` | adaptive average pool flattened to 512. This is the input of `fc1` |

`fc1` is `Linear(512, 1)`. Pooling and the pre-classifier vector are the same 512 numbers. The logit is `wᵀ pre_fc + b`.

## What is saved

Full spatial maps are not stored. A float32 copy of `layer1` for the AttGAN test alone is about 13 GB, and the reported norms do not need the maps after the reductions. For every image and stage the script stores the L2 norms of the factorial effects, cosines, mean absolute changes, a top-10 channel energy fraction, and a participation ratio. Those reductions are computed from the full tensor before it is discarded. The 512-d `pre_fc` vector is saved for every arm, because the classifier decomposition is exact only there.

## Run

```bash
.venv/bin/python experiments/npr_activation_factorial/extract_activations.py
.venv/bin/python experiments/npr_activation_factorial/analyze_factorial.py
```

Extraction is resumable through `results/stage_effects.csv` and `results/pre_fc_shards/`. A logit that differs from `factorial_predictions.csv` by more than 1e-3 aborts the run.
