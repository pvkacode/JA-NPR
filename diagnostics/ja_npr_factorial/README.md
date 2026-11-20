# JA-NPR on the existing four-arm JPEG factorial

Controlled scoring of models that already exist. Nothing here retrains a network, edits a JPEG, or chooses a threshold from test labels.

## What is scored

- Released NPR, checkpoint `NPR-DeepfakeDetection/NPR.pth`, SHA-256 `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`. Input is ImageNet-normalized decoded RGB and the original NPR residual inside `ResNet.forward`.
- JA-NPR `janpr_noadv` weights shipped with the teammate repo: AttGAN seeds 0, 1, and 2, and AttGAN+SNGAN seed 0. SHA-256 is in `results/run_config.json`. The implementation is `JA-NPR/npr_t3_colab.py` (same file as `/tmp/janpr/JA-NPR/npr_t3_colab.py`).
- No plain fine-tuned NPR checkpoint is in that repo. That arm is not scored.

## JPEG arms

The four files already written by `experiments/npr_chroma_factorial` at Q95 with libjpeg-turbo `cjpeg -quality 95 -dct int`. Pixels are `djpeg -dct int` with the default smooth upsample. Each file is rejected unless:

- the SOF sampling factors are 4:4:4, 4:2:2, 4:4:0, or 4:2:0 as named;
- the DQT hash is `28d506d794713d12460be674da01ebd14fe2262038e902c081a6946578ac98d3`;
- Pillow `get_sampling` is 0, 1, -1, 2 for those four arms;
- `qvec_of` sets `q[3]` to 1 only for 4:2:0.

## JA-NPR path that is actually scored

`chroma_half=True`, which is the default of `Net.forward`. Decoded RGB in 0–255 is converted with the script's `rgb2ycc`. Luma gets `nn_residual`. Chroma is average-pooled 2×2, then `nn_residual`, then nearest-neighbor expanded back to full size. That pool runs for every arm, including 4:4:4 and 4:4:0. It is not a read of the JPEG chroma coefficients. The residual is divided by `255 * 0.226` and multiplied by 2/3 before `conv1`. QT-Norm uses the JPEG domain (`q[0] = 1`) on every arm. FiLM is an affine function of the four-number header. Instrumented logits are checked against `Net.forward` to 1e-4. Released-NPR logits are checked against `factorial_predictions.csv` to 1e-3.

## Run

```bash
.venv/bin/python -u experiments/ja_npr_factorial/evaluate_models.py --batch-size 4
.venv/bin/python experiments/ja_npr_factorial/analyze_factorial.py
.venv/bin/python experiments/ja_npr_factorial/analyze_conditioning.py
```

`extract_activations.py` calls the same scoring entry point. Do not start a second PyTorch process on MPS while scoring is running.

## Outputs

`results/logits.csv` has one row per image per model. AUC uses the raw logit. `results/ja_stage_effects.csv` has the factorial L2 norms and cosine similarities for JA-NPR AttGAN seed 0. `results/ja_pre_fc_shards/` holds the 512-D pooled vector in arm order 444, 422, 440, 420. Bootstrap intervals use 2,000 paired draws. AttGAN and STGAN resample face clusters. BEGAN and SNGAN fall back to image resampling when every fake filename is one cluster.
