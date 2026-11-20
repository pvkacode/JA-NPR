# Does JA-NPR remove the horizontal-by-vertical JPEG interaction?

## 1. Research question

The released NPR checkpoint has a horizontal-by-vertical interaction on matched Q95 JPEGs. On the AttGAN face-disjoint test, within-image logit interaction is about −44 in both classes, 4:2:2 ranking falls to chance, and 4:2:0 ranking reverses. That interaction is already present at the NPR residual, before the first learned convolution.

JA-NPR is an existing trained model: a chroma-pooled YCbCr residual, dual batch-norm statistics, and FiLM on a four-number JPEG header. This experiment asks whether that model removes the interaction in the representation, or whether ranking improves while the interaction remains.

No model was retrained. No JPEG was rewritten. Thresholds were not chosen from test labels. Score signs were not flipped.

## 2. Models and checkpoints

Released NPR: `NPR-DeepfakeDetection/NPR.pth`, SHA-256 `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`. Input is ImageNet-normalized decoded RGB. The residual is the network’s own `forward`. Recomputed logits match `experiments/npr_chroma_factorial/results/factorial_predictions.csv` with maximum absolute gap 0.

JA-NPR arm `janpr_noadv`, implementation `JA-NPR/npr_t3_colab.py`. Weights, SHA-256:

| Checkpoint | SHA-256 |
|---|---|
| `janpr_AttGAN_seed0.pth` | `ddc2c9545da738c9ce4b69ba86b6f4e03dfc940adb3c94928c755a82e2e0a12b` |
| `janpr_AttGAN_seed1.pth` | `4d10987436841920452caf676409f9a7d7b196a301e436766da3103e59ea97e1` |
| `janpr_AttGAN_seed2.pth` | `2998ef82efcb87ac8b5237cfe5669fab13ec02513738dc8b97415ece4f17db46` |
| `janpr_AttGAN+SNGAN_seed0.pth` | `c9888605be71c2f3dc62ff60947e8d5b5dc74a3afe893e3b322dcb3eb74df98c` |

Activations and the classifier decomposition use AttGAN seed 0. Seeds 1 and 2, and the AttGAN+SNGAN seed, are scored at the logit only.

The plain fine-tuned NPR checkpoint used in the teammate’s tables is not in the repository. It was not scored. Those published Pillow numbers are a different encode and are not cited here as results of this test.

## 3. Dataset and JPEG verification

The four arms are the existing files under `experiments/npr_chroma_factorial/data/jpeg`, listed in `jpeg_factorial_manifest.csv`. Same source path, label, and face id in every arm. Quality is Q95. Encoder settings are the ones already recorded for that factorial: libjpeg-turbo `cjpeg -quality 95 -dct int`. Pixels scored here are `djpeg -dct int` with the default smooth chroma upsample.

Every file was checked before scoring. A mismatch aborts the run. The run finished, so every file matched:

| Arm | SOF sampling factors (Y, Cb, Cr), horizontal × vertical | Pillow `get_sampling` | `q[3]` |
|---|---|---:|---:|
| 4:4:4 | Y 1×1, Cb 1×1, Cr 1×1 | 0 | 0 |
| 4:2:2 | Y 2×1, Cb 1×1, Cr 1×1 | 1 | 0 |
| 4:4:0 | Y 1×2, Cb 1×1, Cr 1×1 | −1 | 0 |
| 4:2:0 | Y 2×2, Cb 1×1, Cr 1×1 | 2 | 1 |

One quantization table, SHA-256 `28d506d794713d12460be674da01ebd14fe2262038e902c081a6946578ac98d3`. Luma uses table 0 and chroma uses table 1, so the table-strength features cannot differ across arms.

AttGAN test: 400 real, 406 fake, 29 fake-face clusters. BEGAN, SNGAN, and STGAN are the existing 150-per-class subsets and are reported separately. BEGAN and SNGAN fake filenames form one cluster, so their intervals are image resamples. STGAN has 105 fake-face clusters.

## 4. JA-NPR preprocessing and conditioning, as executed

`Net.forward` was called with its default `chroma_half=True`.

1. Decoded RGB stays in 0–255. There is no ImageNet normalization on this path.
2. `rgb2ycc` uses the script’s full-range BT.601-style matrix. Cb and Cr are offset by 128.
3. Luma is passed through `nn_residual`: each pixel minus the top-left pixel of its 2×2 cell.
4. Cb and Cr are average-pooled 2×2, then passed through the same residual, then nearest-neighbor expanded by 2. This pool runs for every arm, including 4:4:4 and 4:4:0. It reads decoded pixels. It does not read native JPEG chroma coefficients.
5. The three residual maps are divided by `255 * 0.226` and multiplied by 2/3. That tensor is the input to `conv1`.
6. `conv1` weights are the trained weights. Initialization rewrites `conv1` from the released NPR and then `load_state_dict` replaces it.
7. Every batch-norm is a `QTNorm`. `q[0] = 1` on all four arms, so every arm uses the JPEG running statistics.
8. FiLM is `Linear(4, 2C)` inside each of the 24 `QTNorm` layers. Gamma and beta are applied as `y * (1 + gamma) + beta`.
9. The trunk is the truncated ResNet-50: ReLU, max-pool, layer1, layer2, adaptive average pool to 512, then `fc1` to one logit. Higher logit means more fake.

The instrumented forward was compared with `Net.forward` on every seed-0 batch and aborted above 1e-4. The job completed.

`qvec_of` returns `[has table, log1p(mean luma quantizer)/5, log1p(mean chroma quantizer)/5, 1 if Pillow sampling == 2 else 0]`. On all 1,706 images and all four arms there is one pattern:

`q = [1, 0.38237092, 0.45481139, q3]`

with `q3 = 0, 0, 0, 1` for 4:4:4, 4:2:2, 4:4:0, 4:2:0. Pillow has no sampling code for 4:4:0, so `get_sampling` returns −1 and `q[3]` stays 0.

Across all 24 FiLM layers, gamma and beta for 4:2:2 and for 4:4:0 match 4:4:4 with maximum absolute difference 0. The 4:2:0 vector differs. At `bn1` the difference has L2 norm 0.146 in gamma and 0.127 in beta (64 channels). The largest gamma difference is at `layer2.3.bn3` (512 channels, L2 2.10). Logged mean-absolute `bn1` FiLM is constant across the 806 AttGAN images: gamma 0.0794 and beta 0.0646 on the three arms that share `q`, and gamma 0.0883 and beta 0.0603 on 4:2:0.

The header does not distinguish horizontal-only subsampling, vertical-only subsampling, and no subsampling. Any AUC gap among 4:4:4, 4:2:2, and 4:4:0 is carried by the pixels and the trained weights. Only contrasts that involve 4:2:0 can involve a different FiLM output. This experiment does not swap `q[3]`, so it does not measure how much of the 4:2:0 logit that bit explains.

## 5. Four-arm ranking

Raw logits. No sigmoid. Paired face-cluster bootstrap, 2,000 draws, same images in every arm. AttGAN, seed 0 unless noted.

| Arm | Released NPR AUC | JA-NPR seed 0 AUC | Seed 0 − NPR |
|---|---:|---:|---:|
| 4:4:4 | 0.876 [0.825, 0.920] | 1.000 [1.000, 1.000] | +0.124 [0.079, 0.175] |
| 4:2:2 | 0.524 [0.420, 0.617] | 0.986 [0.976, 0.993] | +0.462 [0.364, 0.556] |
| 4:4:0 | 0.805 [0.745, 0.855] | 0.992 [0.986, 0.996] | +0.187 [0.136, 0.245] |
| 4:2:0 | 0.233 [0.167, 0.314] | 0.941 [0.915, 0.962] | +0.708 [0.622, 0.788] |

Average precision, seed 0: NPR 0.880, 0.568, 0.788, 0.377; JA-NPR 1.000, 0.981, 0.990, 0.932.

Seeds 1 and 2 on the same AttGAN files: 4:2:0 AUC 0.954 [0.934, 0.971] and 0.952 [0.934, 0.969]. 4:4:4, 4:2:2, and 4:4:0 stay at 0.986 or above. The AttGAN ranking gain is present in every shipped AttGAN seed.

Mean logits, AttGAN seed 0. The fake−real gap stays positive on every arm.

| Arm | NPR real | NPR fake | NPR gap | JA real | JA fake | JA gap |
|---|---:|---:|---:|---:|---:|---:|
| 4:4:4 | −98.57 | −58.25 | +40.32 | −5.45 | +3.67 | +9.12 |
| 4:2:2 | −73.96 | −71.23 | +2.73 | −2.96 | +3.39 | +6.35 |
| 4:4:0 | −50.17 | −32.32 | +17.85 | −2.63 | +3.80 | +6.43 |
| 4:2:0 | −68.37 | −90.16 | −21.79 | −1.52 | +3.03 | +4.54 |

Balanced accuracy at logit > 0 is a fixed cut, not a threshold fit on these labels, and it is not a validation-calibrated operating point from this study. For JA-NPR seed 0 it is 0.995, 0.939, 0.929, 0.841. For released NPR it is about 0.50 on every arm, because both class means sit below 0 even when the ranking is intact.

The AUC interaction `AUC(420) − AUC(422) − AUC(440) + AUC(444)` is −0.220 [−0.320, −0.114] for NPR and −0.037 [−0.056, −0.021] for JA-NPR seed 0. A small ranking interaction remains: 4:2:0 is a bit worse than the sum of the one-factor arms would suggest. Both one-factor arms are already above 0.98.

## 6. Logit factorial

`z00, z10, z01, z11` are the 4:4:4, 4:2:2, 4:4:0, and 4:2:0 logits. `H0 = z10 − z00`, `H1 = z11 − z01`, `V0 = z01 − z00`, `V1 = z11 − z10`, `I = z11 − z10 − z01 + z00`, `Phi_H = 0.5 (H0 + H1)`, `Phi_V = 0.5 (V0 + V1)`. On the saved logits, `Phi_H + Phi_V − (z11 − z00)` has maximum absolute value 0.

AttGAN, all 806 images, face-cluster intervals:

| Effect | Released NPR | JA-NPR seed 0 |
|---|---:|---:|
| H0 | +5.68 [3.47, 7.81] | +1.09 [0.96, 1.21] |
| H1 | −38.16 [−41.34, −35.03] | +0.16 [−0.02, 0.31] |
| V0 | +37.08 [34.94, 39.17] | +1.46 [1.36, 1.56] |
| V1 | −6.76 [−10.81, −2.64] | +0.54 [0.39, 0.67] |
| I | −43.84 [−46.63, −40.97] | −0.93 [−1.06, −0.81] |
| Phi_H | −16.24 [−18.52, −13.93] | +0.63 [0.49, 0.76] |
| Phi_V | +15.16 [12.27, 18.05] | +1.00 [0.89, 1.11] |

For released NPR, horizontal subsampling after vertical subsampling is a large negative move, and vertical subsampling alone is a large positive move. For JA-NPR seed 0, both one-factor moves are about +1 logit and the second factor is smaller than the first. `H1`’s interval includes 0. `I` stays negative and its interval excludes 0.

Class-specific logit interaction, AttGAN seed 0:

| Quantity | Released NPR | JA-NPR seed 0 |
|---|---:|---:|
| Real I | −42.81 [−45.48, −40.18] | −1.38 [−1.54, −1.22] |
| Fake I | −44.85 [−49.98, −39.71] | −0.49 [−0.68, −0.29] |
| Fake − real | −2.05 [−8.05, 3.91] | +0.89 [0.63, 1.14] |

Both classes carry a negative within-image interaction under both models. JA-NPR’s two class intervals exclude 0 and exclude each other: the real interaction is larger in magnitude. The paired change in the class-gap interaction, JA minus NPR, is +2.93 [−2.88, 8.80] and includes 0. With 29 fake-face clusters, this experiment does not resolve a change in the class-gap interaction. It does resolve the within-image interaction: JA minus NPR is +42.91 [40.05, 45.85].

The logit scale also changed. NPR’s 4:4:4 class gap is +40 and `|I|` is about the same size. JA-NPR’s 4:4:4 class gap is +9.1 and `|I|` is 0.93, about one tenth of that gap. The shrinkage of `I` is smaller than a pure rescaling of the NPR logits would produce.

Seeds 1 and 2 have AttGAN `I` of −1.87 [−1.99, −1.75] and −2.23 [−2.33, −2.12]. Same sign, same order of magnitude, still far from −44.

## 7. Activation factorial

Norms use the full tensor. The interaction ratio is `||I|| / (||H0|| + ||V0||)`. Cosine similarities use L2 norms. Stage identity `||Phi_H + Phi_V − (A11 − A00)||` is at most 5.8×10⁻⁵.

JA-NPR’s residual is a different operator from NPR’s RGB residual. Raw L2 norms are not compared across models. Ratios are dimensionless and can be compared as summaries of each operator. AttGAN, 806 images:

| Stage | Ratio | ||I|| | ||H0|| | ||V0|| | cos(H0, V0) | cos(4:4:4, 4:2:0) |
|---|---:|---:|---:|---:|---:|---:|
| YCbCr | 0.628 | 452.0 | 364.8 | 361.8 | 0.413 | 1.000 |
| Luma NPR | 0.682 | 87.8 | 64.6 | 64.3 | 0.505 | 1.000 |
| Pooled chroma | 0.657 | 179.9 | 137.6 | 137.1 | 0.349 | 1.000 |
| Chroma NPR, expanded | 0.618 | 399.8 | 329.7 | 318.3 | 0.396 | 0.858 |
| Residual | 0.620 | 7.10 | 5.83 | 5.64 | 0.400 | 0.987 |
| Residual × 2/3 | 0.620 | 4.74 | 3.89 | 3.76 | 0.400 | 0.987 |
| conv1 | 0.635 | 9.34 | 7.43 | 7.31 | 0.401 | 0.994 |
| bn1, before ReLU | 0.633 | 357.6 | 281.1 | 285.1 | 0.435 | 0.936 |
| ReLU | 0.633 | 240.9 | 189.5 | 192.4 | 0.444 | 0.943 |
| max-pool | 0.600 | 163.5 | 136.4 | 138.1 | 0.486 | 0.972 |
| layer1 | 0.685 | 1244 | 913 | 907 | 0.466 | 0.712 |
| layer2 | 0.842 | 1450 | 872 | 862 | 0.479 | 0.428 |
| Pooled 512 | 3.578 | 13.60 | 2.31 | 2.30 | 0.630 | 0.844 |

The ratio is 0.63 in decoded YCbCr, 0.62 after the chroma NPR, and 0.62 at the residual that enters `conv1`. The 2/3 scale does not change it. FiLM begins at `bn1` and does not remove it: the ratio is still 0.63 there. It rises through layer2 (0.84) and jumps at global average pooling.

That jump is inside one representation family. At layer2, `||I||` and `||H0||` are the same spatial shape and `||I||` is larger but not dominant. After pooling, both are 512-vectors, `||H0||` and `||V0||` are 2.3, and `||I||` is 13.6. The pooled interaction is large relative to the pooled one-factor effects. Raw L2 values from earlier stages are different shapes and are not a measure of growth.

Paired with the already measured NPR residual, same 806 paths:

| Ratio | NPR | JA-NPR seed 0 | JA − NPR |
|---|---:|---:|---:|
| Residual | 0.591 [0.587, 0.595] | 0.620 [0.619, 0.622] | +0.029 [0.025, 0.034] |
| Pooled 512 | 0.783 [0.705, 0.869] | 3.578 [3.422, 3.733] | +2.80 [2.65, 2.95] |

JA-NPR’s residual ratio is slightly higher. The chroma-aligned residual leaves the interaction in place. The pooled representation makes the interaction a larger share of the factorial contrast than it is for released NPR.

Per-class residual ratios: NPR real 0.578 [0.574, 0.583], fake 0.604 [0.596, 0.611]; JA-NPR real 0.622 [0.621, 0.624], fake 0.619 [0.617, 0.621].

## 8. Real versus fake direction

At the JA-NPR residual, the mean interaction vector of the 400 reals and the mean of the 406 fakes have cosine 0.177. Those mean vectors have L2 norms 0.43 and 0.57, while the average per-image `||I||` is 7.10. Image-level residual interactions point in many directions, so the class means cancel. The same pattern holds for the luma NPR (cosine −0.006) and the expanded chroma NPR (cosine 0.182).

At the pooled 512, the class-mean interaction vectors have cosine 0.976, norms 13.43 and 13.26, and the gap vector has norm 2.91. The average per-image `||I||` is 13.60, so the per-image pooled interactions agree with the class mean. By the classifier input, real and fake images share one interaction direction.

The logit projection of that shared direction is negative in both classes, and larger for reals (−1.38) than for fakes (−0.49).

## 9. Classifier decomposition

`fc1` is linear. With the pooled interaction `I_c` and the classifier weight `w`,

`I_logit = Σ_c w_c I_c`.

Reconstruction error against the logit interaction has maximum absolute value 1.55×10⁻⁶. The bias does not enter.

Mean over 806 AttGAN images: positive channels sum to +12.62, negative channels sum to −13.55, and the total is −0.93. There are 210 positive and 302 negative channels. The largest-magnitude 20 channels hold 11% of the absolute mass. 126 channels hold half of the absolute mass, and 253 hold 80%. The signed top-20 share is not a useful fraction here, because the net is a small difference of two large opposite sums.

Released NPR, from the earlier activation study on these same images, had a mean logit interaction of −43.84, a positive-channel sum of +0.77, and a negative-channel sum of −44.60. About 102 channels held half the absolute mass and 211 held 80%. Both decompositions are broad. JA-NPR’s is broad and two-sided. The small logit interaction is the remainder after cancellation across the 512 channels, in front of a pooled interaction whose L2 norm is 13.6.

## 10. Bootstrap

AttGAN and STGAN: resample fake-face clusters and, independently, real images as their own clusters. The same draw is applied to all four arms and, for paired contrasts, to both models. BEGAN and SNGAN: image resampling, because the fake names collapse to one cluster. Those intervals are not face-clustered intervals. Channels were not treated as samples. Two thousand replicates. AttGAN paired contrasts of seed 0 against released NPR are in sections 5 and 6.

## 11. Other generators

AttGAN-trained JA-NPR seed 0, not pooled with AttGAN. NPR values are the recomputed logits on the same files.

| Split | Arm | NPR AUC | JA-NPR AUC | JA − NPR |
|---|---|---:|---:|---:|
| BEGAN, n=300, image bootstrap | 4:4:4 | 0.304 [0.245, 0.362] | 0.797 [0.744, 0.846] | +0.492 [0.417, 0.562] |
| BEGAN | 4:2:2 | 0.253 [0.199, 0.309] | 0.702 [0.639, 0.763] | +0.449 [0.370, 0.527] |
| BEGAN | 4:4:0 | 0.481 [0.417, 0.545] | 0.706 [0.645, 0.766] | +0.225 [0.139, 0.311] |
| BEGAN | 4:2:0 | 0.096 [0.063, 0.132] | 0.712 [0.654, 0.768] | +0.617 [0.547, 0.686] |
| SNGAN, n=300, image bootstrap | 4:4:4 | 0.651 [0.593, 0.710] | 0.621 [0.558, 0.682] | −0.030 [−0.115, 0.049] |
| SNGAN | 4:2:2 | 0.592 [0.531, 0.656] | 0.544 [0.478, 0.605] | −0.049 [−0.145, 0.041] |
| SNGAN | 4:4:0 | 0.573 [0.512, 0.635] | 0.549 [0.485, 0.613] | −0.024 [−0.111, 0.058] |
| SNGAN | 4:2:0 | 0.497 [0.433, 0.562] | 0.490 [0.424, 0.552] | −0.007 [−0.117, 0.096] |
| STGAN, n=300, 105 fake faces | 4:4:4 | 0.893 [0.853, 0.930] | 0.839 [0.787, 0.888] | −0.055 [−0.120, 0.010] |
| STGAN | 4:2:2 | 0.749 [0.680, 0.806] | 0.692 [0.628, 0.756] | −0.058 [−0.142, 0.034] |
| STGAN | 4:4:0 | 0.807 [0.753, 0.856] | 0.652 [0.590, 0.717] | −0.155 [−0.230, −0.083] |
| STGAN | 4:2:0 | 0.486 [0.414, 0.560] | 0.573 [0.507, 0.642] | +0.088 [−0.019, 0.196] |

On this BEGAN subset the AttGAN-trained model raises every arm, including 4:2:0 from 0.10 to 0.71. The interval is an image resample. BEGAN’s real JPEG-grid confound is a property of those files; this table does not explain it. Logit `I` is −1.07 [−1.35, −0.81].

On SNGAN, every AUC difference includes 0. Seed 0 logit `I` is −0.06 [−0.40, 0.27]. Seed 1’s SNGAN `I` is −3.31 [−3.67, −2.95] while 4:2:0 AUC is 0.487 [0.423, 0.550]: a shared within-image shift with no ranking gain. The AttGAN+SNGAN checkpoint, which saw SNGAN in training, still has SNGAN 4:2:0 AUC 0.554 [0.492, 0.614].

On STGAN, 4:4:0 AUC falls by 0.155 and the interval excludes 0. The 4:2:0 gain includes 0. Logit `I` still shrinks, from −42.2 to −0.78 [−1.09, −0.46].

Seeds 1 and 2 repeat the AttGAN pattern and the SNGAN near-chance 4:2:0 result. Their BEGAN 4:2:0 AUCs are 0.688 and 0.674.

## 12. Limitations

The plain JPEG fine-tune is absent, so this comparison cannot separate “trained on JPEG” from “FiLM plus chroma pooling.” The teammate’s ablation is not re-estimated here.

`q[3]` was not swapped. FiLM can affect only the 4:2:0 arm relative to the other three, and the size of that effect is unmeasured.

The residual ratio compares two different operators. A higher JA-NPR ratio means that operator still contains a large interaction. It is not a claim about the same coordinates.

BEGAN and SNGAN intervals are image resamples. The 4:4:4 AttGAN AUC is saturated, so its interval collapses to [1, 1] under this bootstrap. Global-pool L2 norms are not comparable to spatial-map L2 norms. The decoder and the Q95 table were held fixed. 4:4:0 is an experimental control.

## 13. Outcome

**Outcome B — downstream compensation.**

The residual interaction ratio is 0.620 [0.619, 0.622], slightly above the released NPR residual ratio of 0.591 [0.587, 0.595]. The interaction is already in decoded YCbCr and is still there after chroma pooling, the NPR operator, `conv1`, and FiLM at `bn1`. AttGAN 4:2:0 AUC rises from 0.233 to 0.941, and the within-image logit interaction falls from −43.8 to −0.93. The pooled 512-vector carries a larger interaction ratio (3.58), shared by real and fake. The linear classifier cancels it across channels (+12.6 against −13.5).

This is not outcome A: the residual interaction is not removed. It is not outcome C: the decision-level interaction does change, by about 43 logits, and the change is large relative to JA-NPR’s own class gap. It is not outcome E on AttGAN: the paired AUC gains exclude 0 on every arm and in every AttGAN seed. Outcome D is the wrong summary for AttGAN, where the remaining logit interaction is about one logit and the remaining AUC interaction is −0.037. The residual interaction is still substantial, which is why the outcome is compensation downstream of that residual, including a classifier that cancels a large pooled interaction.

SNGAN does not show a ranking gain. That limit sits beside outcome B. It does not turn the AttGAN result into outcome E.

## 14. What this establishes

On these exact Q95 files, the shipped JA-NPR weights rank the AttGAN test better than released NPR on every arm, including 4:2:0, in all three AttGAN seeds. They do this with a residual whose horizontal-by-vertical interaction ratio is at least as large as NPR’s. The header gives 4:4:4, 4:2:2, and 4:4:0 the same conditioning, so the high AUC on 4:2:2 and 4:4:0 is a property of the trained pixel path. The small AttGAN logit interaction is a cancellation in `fc1` of a pooled interaction that is large and shared by both classes.

This does not establish that FiLM is the cause, that the chroma pool extracts JPEG chroma, that the interaction is gone from the representation, or that the gain holds for SNGAN. It does not establish a result for the missing plain fine-tune.

## Next experiment

Rescore this seed-0 checkpoint on these same four files twice: once with the true `q`, and once with `q[3]` forced to 0 on the 4:2:0 arm (and, separately, forced to 1 on all four). No retraining. The difference is the header’s contribution to the 4:2:0 logit and AUC. If the plain fine-tune weights can be recovered from the same training script, score them on these files as well. Until that exists, the published Pillow table should stay out of this comparison.
