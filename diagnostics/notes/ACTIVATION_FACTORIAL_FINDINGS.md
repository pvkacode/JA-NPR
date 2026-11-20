# Activation-space factorial findings

## 1. Research question

Where in the frozen NPR network do the horizontal and vertical JPEG subsampling effects, and their interaction, appear, and which classifier channels carry the logit interaction?

The four arms are the existing Q95 factorial: 4:4:4, 4:2:2, 4:4:0, and 4:2:0, on the same source images. The checkpoint is the released NPR weight file. Nothing was trained, and no score was flipped.

## 2. Setup

AttGAN is the face-disjoint test used before: 400 real images, 406 fake images, 29 fake-face clusters, 400 real-image clusters. BEGAN, SNGAN, and STGAN are the same 150-per-class subsets. JPEG bytes are the files already stored by the factorial experiment. They were decoded with `djpeg -dct int` and the default smooth upsampler, then ImageNet-normalized as a full image. No crop and no sigmoid.

The instrumented forward matches `model.forward`. The largest absolute logit difference from `factorial_predictions.csv` is 3.1e-5. Checkpoint SHA-256 is `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`, 1,437,761 parameters, `eval` mode.

Intervals are 2,000 resamples. Fake faces are clusters. Each real image is its own cluster. The same draw is applied to all four arms. BEGAN and SNGAN filenames collapse every fake into one cluster, so those two intervals resample images. With 29 AttGAN fake faces, small class gaps are not resolved.

## 3. Architecture and recorded stages

The forward in `networks/resnet.py` is, in order:

| Stage | What is recorded |
|---|---|
| `npr` | `x - nearest_down_up(x, 0.5)`, before scaling |
| `npr_scaled` | that residual times 2/3, the input of `conv1` |
| `conv1` | 3×3 convolution, stride 2, 64 channels |
| `bn1` | batch-norm, copied out before the inplace ReLU |
| `relu` | ReLU |
| `maxpool` | 3×3 max-pool, stride 2 |
| `layer1` | three bottleneck blocks, 256 channels |
| `layer2` | four bottleneck blocks, stride 2, 512 channels |
| `pre_fc` | adaptive average pool, flattened to 512 |

`fc1` is `Linear(512, 1)` with a bias. The pool output and `pre_fc` are the same 512 numbers. There is no nonlinearity between them and the logit.

Full spatial maps were not written to disk. Every norm and cosine below was computed from the full tensor and then the tensor was discarded. The 512-d vector was kept, because that is the input of the classifier.

## 4. Definitions

For a representation `A`, with arms `00, 10, 01, 11`:

- `H0 = A10 - A00`, `H1 = A11 - A01`
- `V0 = A01 - A00`, `V1 = A11 - A10`
- `I = A11 - A10 - A01 + A00`
- `Phi_H = 0.5 (H0 + H1)`, `Phi_V = 0.5 (V0 + V1)`

`Phi_H + Phi_V = A11 - A00` and `H1 - H0 = V1 - V0 = I` hold algebraically. The largest L2 norm of either residual, over every image and every stage, is 2.2e-4. That is floating-point error, not a result.

The cross-stage comparisons use two dimensionless ratios.

- `||I|| / ||A00||`, the interaction relative to that stage's 4:4:4 activation.
- `||I|| / (||H0|| + ||V0|| + 1e-8)`, the interaction relative to the two simple effects that start from 4:4:4.

Raw L2 norms are not compared across stages. `layer1` has hundreds of times more elements than the NPR residual, so a larger raw L2 there is mostly dimension. `||I|| / ||A00||` is also sensitive to whether the stage subtracts a mean. Batch-norm does that with fixed running statistics, which shrinks `||A00||` without creating a new interaction. The ratio of the effect norms to each other is the quantity that survives a common rescaling.

## 5. Main AttGAN results

Mean over 806 images. Brackets are 95% face-cluster intervals.

| Stage | `||I|| / (||H0||+||V0||)` | `||I|| / ||A00||` | `||H0|| / ||A00||` | `||V0|| / ||A00||` | cosine of 4:4:4 with 4:2:0 |
|---|---:|---:|---:|---:|---:|
| `npr` | 0.591 [0.587, 0.595] | 0.149 [0.143, 0.155] | 0.129 | 0.127 | 0.990 |
| `npr_scaled` | 0.591 [0.587, 0.595] | 0.149 [0.144, 0.156] | 0.129 | 0.127 | 0.990 |
| `conv1` | 0.607 [0.602, 0.611] | 0.116 [0.112, 0.122] | 0.096 | 0.098 | 0.994 |
| `bn1` | 0.592 [0.588, 0.597] | 0.495 [0.482, 0.508] | 0.413 | 0.428 | 0.891 |
| `relu` | 0.593 [0.589, 0.598] | 0.462 [0.451, 0.474] | 0.385 | 0.400 | 0.905 |
| `maxpool` | 0.549 [0.542, 0.556] | 0.276 [0.269, 0.283] | 0.244 | 0.269 | 0.954 |
| `layer1` | 0.690 [0.685, 0.694] | 1.086 [1.074, 1.099] | 0.795 | 0.775 | 0.632 |
| `layer2` | 0.682 [0.675, 0.689] | 1.180 [1.144, 1.217] | 0.872 | 0.843 | 0.560 |
| `pre_fc` | 0.783 [0.698, 0.863] | 0.475 [0.423, 0.525] | 0.260 | 0.365 | 0.976 |

The interaction ratio is already 0.59 at the NPR residual, before `conv1`. Multiplying by 2/3 does not change it: every norm scales by the same constant. `conv1`, batch-norm, ReLU, and max-pool leave the ratio between 0.55 and 0.61. The jump in `||I|| / ||A00||` at `bn1` is the denominator. Batch-norm recenters `A00`. The effect-to-effect ratio does not move.

`layer1` is where the four activation maps stop being near-duplicates. Cosine between 4:4:4 and 4:2:0 falls from 0.95 after max-pool to 0.63, and `||I||` exceeds `||A00||`. The effect-to-effect ratio only rises from 0.55 to 0.69. Pooling then lifts that ratio to 0.78 [0.70, 0.86]. The interaction is present in the input of the first learned layer. Later stages change how large it is relative to the activation, and they decorrelate the four maps. They are not the first place the non-additivity exists.

At the NPR residual the two simple effects are similar in size and positively aligned: cosine of `H0` with `V0` is 0.514 [0.511, 0.517]. The interaction is negatively aligned with `H0`, cosine −0.692 [−0.695, −0.688]. That is the representation version of the logit fact that the second subsampling step does not continue the first.

Channel energy of the interaction is not sparse. At `npr` there are only three channels, and the participation ratio is 2.51 of a maximum of 3, so the interaction is shared across the three input channels. A "top 10 channel" fraction is meaningless there because there are not 10 channels. At `layer1` the top 10 channels hold 0.111 [0.109, 0.112] of the interaction energy, and the participation ratio is 180 of 256 channels. At `layer2` the top 10 hold 0.081 and the participation ratio is 252 of 512. At `pre_fc` the top 10 hold 0.102 [0.098, 0.105] and the participation ratio is 208 of 512.

## 6. Class comparison

`||I||` itself is slightly smaller for fakes than for reals at the NPR residual: fake minus real is −1.07 [−1.63, −0.56] on a real-image mean of 11.5. The relative ratio differs by only 0.026 [0.017, 0.034].

At `pre_fc` the raw interaction norms are 26.6 [25.2, 28.2] for reals and 25.9 [22.8, 28.9] for fakes. Their difference is −0.70 [−4.13, 2.68] and includes 0. The ratio `||I|| / (||H0||+||V0||)` is 0.548 [0.530, 0.568] for reals and 1.015 [0.849, 1.173] for fakes. The gap, 0.467 [0.299, 0.625], excludes 0 because the fake vertical simple effect is smaller: `||V0||` differs by −14.3 [−16.3, −12.1]. A larger ratio is not a larger interaction.

The class-mean interaction vectors at `pre_fc` have cosine 0.991. Their norms are 25.06 and 25.16. The norm of their difference is 3.36. The direction of the average interaction is shared. It is not a direction that separates the classes. That matches the logit result already on record: within each class the logit interaction is about −43, and the class-gap interaction `G_HV` is −2.0 with an interval that includes 0.

## 7. Classifier decomposition

Because `z = wᵀ p + b`, the bias and any component common to all four arms cancel, and

`sum_c w_c I_c = I_logit`.

The largest absolute reconstruction error on an AttGAN image is 2.5e-5. The mean logit interaction is −43.84.

The mean over images of the channel contributions sums to that same −43.84. The positive channels sum to +0.77. The negative channels sum to −44.60. The logit interaction is a broad negative sum, not a cancellation between two large opposite groups.

The largest 20 channels by absolute mean contribution are 117, 415, 325, 385, 231, 41, 172, 181, 333, 423, 246, 359, 199, 36, 142, 503, 376, 143, 106, and 386. Each mean contribution is between −0.35 and −0.26. Together they are 13.4% of the signed mean and 12.9% of the absolute mass. Half of the absolute mass requires 102 channels. Eighty percent requires 211. Two hundred ninety-three channels have a mean absolute contribution above 0.05. No small set of channels accounts for the interaction.

The top 20 are stable under the face bootstrap. Mean Jaccard with the point-estimate top 20 is 0.97, interval [0.90, 1.00]. Eighteen of those channels are in the top 20 of every one of the 2,000 draws. Channel 386 is in 75% of them. The real-image top 20 and the fake-image top 20 share 18 channels. Their mean contributions have the same sign and almost the same size. Stability of the ranking is not the same thing as those channels explaining the effect.

Nothing in the weight or the contribution identifies a channel as a chroma detector or a frequency detector. The weights on these 20 channels happen to be negative and of similar size, about −0.12 to −0.16. That is a property of this linear layer, not a label for the channel.

## 8. Other generators

The same interaction ratio is already present at the NPR residual: BEGAN 0.651 [0.649, 0.653], SNGAN 0.596 [0.591, 0.601], STGAN 0.568 [0.563, 0.573]. At `pre_fc` it is 1.20, 0.78, and 0.64. BEGAN and SNGAN use image resampling because their fake filenames form one cluster. The pattern is not an AttGAN peculiarity. It is also not evidence that the logit interaction has the same class-gap consequence on each generator. Those logit results are in the factorial write-up.

## 9. Figures

All seven are in `experiments/npr_activation_factorial/figures/`. They are AttGAN image means, with cluster intervals where a bar has a tick.

- `stage_interaction_magnitude.png` plots `||I|| / ||A00||`. The rise at `bn1` is a smaller reference norm, described above.
- `stage_horizontal_vertical_magnitude.png` plots the two simple effects on the same relative scale. They stay close to each other until the pooled vector, where the vertical effect is larger.
- `stage_interaction_ratio.png` is the effect-to-effect ratio. It is high at the first stage and stays high.
- `stage_real_fake_interaction.png` splits that relative-to-`A00` curve by class.
- `classifier_contribution_distribution.png` is the histogram of 512 image-averaged channel contributions.
- `classifier_top20_channels.png` is the signed mean of the 20 largest, with the bootstrap inclusion frequency under each bar.
- `representation_vs_logit_interaction.png` plots, for each image, `||I||` of the pooled vector against the logit interaction. The cloud is not a line. The scalar `wᵀ I` is the logit interaction. A large representation interaction and the logit interaction are different numbers.

## 10. What the similarities do and do not say

Per-image cosine between two arms is the alignment of the full activation, including the part that is common to both. It stays near 0.99 at the NPR residual: the subsampling change is a small angle on a large feature. By `layer2` the cosine between 4:4:4 and 4:2:0 is 0.56 [0.55, 0.57]. After pooling it returns to 0.98, because average pooling removes spatial disagreement that the cosine of the map was counting.

Relative L2 is `||A_arm - A_444|| / ||A_444||`, which is the simple-effect column in the table for the two single-direction arms, and `||I||` is not that distance. Cosine does not say that one arm caused the other, and a high cosine does not mean the logit is unchanged. The logit moves by tens of units while the early features remain within a few degrees.

Centered kernel alignment was not used. The class-mean interaction vectors above are already contrasts, so a further centering of the four raw arms would answer a different question.

## 11. What this establishes

The horizontal-by-vertical interaction in the activation is already in the NPR residual, at a ratio of 0.59, with both simple effects about 0.13 of the 4:4:4 feature and a negative alignment between the interaction and the horizontal step. The 2/3 scale does not change that geometry. Learned layers keep the effect-to-effect ratio in a band from about 0.55 to 0.78 and, from `layer1` onward, make the four maps less alike. The logit interaction is exactly the linear image of the pooled interaction. That image is a sum of a few tenths of a logit across hundreds of channels, shared by real and fake images.

## 12. What this does not establish

It does not establish that a late layer creates the interaction, that `bn1` or `layer1` is its origin because a relative-to-baseline curve rises there, or that any channel detects chroma. It does not establish a class-differential interaction in the pooled vector: the norms of `I` agree across classes within the interval, and the class-mean directions agree. It does not replace the logit factorial. A large `||I||` is not, by itself, a change in ranking. The decoder was held fixed, so this is the activation consequence of the four encoder settings, not a separation of subsampling from upsampling.

## 13. Recommended next experiment

The non-additivity is in the three-channel NPR residual and is spread across the classifier. A useful next measurement is geometric, inside that residual: how the 2×2 NPR cell responds when the missing chroma samples are horizontal, vertical, or both. A search for a chroma channel, or another correction layer, is not supported by the spread of the contributions.
