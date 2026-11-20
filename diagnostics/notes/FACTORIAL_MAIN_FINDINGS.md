# Factorial main findings

Frozen NPR, raw logit, matched Q95, full image, no crop. AttGAN is the face-disjoint test: 400 real images, 406 fake images, 29 fake-face clusters, 400 real-image clusters. Intervals are 2,000 cluster resamples. The same draw is applied to all four arms. Real images are resampled as their own clusters, which is the existing `metrics_cluster` procedure. The intervals are not conditional on a fixed real sample. With 29 fake faces, a class-gap effect of only a few logits is not resolved.

The four logits of one image satisfy Phi_H + Phi_V = z_420 − z_444 and I_HV = H1 − H0 = V1 − V0, with a maximum residual of 0 across 1,706 images.

## AttGAN ranking

| Arm | What changed | AUC | 95% interval | AP |
|---|---|---:|---|---:|
| 4:4:4 | neither direction | 0.876 | [0.825, 0.920] | 0.880 |
| 4:2:2 | horizontal only | 0.524 | [0.420, 0.618] | 0.568 |
| 4:4:0 | vertical only | 0.805 | [0.745, 0.855] | 0.788 |
| 4:2:0 | both | 0.233 | [0.167, 0.314] | 0.377 |

Horizontal subsampling by itself takes AttGAN from a useful ranking to chance. The 4:2:2 interval includes 0.5. Vertical subsampling by itself does not. The 4:4:0 interval stays above 0.74. The reversal, AUC 0.233, appears only when both directions are subsampled.

Descriptive AUC contrasts, recomputed inside each bootstrap draw:

| Contrast | Estimate | 95% interval |
|---|---:|---|
| Horizontal, AUC_H | −0.462 | [−0.513, −0.406] |
| Vertical, AUC_V | −0.181 | [−0.239, −0.126] |
| Interaction, AUC_I | −0.220 | [−0.320, −0.114] |

Both main effects and the AUC interaction exclude 0. AUC_I is a contrast of four AUC numbers. It is not an image-level decomposition. The AUC interaction is expected once a nonlinear ranking functional is applied to effects that sit on different parts of the score distribution: vertical alone stays in a high-AUC region, while the same vertical step after horizontal crosses from chance into reversal.

## AttGAN class-gap effects

G is the fake mean minus the real mean. A negative G means the fake-minus-real gap falls, which is the direction that hurts ranking.

| Quantity | Estimate | 95% interval |
|---|---:|---|
| H0, horizontal without vertical | −37.6 | [−41.9, −33.4] |
| H1, horizontal given vertical | −39.6 | [−46.0, −33.2] |
| V0, vertical without horizontal | −22.5 | [−26.6, −18.3] |
| V1, vertical given horizontal | −24.5 | [−32.8, −16.2] |
| G_H, order-averaged horizontal | −38.6 | [−43.0, −33.8] |
| G_V, order-averaged vertical | −23.5 | [−29.4, −17.6] |
| G_HV, class difference of the interaction | −2.0 | [−8.0, 3.9] |
| Total 4:4:4 to 4:2:0 gap change | −62.1 | [−70.8, −53.0] |

G_H and G_V exclude 0. Their sum is the total gap change. G_HV is not added again. G_HV includes 0. On the class-gap scale, this experiment does not establish a horizontal-by-vertical interaction. That is not a demonstration that the interaction is zero. The interval runs from about −8 to +4 logits, and 29 clusters set that width.

The within-image interaction is a different quantity, and it is large in both classes:

| Class | H0 | H1 | V0 | V1 | I_HV |
|---|---:|---:|---:|---:|---:|
| Real | +24.6 [22.5, 26.8] | −18.2 [−19.5, −16.9] | +48.4 [46.6, 50.2] | +5.6 [4.0, 7.1] | −42.8 [−45.5, −40.2] |
| Fake | −13.0 [−16.6, −9.5] | −57.8 [−64.0, −51.3] | +25.9 [22.2, 29.7] | −18.9 [−27.0, −10.9] | −44.9 [−50.0, −39.7] |

Horizontal alone raises real scores and lowers fake scores. Horizontal after vertical subsampling lowers both, and it lowers fakes by much more. Vertical alone raises both classes, reals by more, so the gap still falls. Vertical after horizontal subsampling raises reals only slightly and lowers fakes. The interaction itself is about −43 logits in both classes, so it cancels in G_HV. Order-averaged Phi_V for fakes is +3.5 [−2.3, 9.2] and includes 0, because +26 and −19 cancel. The average is a poor summary of the fake vertical effect. The two simple effects are the ones that describe it.

Which effect is larger depends on the class and on the order. For reals, vertical alone (+48) is larger than horizontal alone (+25). For fakes, horizontal after vertical (−58) is the largest single step, and the order-averaged horizontal effect (−35) is larger than the order-averaged vertical effect (+3.5, unresolved).

Route A (4:4:4 → 4:2:2 → 4:2:0) and route B (4:4:4 → 4:4:0 → 4:2:0) have the same total on every image. The difference between the horizontal step before and after vertical subsampling is I_HV. That identity is exact in the saved contrasts. It is a statement about logits. It is not a statement about gradients, and it is not a statement about chroma versus luminance.
