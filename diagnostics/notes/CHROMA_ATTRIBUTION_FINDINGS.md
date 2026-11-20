# Chroma attribution

Date: 2026-11-19. Frozen NPR. Quality 95. Pillow, same `jpeg_pil` call as the logit audit.

Regenerated AttGAN logits match `attgan_test_logits.csv` to a maximum absolute error of \(5\times 10^{-6}\) (2,418 paired scores: clean, Q95 4:4:4, Q95 4:2:0). The raw-logit AUCs are the ones already reported: AttGAN test clean 0.956, 4:4:4 0.876, 4:2:2 0.524, 4:2:0 0.233.

The colour mix is BT.601 full range, then clip to 8-bit. Round-trip max pixel error on these images is 1 level. Shapley sums \(\phi_Y+\phi_C-(z_{11}-z_{00})\) are 0 to numerical noise. The four images of each contrast are the converted images, and they sit within one level of the JPEG decode.

## AttGAN face-disjoint test

806 images, 29 fake faces. Intervals are 2,000 face-cluster resamples of the fake-minus-real mean. \(D_C\) is that gap for chroma. \(D_Y\) is the same gap for luminance.

| Contrast | \(D_Y\) | \(D_C\) | \(D_C-D_Y\) | Raw logit gap, fake − real |
|---|---:|---:|---:|---:|
| Clean → 4:4:4 | −7.6 [−10.4, −4.8] | **+12.9 [6.0, 20.4]** | +20.5 [15.3, 26.1] | +1.6 [−8.3, 11.9] |
| Clean → 4:2:2 | −16.3 [−18.7, −13.9] | −15.7 [−24.3, −6.6] | +0.6 [−6.9, 8.0] | −36.0 [−47.2, −24.6] |
| Clean → 4:2:0 | −12.4 [−13.8, −10.9] | **−46.1 [−53.4, −38.7]** | −33.7 [−40.3, −27.2] | −60.6 [−68.8, −52.0] |
| 4:4:4 → 4:2:2 | −11.7 [−12.7, −10.9] | **−25.7 [−29.4, −22.1]** | −13.9 [−17.1, −10.9] | −37.6 [−42.1, −33.4] |
| 4:4:4 → 4:2:0 | −17.1 [−18.7, −15.6] | **−46.7 [−54.5, −39.4]** | −29.6 [−36.9, −22.7] | −62.1 [−71.3, −53.9] |

4:4:4 does not reverse AttGAN. Its chroma term is positive: both classes are pushed toward “real” (mean \(\phi_C\) about −60 for reals and −48 for fakes), and fakes are pushed less, so the class gap stays. The raw gap’s interval covers zero, and the AUC stays 0.876.

4:2:0 reverses the ranking. From 4:4:4 to 4:2:0, chroma accounts for −46.7 of the class-gap change and luminance for −17.1. Their sum, −63.8, matches the raw logit-gap change of −62.1 within the colour-conversion discrepancy of about one level. Chroma is the larger term. The interval on \(D_C-D_Y\) excludes zero.

The class-conditional means for 4:4:4 → 4:2:0:

| Class | Mean \(\phi_Y\) | Mean \(\phi_C\) |
|---|---:|---:|
| Real | −0.8 | **+30.4** |
| Fake | −17.9 | **−16.3** |

Reals become more “fake” when 4:4:4 chroma is replaced by 4:2:0 chroma. Fakes become more “real”. Luma also pulls fakes down and barely moves reals. The crossing of the two class means is this pair of shifts, not a uniform bias.

## Horizontal and vertical

Raw AUC on this same test: 4:4:4 **0.876**, 4:2:2 **0.524**, 4:2:0 **0.233**.

4:2:2, which subsamples chroma horizontally, takes a working ranking to chance. The chroma Shapley gap for 4:4:4 → 4:2:2 is −25.7. Going on to 4:2:0, which also subsamples vertically, moves the chroma gap to −46.7 and the AUC from 0.52 to 0.23. Horizontal subsampling removes the ranking. The extra vertical step is what turns chance into a reversal. The pilot’s claim that horizontal subsampling already produced most of the reversal was too strong. On the full test it produces most of the collapse to chance, and the vertical step is a separate, large chroma effect. This comparison is still two JPEG bitstreams, not a pure vertical kernel with quantization held fixed.

## Other generators

150 images per class. BEGAN and SNGAN intervals are image resamples, because every fake filename shares one prefix. STGAN uses face clusters.

Clean → 4:2:0, \(D_C\) versus \(D_Y\):

| Generator | \(D_C\) | \(D_Y\) |
|---|---:|---:|
| BEGAN | −199 [−205, −192] | −18.5 [−20.9, −16.2] |
| SNGAN | −80 [−87, −73] | −6.0 [−7.6, −4.4] |
| STGAN | −163 [−170, −157] | −9.9 [−11.6, −8.4] |

Chroma dominates the clean-to-4:2:0 class gap on all three. That is not the same statement as “4:2:0 reverses every generator.” SNGAN and STGAN raw 4:2:0 AUCs are 0.497 and 0.486. Their large negative \(D_C\) from the clean image is mostly the loss of a clean gap that was already gone by 4:4:4’s standard, not a fresh reversal inside 4:2:0. From 4:4:4 to 4:2:0, \(D_C\) is −23 on BEGAN, −13 on SNGAN, and −47 on STGAN. BEGAN is already reversed at 4:4:4 (AUC 0.304).

Per-image rows: `experiments/npr_chroma_sensitivity/results/channel_shapley.csv`. Intervals: `channel_shapley_bootstrap.csv`.
