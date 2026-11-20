# Pairwise ranking

AttGAN has 400 × 406 = 162,400 real/fake pairs. The margin is the fake logit minus the real logit. AUC is the fraction of positive margins, with half weight for ties. There are no ties in these four arms. The pairwise AUC matches the ROC-AUC of the image scores on each arm.

Pairs share images, so they are not independent samples. The intervals resample the 29 fake faces and the 400 real images, then recompute the pair counts. A pair in which a face is drawn twice is counted twice. That is the same cluster weighting as the AUC interval.

## Where the correct pairs go

Fractions of all pairs. Brackets are 95% cluster intervals.

| Comparison | Stay correct | Correct to wrong | Wrong to correct | Stay wrong | Mean margin change |
|---|---:|---:|---:|---:|---:|
| 4:4:4 → 4:2:2 | 0.519 [0.416, 0.613] | 0.356 [0.291, 0.429] | 0.004 [0.003, 0.007] | 0.120 [0.076, 0.170] | −37.6 [−41.9, −33.4] |
| 4:4:4 → 4:4:0 | 0.765 [0.696, 0.824] | 0.111 [0.085, 0.140] | 0.040 [0.025, 0.058] | 0.084 [0.052, 0.123] | −22.5 [−26.6, −18.3] |
| 4:2:2 → 4:2:0 | 0.189 [0.129, 0.267] | 0.334 [0.251, 0.421] | 0.044 [0.023, 0.068] | 0.433 [0.348, 0.525] | −24.5 [−32.8, −16.2] |
| 4:4:0 → 4:2:0 | 0.228 [0.163, 0.308] | 0.577 [0.500, 0.650] | 0.005 [0.002, 0.009] | 0.190 [0.141, 0.248] | −39.6 [−46.0, −33.2] |
| 4:4:4 → 4:2:0 | 0.226 [0.163, 0.306] | 0.649 [0.564, 0.728] | 0.007 [0.002, 0.012] | 0.118 [0.076, 0.166] | −62.1 [−70.8, −53.0] |

Horizontal alone flips about 36% of pairs from correct to wrong and restores fewer than 1%. Vertical alone flips about 11% and leaves 76% correct. Adding horizontal subsampling on top of vertical subsampling flips 58% of pairs. Adding vertical subsampling on top of horizontal subsampling flips a further 33%, and by then 43% were already wrong. Almost no pairs are repaired.

## Margin sizes

The 4:4:4 to 4:2:0 margin change has median −63, 10th percentile −107, and 90th percentile −16. Only 2.6% of pairs change by less than 5 logits. 88% change by more than 20. At 4:2:0 itself, 77% of margins are negative and 64% have absolute value above 20. The reversal is a broad shift of large margins, not a handful of huge pairs and not a mass of tiny ones sitting on the boundary.

The horizontal-only margin change has median −36, and 74% of those changes exceed 20 logits in absolute value. The vertical-only change has median −22, with 11% smaller than 5 logits and a 90th percentile of +7, so some pairs move the other way. That matches the AUC: vertical alone removes a minority of correct pairs, horizontal alone removes a much larger share, and the combination removes most of them.

These pair counts are the ranking counterpart of the logit gaps. The mean margin change on each edge equals the class-gap change, as it must for a balanced comparison of every real with every fake. The AUC change is the change in the fraction of positive margins, which the transition table accounts for directly.
