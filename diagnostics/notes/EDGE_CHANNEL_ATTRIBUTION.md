# Edge channel attribution

Each of the four factorial edges is split into a luminance change and a chroma change with the existing BT.601 full-range mix. For an edge from arm A to arm B, z00 is the round trip of A, z11 is the round trip of B, z10 keeps chroma from A and takes luminance from B, and z01 keeps luminance from A and takes chroma from B. Phi_Y + Phi_C equals z11 − z00 on every image. The maximum residual in the saved table is 0.

The round-trip pixel error is at most 1 gray level. The round-trip logit is not the decoded logit. Median absolute differences are 1–2 logits. On the AttGAN vertical-only edge the signed mean error for fakes is −3.0 logits, and 6.7% of those fakes differ by more than 10. Class means below are used. Individual tail values are not treated as the decoded-logit attribution. See `docs/FACTORIAL_ENCODER_VALIDATION.md`.

Numbers are fake-mean minus real-mean, AttGAN, 2,000 face-cluster resamples.

| Edge | Phi_Y | Phi_C | Chroma minus luminance | Real Phi_C | Fake Phi_C |
|---|---:|---:|---:|---:|---:|
| 4:4:4 → 4:2:2, horizontal only | −11.7 [−12.7, −10.9] | −25.7 [−29.7, −22.3] | −13.9 [−17.3, −11.0] | +18.2 | −7.5 |
| 4:4:4 → 4:4:0, vertical only | −7.2 [−8.3, −6.1] | −16.9 [−20.8, −13.1] | −9.7 [−13.8, −5.4] | +40.3 | +23.4 |
| 4:4:0 → 4:2:0, horizontal given vertical | −8.0 [−9.4, −6.6] | −31.7 [−37.5, −26.3] | −23.7 [−29.0, −18.5] | −11.4 | −43.1 |
| 4:2:2 → 4:2:0, vertical given horizontal | −7.7 [−8.8, −6.5] | −18.8 [−25.6, −11.9] | −11.1 [−17.6, −4.8] | +10.4 | −8.4 |

Chroma is the larger share of the class-gap change on every edge. Of the hybrid class-gap change, chroma is about 69% on horizontal alone (−25.7 of −37.4), 70% on vertical alone (−16.9 of −24.1), 80% on horizontal given vertical (−31.7 of −39.7), and 71% on vertical given horizontal (−18.8 of −26.4). Luminance is negative on every edge and is not negligible. The predeclared chroma-minus-luminance contrasts on the two single-direction edges both exclude 0.

The class-conditional direction is not the same on every edge.

- Horizontal alone moves chroma in opposite directions: reals up by 18, fakes down by 8.
- Vertical alone moves chroma in the same direction: reals up by 40, fakes up by 23. The gap falls because reals move further, not because fakes reverse.
- Horizontal after vertical subsampling moves both classes down, fakes by 43 and reals by 11.
- Vertical after horizontal subsampling returns to opposite directions: reals up by 10, fakes down by 8.

So vertical chroma does have a class-conditional pattern different from horizontal chroma, and that pattern itself changes once the other direction is present. The chroma-minus-luminance class gap is −14, −10, −24, and −11 across the four edges. The excess of chroma over luminance is largest for horizontal subsampling when vertical subsampling is already there.

Averaging the chroma class gaps along the two routes from 4:4:4 to 4:2:0 gives about −46.5. Averaging the luminance class gaps gives about −17.3. The earlier direct 4:4:4-to-4:2:0 decomposition was −46.7 for chroma and −17.1 for luminance. The edge split agrees with that result. It does not replace it, and the sum of edge Shapley values is not an algebraic identity for the direct edge.

The same tables for BEGAN, SNGAN, and STGAN are in `results/edge_channel_shapley_bootstrap.csv` and are interpreted per generator in `docs/CROSS_GENERATOR_FACTORIAL_FINDINGS.md`.
