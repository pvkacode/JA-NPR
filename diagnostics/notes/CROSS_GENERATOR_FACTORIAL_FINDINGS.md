# Cross-generator factorial findings

BEGAN, SNGAN, and STGAN use the existing 150-real / 150-fake subsets and the same four Q95 arms. They are reported separately. They are not pooled.

BEGAN and SNGAN filenames put every fake into one `celeb_id` cluster, so a face bootstrap would be a single cluster. Their intervals resample images. STGAN has 105 fake-face clusters in this subset, and its intervals resample faces. AttGAN’s face bootstrap is not reused as if these subsets had the same dependence structure.

## BEGAN

| Arm | AUC | 95% interval |
|---|---:|---|
| 4:4:4 | 0.304 | [0.245, 0.362] |
| 4:2:2 | 0.253 | [0.199, 0.309] |
| 4:4:0 | 0.481 | [0.417, 0.546] |
| 4:2:0 | 0.096 | [0.063, 0.132] |

BEGAN is already reversed at 4:4:4. Vertical alone moves it to 0.481, toward chance. Horizontal alone leaves it reversed. Both directions together deepen the reversal to 0.096.

| Contrast | Estimate | 95% interval |
|---|---:|---|
| G_H | −28.9 | [−32.6, −25.2] |
| G_V | −5.0 | [−8.9, −1.1] |
| G_HV | −41.9 | [−48.1, −35.4] |
| AUC_V | +0.010 | [−0.026, 0.048] |
| AUC_I | −0.334 | [−0.398, −0.269] |

The order-averaged vertical effect on AUC includes 0 because the two vertical steps have opposite ranking consequences. Vertical without horizontal raises the class gap by +16.0 [12.2, 19.8]. Vertical after horizontal lowers it by −26.0 [−32.1, −20.1]. The class-gap interaction excludes 0. This is the opposite of AttGAN, where that interaction included 0.

Real and fake scores both fall from 4:4:4 to 4:2:0. The real mean change is −24.1 and the fake mean change is −58.0. This is not the AttGAN pattern, in which reals rise by +30 and fakes fall by −32. BEGAN reals already carry the JPEG-grid confound documented earlier. Their movement under further chroma subsampling is mostly the horizontal factor: real Phi_H is −23.4 and real Phi_V is −0.7. The extra reversal is not something to attribute to fake chroma alone.

Channel class gaps on BEGAN, chroma then luminance: horizontal only −7.0 and −0.3; vertical only +13.6 and −0.8; horizontal given vertical −38.5 and −7.7; vertical given horizontal −17.2 and −8.8. Vertical-only chroma widens the BEGAN gap. The large negative chroma term appears when horizontal subsampling is added on top of vertical subsampling.

## SNGAN

| Arm | AUC | 95% interval |
|---|---:|---|
| 4:4:4 | 0.651 | [0.594, 0.710] |
| 4:2:2 | 0.592 | [0.531, 0.656] |
| 4:4:0 | 0.573 | [0.512, 0.635] |
| 4:2:0 | 0.497 | [0.433, 0.562] |

4:2:0 includes 0.5. SNGAN loses a weak separation. It does not reverse. G_H is −7.1 [−11.0, −3.3], G_V is −9.8 [−13.8, −6.0], and G_HV is −0.4 [−6.1, 5.3]. The interaction includes 0, and the AUC interaction −0.017 [−0.081, 0.048] also includes 0. From 4:4:4 to 4:2:0 the real mean moves by +11.2 and the fake mean by −5.7. Both directions contribute a modest gap reduction. Neither produces the AttGAN reversal.

## STGAN

| Arm | AUC | 95% interval |
|---|---:|---|
| 4:4:4 | 0.893 | [0.853, 0.930] |
| 4:2:2 | 0.749 | [0.680, 0.807] |
| 4:4:0 | 0.807 | [0.753, 0.856] |
| 4:2:0 | 0.486 | [0.414, 0.560] |

4:2:0 includes 0.5. STGAN collapses to chance. The interval does not support a reversal. G_H is −29.3 [−33.2, −25.5] and G_V is −29.2 [−33.7, −24.6]. The two directions are about equal. G_HV is +1.0 [−5.2, 7.5] and includes 0. Real scores rise by +46.3 and fake scores fall by −12.2, so the gap closes. The AUC interaction is −0.178 [−0.237, −0.113] and excludes 0, even though the logit class-gap interaction does not. As on AttGAN, that AUC interaction is the nonlinearity of ranking, not an extra logit term.

## What is shared

On all four generators the order-averaged horizontal class gap is negative and its interval excludes 0. The same is true of the order-averaged vertical class gap. The size is not shared: SNGAN’s gaps are about −7 and −10, while AttGAN and STGAN are several times larger. The class-gap interaction is supported on BEGAN and not supported on AttGAN, SNGAN, or STGAN. A single interaction claim does not describe these four generators.
