# JA-NPR: JPEG-Aware Neighboring Pixel Relationships

Compression-robust deepfake detection built on **NPR** (Tan et al., CVPR 2024), a 1.44M-parameter
detector of CNN up-sampling artifacts. JA-NPR keeps NPR's 2×2 residual operator and makes the model
aware of how a JPEG file stores the image. It adds 0.05M parameters (1.488M in total).

| AttGAN test faces, AUC (mean of 3 seeds) | PNG | Q95 | Q90 | Q70 | Q50 | Q30 |
|---|---|---|---|---|---|---|
| Released NPR (base paper; seed 0) | 0.956 | 0.233 | 0.245 | 0.330 | 0.323 | 0.379 |
| NPR fine-tuned on JPEG (same protocol) | 0.999 | 0.799 | 0.810 | 0.762 | 0.676 | 0.574 |
| **JA-NPR (ours)** | **1.000** | **0.922** | **0.895** | **0.825** | **0.788** | **0.702** |

JPEG settings are 4:2:0 at the quality shown. The released NPR model is *inverted* under JPEG:
an AUC below 0.5 means it ranks fakes as more real than real images.

On **STGAN, a generator never seen in training** (trained on AttGAN + SNGAN), AUC at Q95 rises from
0.515 to 0.764, and the method passes the pre-registered test there.

## The method

1. **Chroma-aligned NPR residual (CA-NPR).**
   - NPR is computed on brightness (Y) at full resolution, and on colour (Cb, Cr) at half
     resolution. That is the resolution where a 4:2:0 JPEG file actually stores colour.
   - The first convolution is re-expressed, so the model starts *exactly* at the released NPR
     weights. The script checks this before every run.
   - This part adds no parameters.
2. **Quantization-table-conditioned normalization (QT-Norm).**
   - A 4-number descriptor `q` is read from the file's own JPEG header: whether a table exists,
     luma table strength, chroma table strength, and whether colour is 4:2:0.
   - Every batch-norm layer gets separate running statistics for clean files and JPEG files.
   - Each batch-norm layer also gets a zero-initialised FiLM layer driven by `q`.
   - This part adds 50,560 parameters.

A third component, a compression-adversarial head with gradient reversal, was tested and added
nothing. It is kept in the code as the arm `janpr` for the ablation. **The final method is the arm
`janpr_noadv`.**

### What the controls show (`experiments/npr_t5_colab.py`)

| Control | AttGAN Q95 AUC | Unseen STGAN Q95 AUC |
|---|---|---|
| NPR fine-tuned | 0.794 | 0.515 |
| NPR fine-tuned with fast-learning batch norm (`plain_fastbn`) | 0.839 | 0.546 |
| JA-NPR with the header replaced by a constant (`janpr_constq`) | 0.938 | 0.645 |
| **JA-NPR (reads the header)** | **0.940** | **0.762** |

The unseen-STGAN column comes from the run trained on AttGAN + SNGAN.

- **Not an optimization trick.** JA-NPR beats `plain_fastbn` by +0.101 [+0.051, +0.149] at Q95 on
  AttGAN.
- **On the training generator,** the gain comes from compression-aware normalization itself: the
  constant-input control does as well.
- **On an unseen generator,** reading the quantization table matters: +0.117 [+0.103, +0.132] at
  Q95 on STGAN over the constant-input control.
- **The STGAN header effect replicates on a second seed** (+0.159). Across 7 unseen generators it
  is not consistent: clearly positive on STGAN, neutral on 3, negative on InfoMaxGAN and S3GAN.
- **Compression-aware normalization itself** passes the pre-registered test on 4 to 5 of 7 unseen
  generators. See RESULTS.md §6.

## Repository layout

```
npr_t3_colab.py          main implementation: model, training, evaluation, statistics (version t3)
weights/                 trained JA-NPR weights (state_dict of arm janpr_noadv)
  janpr_AttGAN_seed{0,1,2}.pth     trained on AttGAN
  janpr_AttGAN+SNGAN_seed0.pth     trained on AttGAN+SNGAN (held-out STGAN/BEGAN test)
RESULTS.md               every result in detail (tables, CIs, verdicts, failed alternatives)
results/csv/             every number in the paper, one row per (arm, generator, condition)
results/logs/            full console output of every run, including failed alternatives
results/make_tables.py   rebuilds the paper's tables and figures from results/csv
experiments/             earlier scripts (t1/t2, QC-NPR) and the control scripts t4/t5 (same code as t3 plus control arms)
JA_NPR_colab.ipynb       one-click Colab notebook (setup + all runs)
```

## Reproducing the results (Google Colab, free T4 GPU)

Open `JA_NPR_colab.ipynb` in Colab, or run these cells in a new notebook. First choose
**Runtime → Change runtime type → T4 GPU**.

**1. Setup (about 3 minutes).** This clones the NPR code and released weights, and downloads four
GANGen-Detection generators.

```python
%cd /content
!git clone -q https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git NPR
!git clone -q https://github.com/<your-account>/JA-NPR.git
!pip install -q -U gdown
import gdown, os, subprocess
os.makedirs('/content/data/GANGen-Detection', exist_ok=True); os.chdir('/content/data/GANGen-Detection')
ids = {'AttGAN': '1k7msxU6dS4NPKzE8jxb6AsDx-Kp7MbFn', 'BEGAN': '1ck_j-056-_L0bc1SJD5qt-ADDSxyGNJc',
       'SNGAN': '1L1iS_KdXaShhwNBNf3tGOYhxvRyeypek', 'STGAN': '1ChS-8jejuR1i7cWV0FoR9wjD0b1nvj30'}
for name, fid in ids.items():
    if os.path.isdir(name): continue
    gdown.download(id=fid, output=f'{name}.tar.gz', quiet=True)
    subprocess.run(['tar', '-zxf', f'{name}.tar.gz'], check=True); os.remove(f'{name}.tar.gz')
os.chdir('/content')
```

If a download fails because of a Google Drive rate limit, run the cell again. It skips the
generators already downloaded.

**2. Main experiment (about 60 minutes).** Trains NPR+JPEG, ycdsbn, JA-NPR and JA-NPR+CAdv on
AttGAN, scores the released NPR, and evaluates all of them on AttGAN, BEGAN, SNGAN and STGAN.

```python
!python /content/JA-NPR/npr_t3_colab.py --gen AttGAN --extra-gens BEGAN,SNGAN,STGAN
```

**3. Extra seeds (about 20 minutes each).**

```python
!python /content/JA-NPR/npr_t3_colab.py --gen AttGAN --arms plain,janpr_noadv --seed 1
!python /content/JA-NPR/npr_t3_colab.py --gen AttGAN --arms plain,janpr_noadv --seed 2
```

**4. Unseen generators (about 30 minutes).**

```python
!python /content/JA-NPR/npr_t3_colab.py --train-gens AttGAN,SNGAN --test-gens STGAN,BEGAN --arms plain,janpr_noadv
```

Each run prints two self-checks first:
- every arm starts exactly at the released NPR;
- the JPEG header is read correctly.

Each run then prints per-arm results with 95% confidence intervals, paired comparisons against
the baseline, and a PASS/FAIL verdict under a rule fixed in the code before any run. A CSV of all
results is written to `/content`.

**Rebuild the paper's tables and figures** from the CSVs:

```python
!pip install -q pandas matplotlib
!python /content/JA-NPR/results/make_tables.py
```

## Using the trained weights

This is an example of loading the weights. It is not part of the reported runs. Image width and
height must be even.

```python
import sys, types, torch
sys.path.insert(0, "/content/JA-NPR")
import npr_t3_colab as m
args = types.SimpleNamespace(npr="/content/NPR", ckpt="/content/NPR/NPR.pth")
net = m.make_net("janpr_noadv", args, "cpu")
net.load_state_dict(torch.load("/content/JA-NPR/weights/janpr_AttGAN_seed0.pth", map_location="cpu"))
net.eval()
im, q = m.open_with_q("my_image.jpg")          # q is read from the file's own JPEG header
with torch.no_grad():
    logit, _ = net(m.to255(im)[None], torch.tensor([q]))
print("probability fake:", torch.sigmoid(logit).item())
```

## Evaluation protocol

- **Splits.** AttGAN fakes are edits of only 143 source faces, so the train/test split is grouped
  by face: 3,194 training images, 806 test images, and no shared faces.
- **Confidence intervals.** 95% CIs come from 1,000 bootstrap resamples of whole faces (or of
  single images when names carry no face id). Comparisons are paired.
- **Same protocol for all arms.** Every trained arm uses the same data and settings: Adam, 6
  epochs, batch 16, JPEG quality 30–95, 4:2:0 and 4:4:4. Any difference therefore comes from the
  model.
- **No header leakage.** All 16,000 original images are PNG, and both classes are re-encoded with
  identical JPEG settings. The header therefore carries no class information.

## Limitations

- **Scale.** The study is small: four generators, three seeds (one in the held-out setting).
  Seed 1 narrowly misses the strict test at Q90; the CI is [−0.001, +0.114].
- **BEGAN.** BEGAN's real images carry a hidden JPEG grid, and its absolute results are weak.
- **SNGAN.** SNGAN stays near chance under JPEG.
- **Calibration.** The 0.5 threshold drifts at low JPEG quality.
- **Header value.** The quantization-table content helps on unseen STGAN, not on the training generator.
- **Colour fix.** The chroma-aligned residual's contribution is small and mixed.
- **Shortcut.** Models partly treat old JPEG traces as evidence of a real image (BEGAN control in the t4 log).

## Credits

The base model, architecture and released weights come from
[NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection). That code is
cloned at run time, not redistributed here. The data comes from the GANGen-Detection collection
released with NPR.
