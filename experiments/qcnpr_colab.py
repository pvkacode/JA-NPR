"""qcnpr_colab.py  --  quantization-constrained NPR (QC-NPR), version v1

What it tests
-------------
NPR looks at the decoded image.  QC-NPR first replaces a JPEG image by the image that
(a) still fits the JPEG file (every DCT value stays inside the range the quantization
table allows) and (b) is as close as possible to "made by 2x up-sampling" (constant in
every 2x2 cell).  The unchanged NPR network then runs on that image.  For a PNG the
ranges have zero width, so QC-NPR is exactly NPR.

Stages
------
probe : no training.  Released NPR weights, plain input vs QC input.
train : two fine-tunes from the released weights with identical settings:
          plain = clean view + JPEG view            (the current project method)
          qc    = clean view + JPEG view through QC (the new method)
        then both are scored on held-out faces.
all   : probe, then train.

The split, optimiser, epochs, crop and JPEG settings copy scripts/run_jpeg_consistency.py
so the plain arm should land near that script's 0.83 AUC at matched Q95 4:2:0.
"""

import argparse
import csv
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

VERSION = "qcnpr_colab v1"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Standard JPEG base tables (natural row order), scaled by quality the way libjpeg / Pillow do.
BASE_L = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
                   14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
                   18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                   49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float).reshape(8, 8)
BASE_C = np.array([17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
                   24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32, float).reshape(8, 8)


def qtables(q):
    """Luma and chroma quantization tables for a Pillow quality setting."""
    s = 5000 / q if q < 50 else 200 - 2 * q
    f = lambda b: np.clip(np.floor((b * s + 50) / 100), 1, 255)
    return f(BASE_L), f(BASE_C)


# ----------------------------------------------------------------------------- QC front-end
_n = torch.arange(8, dtype=torch.float64)
C8 = torch.sqrt(torch.tensor(2 / 8, dtype=torch.float64)) * torch.cos((2 * _n[None, :] + 1) * _n[:, None] * torch.pi / 16)
C8[0] = C8[0] / np.sqrt(2)                      # orthonormal 8x8 DCT-II matrix


def _blk(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)


def _unblk(x):
    b, c, hb, wb, _, _ = x.shape
    return x.permute(0, 1, 2, 4, 3, 5).reshape(b, c, hb * 8, wb * 8)


def _dct(x, c8):
    return c8 @ _blk(x) @ c8.T


def _idct(d, c8):
    return _unblk(c8.T @ d @ c8)


def cellmean(x):
    """Replace every 2x2 cell by its average: the closest 'made by 2x up-sampling' image."""
    return F.avg_pool2d(x, 2).repeat_interleave(2, -1).repeat_interleave(2, -2)


def rgb2ycc(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([0.299 * r + 0.587 * g + 0.114 * b,
                      128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                      128 + 0.5 * r - 0.418688 * g - 0.081312 * b], 1)


def ycc2rgb(x):
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 128, x[:, 2:3] - 128
    return torch.cat([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], 1)


def _project(plane, table, c8, iters, tol):
    """Point inside the JPEG ranges that is closest to a cell-constant image."""
    d = _dct(plane - 128.0, c8)
    n = torch.round(d / table)
    lo, hi = (n - 0.5) * table - tol, (n + 0.5) * table + tol
    z = plane
    for _ in range(iters):
        z = _idct(torch.minimum(torch.maximum(_dct(cellmean(z) - 128.0, c8), lo), hi), c8) + 128.0
    return z


@torch.no_grad()
def qc_project(rgb, q, sub, iters=25, tol=0.0, flat420=True):
    """rgb: B x 3 x H x W, values 0..255, decoded JPEG pixels.
    q: per-image Pillow quality (0 means 'not a JPEG': returned unchanged).
    sub: 2 for 4:2:0 chroma, 0 for 4:4:4."""
    b, _, h, w = rgb.shape
    if h % 8 or w % 8:
        raise SystemExit(f"QC needs height and width to be multiples of 8, got {h}x{w}")
    q = torch.as_tensor(q).reshape(-1).tolist()
    if len(q) == 1:
        q = q * b
    x = rgb.double()
    c8 = C8.to(x.device)
    tl = torch.stack([torch.from_numpy(qtables(v if v > 0 else 100)[0]) for v in q]).to(x.device).view(b, 1, 1, 1, 8, 8)
    tc = torch.stack([torch.from_numpy(qtables(v if v > 0 else 100)[1]) for v in q]).to(x.device).view(b, 1, 1, 1, 8, 8)
    ycc = rgb2ycc(x)
    y = _project(ycc[:, 0:1], tl, c8, iters, tol)
    if sub == 2:
        # 4:2:0 stores one colour sample per 2x2 cell, so the file says nothing about colour
        # differences inside a cell.  Report "no evidence" instead of "smooth colour".
        c = cellmean(ycc[:, 1:3]) if flat420 else ycc[:, 1:3]
    else:
        c = _project(ycc[:, 1:3], tc, c8, iters, tol)
    out = ycc2rgb(torch.cat([y, c], 1))
    keep = torch.tensor([v <= 0 for v in q], device=x.device).view(b, 1, 1, 1)
    return torch.where(keep, x, out).float()


# ----------------------------------------------------------------------------- data
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_id(path):
    return Path(path).stem.split("_")[0]


def list_split(gen_dir, n_test, seed):
    """Face-grouped split, copied from run_jpeg_consistency.py (group=True)."""
    pairs = []
    for label, name in ((0, "0_real"), (1, "1_fake")):
        d = Path(gen_dir) / name
        if not d.is_dir():
            raise SystemExit(f"missing folder {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if len(files) <= n_test:
            raise SystemExit(f"{d} has {len(files)} images, need more than n_test={n_test}")
        rng = random.Random(seed + label)
        groups = {}
        for p in files:
            groups.setdefault(source_id(p), []).append(p)
        ids = sorted(groups)
        rng.shuffle(ids)
        test_files, train_files = [], []
        for i in ids:
            (test_files if len(test_files) < n_test else train_files).extend(groups[i])
        pairs.append(([(str(p), label) for p in train_files], [(str(p), label) for p in test_files]))
    train = pairs[0][0] + pairs[1][0]
    test = pairs[0][1] + pairs[1][1]
    random.Random(seed).shuffle(train)
    return train, test


def list_all(gen_dir):
    out = []
    for label, name in ((0, "0_real"), (1, "1_fake")):
        d = Path(gen_dir) / name
        if d.is_dir():
            out += [(str(p), label) for p in sorted(d.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return out


def jpeg_image(im, quality, subsampling):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to255(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1)


class TrainSet(Dataset):
    def __init__(self, items, crop, q_lo, q_hi, seed, fresh_crops):
        self.items, self.crop, self.q_lo, self.q_hi, self.seed, self.fresh = items, crop, q_lo, q_hi, seed, fresh_crops

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        im = Image.open(path).convert("RGB")
        # Default copies the original script: one fixed crop and flip per image for all epochs.
        rng = random if self.fresh else random.Random(self.seed + idx * 17 + int(torch.initial_seed() % 100000))
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = im.size
        if w >= self.crop and h >= self.crop:
            x, y = rng.randrange(0, w - self.crop + 1), rng.randrange(0, h - self.crop + 1)
            im = im.crop((x, y, x + self.crop, y + self.crop))
        q = random.randint(self.q_lo, self.q_hi)
        return to255(im), to255(jpeg_image(im, q, 2)), q, label


class EvalSet(Dataset):
    def __init__(self, items, quality, subsampling):
        self.items, self.quality, self.subsampling = items, quality, subsampling

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        im = Image.open(path).convert("RGB")
        if self.quality is not None:
            im = jpeg_image(im, self.quality, self.subsampling)
        return to255(im), label


# ----------------------------------------------------------------------------- model
def load_npr(npr_root, ckpt, dev):
    sys.path.insert(0, str(npr_root))
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(dev)


def norm(x255, dev):
    return (x255 / 255.0 - MEAN.to(dev)) / STD.to(dev)


CONDITIONS = [("orig", None, 2), ("Q95_420", 95, 2), ("Q90_420", 90, 2), ("Q70_420", 70, 2),
              ("Q50_420", 50, 2), ("Q95_444", 95, 0), ("Q75_444", 75, 0)]


@torch.no_grad()
def predict(model, items, dev, quality, sub, use_qc, args):
    model.eval()
    loader = DataLoader(EvalSet(items, quality, sub), batch_size=args.eval_batch, shuffle=False, num_workers=args.workers)
    ys, ps = [], []
    for x, y in loader:
        x = x.to(dev)
        if use_qc and quality is not None:
            x = qc_project(x, quality, sub, args.iters, args.tol, args.chroma420 == "flat")
        ps.append(model(norm(x, dev)).flatten().float().cpu().numpy())   # raw logits: no ties from a saturated sigmoid
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def clusters_of(items):
    cl = {}
    for i, (path, lab) in enumerate(items):
        key = (lab, source_id(path) if lab == 1 else Path(path).stem)
        cl.setdefault(key, []).append(i)
    return cl


def cluster_ci(items, y, p, p2=None, n_boot=1000, seed=0):
    """95% CI resampling whole faces.  With p2, CI of AUC(p2) - AUC(p) on the same resamples."""
    cl = clusters_of(items)
    pos = [v for k, v in cl.items() if k[0] == 1]
    neg = [v for k, v in cl.items() if k[0] == 0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([pos[i] for i in rng.integers(0, len(pos), len(pos))] +
                             [neg[i] for i in rng.integers(0, len(neg), len(neg))])
        a = roc_auc_score(y[idx], p[idx])
        vals.append(roc_auc_score(y[idx], p2[idx]) - a if p2 is not None else a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def train_arm(model, train_items, dev, args, use_qc):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TrainSet(train_items, args.crop, args.q_lo, args.q_hi, args.seed, args.fresh_crops),
                        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        total, n, t0 = 0.0, 0, time.time()
        for clean, jpg, q, y in loader:
            clean, jpg, y = clean.to(dev), jpg.to(dev), y.float().to(dev)
            if use_qc:
                jpg = qc_project(jpg, q, 2, args.iters, args.tol, args.chroma420 == "flat")
            loss = bce(model(norm(clean, dev)).flatten(), y) + bce(model(norm(jpg, dev)).flatten(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * y.shape[0]
            n += y.shape[0]
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {total / max(n, 1):.4f}  {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- checks and reports
def self_check(dev):
    print(VERSION, "| device", dev, flush=True)
    rng = np.random.default_rng(0)
    for q in (95, 90, 75, 70, 50):
        im = Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=q, subsampling=2); buf.seek(0)
        t = Image.open(buf).quantization
        for name, mine, theirs in (("luma", qtables(q)[0], t[0]), ("chroma", qtables(q)[1], t[1])):
            if sorted(int(v) for v in mine.ravel()) != sorted(int(v) for v in theirs):
                raise SystemExit(f"self-check failed: {name} table at Q{q} differs from Pillow's. Stop and report this.")
    small = torch.from_numpy(rng.integers(40, 215, (2, 3, 32, 32)).astype(np.float32))
    up = small.repeat_interleave(2, -1).repeat_interleave(2, -2)          # an ideal 2x up-sampled image
    res = lambda t: float((t - cellmean(t)).abs().mean())
    for sub in (0, 2):
        dec = torch.stack([to255(jpeg_image(Image.fromarray(u.permute(1, 2, 0).numpy().astype(np.uint8)), 90, sub)) for u in up])
        after = qc_project(dec.to(dev), 90, sub).cpu()
        print(f"self-check {'4:2:0' if sub == 2 else '4:4:4'}: up-sampled image, 2x2 residual after JPEG Q90 = {res(dec):.3f}, "
              f"after QC = {res(after):.3f}  (QC should be far smaller)", flush=True)
        if not res(after) < 0.25 * res(dec):
            raise SystemExit("self-check failed: QC did not remove the JPEG noise from an up-sampled image.")
    same = qc_project(up.to(dev), 0, 2).cpu()
    if float((same - up).abs().max()) > 1e-4:
        raise SystemExit("self-check failed: QC changed a non-JPEG image.")
    print("self-check passed: tables match Pillow, QC cleans an up-sampled JPEG, PNG input is unchanged\n", flush=True)


def score_table(tag, model, items, dev, args, use_qc, store, rows, gen="AttGAN", conds=CONDITIONS, ci=True):
    print(f"== {tag} | {gen} ==", flush=True)
    for cond, q, sub in conds:
        y, p = predict(model, items, dev, q, sub, use_qc, args)
        store[(gen, tag, cond)] = (y, p)
        auc = float(roc_auc_score(y, p))
        lo, hi = cluster_ci(items, y, p, n_boot=args.n_boot) if ci else (float("nan"), float("nan"))
        tpr, tnr = float((p[y == 1] > 0).mean()), float((p[y == 0] <= 0).mean())   # logit 0 = probability 0.5
        rows.append({"generator": gen, "model": tag, "condition": cond, "auc": auc, "ci_lo": lo, "ci_hi": hi, "tpr_0.5": tpr, "tnr_0.5": tnr})
        print(f"  {cond:8} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  TPR {tpr:.3f}  TNR {tnr:.3f}", flush=True)


def compare(title, a, b, items, store, args, gen="AttGAN", conds=CONDITIONS):
    print(f"\n{title}\n  condition   {a:>18} {b:>18}   difference [95% CI, whole faces resampled]", flush=True)
    wins = 0
    for cond, _, _ in conds:
        if (gen, a, cond) not in store or (gen, b, cond) not in store:
            continue
        y, pa = store[(gen, a, cond)]
        _, pb = store[(gen, b, cond)]
        lo, hi = cluster_ci(items, y, pa, pb, n_boot=args.n_boot)
        d = roc_auc_score(y, pb) - roc_auc_score(y, pa)
        mark = "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
        wins += lo > 0
        print(f"  {cond:9} {roc_auc_score(y, pa):18.3f} {roc_auc_score(y, pb):18.3f}   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {mark}", flush=True)
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["probe", "train", "all"])
    ap.add_argument("--gen", default="/content/data/AttGAN", help="folder with 0_real and 1_fake")
    ap.add_argument("--npr", default="/content/NPR", help="cloned NPR-DeepfakeDetection repo")
    ap.add_argument("--ckpt", default=None, help="default: <npr>/NPR.pth")
    ap.add_argument("--extra-gens", default="", help="comma list of other generator folders next to --gen, e.g. BEGAN,SNGAN,STGAN")
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--crop", type=int, default=224)
    ap.add_argument("--q-lo", type=int, default=75)
    ap.add_argument("--q-hi", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=25, help="QC projection steps")
    ap.add_argument("--tol", type=float, default=0.0, help="extra slack on each JPEG range")
    ap.add_argument("--chroma420", choices=["flat", "keep"], default="flat",
                    help="4:2:0 colour inside QC: flat = one value per 2x2 cell (default), keep = leave decoded colour as it is")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh-crops", action="store_true", help="new random crop every epoch (the original script reuses one crop per image)")
    ap.add_argument("--out", default="/content/qcnpr_results.csv")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self_check(dev)
    if dev.type != "cuda":
        print("WARNING: no GPU. In Colab choose Runtime > Change runtime type > T4 GPU. Continuing on CPU (slow).\n", flush=True)
    ckpt = args.ckpt or str(Path(args.npr) / "NPR.pth")
    seed_all(args.seed)
    train_items, test_items = list_split(args.gen, args.n_test, args.seed)
    shared = {source_id(p) for p, _ in train_items} & {source_id(p) for p, _ in test_items}
    print(f"train {len(train_items)}  test {len(test_items)}  faces shared by train and test: {len(shared)}\n", flush=True)

    store, rows = {}, []
    extra = [g.strip() for g in args.extra_gens.split(",") if g.strip()]
    xconds = [c for c in CONDITIONS if c[0] in ("orig", "Q95_420", "Q70_420")]

    if args.stage in ("probe", "all"):
        frozen = load_npr(args.npr, ckpt, dev)
        score_table("frozen plain", frozen, test_items, dev, args, False, store, rows)
        score_table("frozen QC", frozen, test_items, dev, args, True, store, rows)
        compare("PROBE (no training): does QC input help the released model?", "frozen plain", "frozen QC", test_items, store, args)
        print("\nHow to read the probe: the 4:4:4 rows are the fair ones. At 4:2:0 QC removes all colour from the\n"
              "residual, which the released weights have never seen, so 4:2:0 rows can only be judged after training.\n", flush=True)
        del frozen

    if args.stage in ("train", "all"):
        for tag, use_qc in (("plain JPEG fine-tune", False), ("QC-NPR fine-tune", True)):
            seed_all(args.seed)
            model = load_npr(args.npr, ckpt, dev)
            print(f"training: {tag}", flush=True)
            train_arm(model, train_items, dev, args, use_qc)
            torch.save(model.state_dict(), str(Path(args.out).with_suffix("")) + ("_qc.pth" if use_qc else "_plain.pth"))
            score_table(tag, model, test_items, dev, args, use_qc, store, rows)
            for g in extra:
                gdir = Path(args.gen).parent / g
                items = list_all(gdir)
                if not items:
                    print(f"  skip {g}: nothing in {gdir}", flush=True)
                    continue
                score_table(f"{tag}", model, items, dev, args, use_qc, store, rows, gen=g, conds=xconds, ci=False)
            del model
        wins = compare("RESULT: QC-NPR against the plain JPEG fine-tune (same data, same settings)",
                       "plain JPEG fine-tune", "QC-NPR fine-tune", test_items, store, args)
        print(f"\nQC-NPR is clearly better on {wins} of {len(CONDITIONS)} conditions.", flush=True)
        print("Anchor: the plain fine-tune should be near AUC 0.83 at Q95_420 (the teammate's result).\n"
              "Pass: QC-NPR 'better' on the 4:2:0 rows, and not 'worse' on orig.\n"
              "Fail: 'no clear difference' or 'worse' on the 4:2:0 rows. Then the method adds nothing over the data change.", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
