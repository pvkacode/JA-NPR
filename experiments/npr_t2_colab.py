"""npr_t2_colab.py  --  version t2

Arms (all start from the released NPR weights, same settings, same seed):

  plain     NPR fine-tuned on clean + JPEG views (the baseline)
  ycnpr     brightness/colour NPR: 2x2 residual on brightness at full size, on colour at HALF size
  dsbn      separate batch-norm statistics for clean files and JPEG files
  ycdsbn    ycnpr + dsbn together (new in t2)
  featcons  plain + feature-level consistency (use --lam-feat 10 for a fair-strength test)

Two modes:

  single     (t1 behaviour) train on one generator's face-grouped split (--gen), test on its held-out
             part, optionally also score other generators (--extra-gens). Reproduces the t1 numbers.
  held-out   train on several generators (--train-gens), test on generators never seen in training
             (--test-gens). Pass rules are judged on the held-out generators.

New in t2: a confidence interval on every row, including other generators. Where file names carry
a face id (AttGAN-style "<id>_<edit>.png") whole faces are resampled; otherwise each image is
resampled on its own. The grouping used is printed for every folder.
"""

import argparse
import csv
import io
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

VERSION = "npr_t2_colab t2"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
YC_SCALE = 0.226  # typical ImageNet std, so the new residual has the same size as NPR's
IMG_EXT = {".png", ".jpg", ".jpeg"}


# ----------------------------------------------------------------------------- data
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_id(path):
    return Path(path).stem.split("_")[0]


@lru_cache(maxsize=None)
def folder_uses_ids(folder):
    """True if '<id>_...' names in this folder form real groups (many groups, none huge).
    Decided on the WHOLE folder, so a small test subset is grouped the same way as the full set."""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT]
    groups = {}
    for p in files:
        groups[source_id(p)] = groups.get(source_id(p), 0) + 1
    return bool(files) and len(groups) >= 50 and max(groups.values()) <= 0.05 * len(files)


def group_key(path):
    p = Path(path)
    return (str(p.parent), source_id(p) if folder_uses_ids(str(p.parent)) else p.stem)


def describe_grouping(gen_dir):
    out = []
    for name in ("0_real", "1_fake"):
        d = Path(gen_dir) / name
        if d.is_dir():
            out.append(f"{name}: {'by face id' if folder_uses_ids(str(d)) else 'per image'}")
    return f"{Path(gen_dir).name} resampling " + ", ".join(out)


def list_split(gen_dir, n_test, seed):
    """Face-grouped split; identical to t1 when names carry face ids."""
    pairs = []
    for label, name in ((0, "0_real"), (1, "1_fake")):
        d = Path(gen_dir) / name
        if not d.is_dir():
            raise SystemExit(f"missing folder {d}")
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if len(files) <= n_test:
            raise SystemExit(f"{d} has {len(files)} images, need more than n_test={n_test}")
        rng = random.Random(seed + label)
        groups = {}
        for p in files:
            groups.setdefault(group_key(p)[1], []).append(p)
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
            out += [(str(p), label) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
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
        # Default copies the original script (one fixed crop/flip per image) so `plain` reproduces 0.83.
        rng = random if self.fresh else random.Random(self.seed + idx * 17 + int(torch.initial_seed() % 100000))
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = im.size
        if w < self.crop or h < self.crop:
            raise SystemExit(f"{path} is {w}x{h}, smaller than --crop {self.crop}. Use a smaller --crop.")
        x, y = rng.randrange(0, w - self.crop + 1), rng.randrange(0, h - self.crop + 1)
        im = im.crop((x, y, x + self.crop, y + self.crop))
        q = random.randint(self.q_lo, self.q_hi)
        return to255(im), to255(jpeg_image(im, q, 2)), label


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


# ----------------------------------------------------------------------------- colour helpers
def rgb2ycc(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([0.299 * r + 0.587 * g + 0.114 * b,
                      128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                      128 + 0.5 * r - 0.418688 * g - 0.081312 * b], 1)


def ycc2rgb(x):
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 128, x[:, 2:3] - 128
    return torch.cat([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], 1)


YCC2RGB = torch.tensor([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]])


def nn_residual(p):
    """NPR's own operation: each pixel minus the top-left pixel of its 2x2 cell."""
    return p - p[:, :, 0::2, 0::2].repeat_interleave(2, -1).repeat_interleave(2, -2)


# ----------------------------------------------------------------------------- model
class DualBN(nn.Module):
    """Batch norm with two sets of running statistics: 0 = no JPEG table (clean), 1 = JPEG."""

    def __init__(self, bn):
        super().__init__()
        self.weight, self.bias = bn.weight, bn.bias
        self.eps, self.momentum = bn.eps, bn.momentum
        for d in (0, 1):
            self.register_buffer(f"rm{d}", bn.running_mean.clone())
            self.register_buffer(f"rv{d}", bn.running_var.clone())
        self.domain = 0

    def forward(self, x):
        rm, rv = (self.rm0, self.rv0) if self.domain == 0 else (self.rm1, self.rv1)
        return F.batch_norm(x, rm, rv, self.weight, self.bias, self.training, self.momentum, self.eps)


def swap_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, DualBN(child))
        else:
            swap_bn(child)


class Net(nn.Module):
    """The released NPR network with a choice of input residual, returning (logit, pooled features)."""

    def __init__(self, base, residual="rgb", dual_bn=False):
        super().__init__()
        self.base, self.mode = base, residual
        if dual_bn:
            swap_bn(self.base)
        if residual == "yc":
            # Re-express conv1 so that, if colour were used at full size, the network would give
            # exactly the released NPR's output.  Only the half-size colour step changes the input.
            w = self.base.conv1.weight.data                                       # out x 3(rgb) x 3 x 3
            m = (YCC2RGB / STD.view(3, 1)).to(w)                                  # rgb_c = sum_k M[c,k] ycc_k, per-channel ImageNet std
            self.base.conv1.weight.data = YC_SCALE * torch.einsum("ocij,ck->okij", w, m)

    def set_domain(self, d):
        for m in self.modules():
            if isinstance(m, DualBN):
                m.domain = d

    def residual(self, x255, chroma_half=True):
        if self.mode == "rgb":
            x = (x255 / 255.0 - MEAN.to(x255.device)) / STD.to(x255.device)
            return x - self.base.interpolate(x, 0.5)
        ycc = rgb2ycc(x255)
        ry = nn_residual(ycc[:, 0:1])
        if chroma_half:
            c = F.avg_pool2d(ycc[:, 1:3], 2)                                        # the colour the 4:2:0 file stores
            rc = nn_residual(c).repeat_interleave(2, -1).repeat_interleave(2, -2)   # 2x2 cells at half size = 4x4 in the image
        else:
            rc = nn_residual(ycc[:, 1:3])
        return torch.cat([ry, rc], 1) / (255.0 * YC_SCALE)

    def forward(self, x255, domain=0, chroma_half=True):
        self.set_domain(domain)
        b = self.base
        h = b.conv1(self.residual(x255, chroma_half) * 2.0 / 3.0)
        h = b.maxpool(b.relu(b.bn1(h)))
        h = b.layer2(b.layer1(h))
        f = b.avgpool(h).flatten(1)
        return b.fc1(f).flatten(), f


def load_base(npr_root, ckpt):
    sys.path.insert(0, str(npr_root))
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


ARMS = {"plain": dict(residual="rgb", dual_bn=False, featcons=False),
        "ycnpr": dict(residual="yc", dual_bn=False, featcons=False),
        "dsbn": dict(residual="rgb", dual_bn=True, featcons=False),
        "ycdsbn": dict(residual="yc", dual_bn=True, featcons=False),
        "featcons": dict(residual="rgb", dual_bn=False, featcons=True)}


def make_net(arm, args, dev):
    cfg = ARMS[arm]
    return Net(load_base(args.npr, args.ckpt), cfg["residual"], cfg["dual_bn"]).to(dev)


def train_arm(net, arm, train_items, dev, args):
    cfg = ARMS[arm]
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loader = DataLoader(TrainSet(train_items, args.crop, args.q_lo, args.q_hi, args.seed, args.fresh_crops),
                        batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    bce = nn.BCEWithLogitsLoss()
    net.train()
    for epoch in range(args.epochs):
        total, n, t0 = 0.0, 0, time.time()
        for clean, jpg, y in loader:
            clean, jpg, y = clean.to(dev), jpg.to(dev), y.float().to(dev)
            lc, fc = net(clean, domain=0)
            lj, fj = net(jpg, domain=1)
            loss = bce(lc, y) + bce(lj, y)
            if cfg["featcons"]:
                loss = loss + args.lam_feat * (1 - F.cosine_similarity(fc.detach(), fj, dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * y.shape[0]
            n += y.shape[0]
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {total / max(n, 1):.4f}  {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- evaluation
CONDITIONS = [("orig", None, 2), ("Q95_420", 95, 2), ("Q90_420", 90, 2), ("Q70_420", 70, 2),
              ("Q50_420", 50, 2), ("Q95_444", 95, 0), ("Q75_444", 75, 0)]
SHORT_CONDS = [c for c in CONDITIONS if c[0] in ("orig", "Q95_420", "Q90_420", "Q70_420")]


def colour_variant(x255, how):
    if how == "normal":
        return x255
    ycc = rgb2ycc(x255)
    if how == "cellmean":   # colour replaced by one value per 2x2 cell: removes the decoder's invented colour detail
        c = F.avg_pool2d(ycc[:, 1:3], 2).repeat_interleave(2, -1).repeat_interleave(2, -2)
    else:                   # "gray": colour removed completely
        c = torch.full_like(ycc[:, 1:3], 128.0)
    return ycc2rgb(torch.cat([ycc[:, 0:1], c], 1))


@torch.no_grad()
def predict(net, items, dev, quality, sub, args, colour="normal"):
    net.eval()
    loader = DataLoader(EvalSet(items, quality, sub), batch_size=args.eval_batch, shuffle=False, num_workers=args.workers)
    ys, ps = [], []
    for x, y in loader:
        x = colour_variant(x.to(dev), colour)
        logit, _ = net(x, domain=0 if quality is None else 1)   # the file's table decides; only dsbn arms react
        ps.append(logit.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def clusters_of(items):
    """Resampling units: whole faces where names carry face ids, otherwise single images."""
    cl = {}
    for i, (path, lab) in enumerate(items):
        cl.setdefault((lab,) + group_key(path), []).append(i)
    return [v for k, v in cl.items() if k[0] == 1], [v for k, v in cl.items() if k[0] == 0]


def bal_acc(y, p):
    return 0.5 * (float((p[y == 1] > 0).mean()) + float((p[y == 0] <= 0).mean()))


def boot_indices(items, n_boot, seed=0):
    pos, neg = clusters_of(items)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([pos[i] for i in rng.integers(0, len(pos), len(pos))] +
                             [neg[i] for i in rng.integers(0, len(neg), len(neg))])


def paired_ci(items, y, pa, pb, metric, n_boot):
    """95% CI of metric(pb) - metric(pa), resampling the same units for both."""
    vals = [metric(y[i], pb[i]) - metric(y[i], pa[i]) for i in boot_indices(items, n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def single_ci(items, y, p, n_boot):
    vals = [roc_auc_score(y[i], p[i]) for i in boot_indices(items, n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score(arm, net, items, dev, args, store, rows, gen, conds, role):
    print(f"== {arm} | {gen} ({role}) ==", flush=True)
    for cond, q, sub in conds:
        y, p = predict(net, items, dev, q, sub, args)
        store[(gen, arm, cond)] = (y, p)
        auc = float(roc_auc_score(y, p))
        lo, hi = single_ci(items, y, p, args.n_boot)
        tpr, tnr = float((p[y == 1] > 0).mean()), float((p[y == 0] <= 0).mean())
        rows.append({"seed": args.seed, "train": args.train_label, "generator": gen, "role": role, "arm": arm,
                     "condition": cond, "auc": auc, "ci_lo": lo, "ci_hi": hi,
                     "bal_acc_0.5": (tpr + tnr) / 2, "tpr_0.5": tpr, "tnr_0.5": tnr})
        print(f"  {cond:8} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  at 0.5: bal.acc {(tpr + tnr) / 2:.3f} "
              f"(fakes caught {tpr:.3f}, reals passed {tnr:.3f})", flush=True)


AUC_RULE = "AUC clearly better than plain on Q95_420 and Q90_420, and AUC not clearly worse on orig"
BAL_RULE = "balanced accuracy at 0.5 clearly better than plain on Q95_420 and Q90_420, and AUC not clearly worse on Q95_420"
RULES = {
    "ycnpr": ("AUC", "PASS if " + AUC_RULE + "."),
    "featcons": ("AUC", "PASS if " + AUC_RULE + "."),
    "dsbn": ("BAL", "PASS if " + BAL_RULE + "."),
    "ycdsbn": ("BOTH", "PASS if " + AUC_RULE + ", AND balanced accuracy at 0.5 clearly better on Q95_420 and Q90_420."),
}


def compare(arm, gen, items, store, args, conds, judge):
    """Paired comparison of `arm` against `plain` on one generator. Returns PASS/FAIL if judge, else None."""
    kind, rule = RULES[arm]
    head = f"\n### {arm} vs plain on {gen}"
    print(head + (f"\nRule (fixed before the run): {rule}" if judge else "  (information only, no verdict)"), flush=True)
    print(f"  {'condition':9} {'AUC plain':>9} {arm + ' AUC':>14}  {'dAUC [95% CI]':>24}   "
          f"{'bal.acc plain':>13} {arm + ' bal.acc':>16}  {'d bal.acc [95% CI]':>24}", flush=True)
    res = {}
    for cond, _, _ in conds:
        y, pa = store[(gen, "plain", cond)]
        _, pb = store[(gen, arm, cond)]
        a_lo, a_hi = paired_ci(items, y, pa, pb, roc_auc_score, args.n_boot)
        b_lo, b_hi = paired_ci(items, y, pa, pb, bal_acc, args.n_boot)
        da = roc_auc_score(y, pb) - roc_auc_score(y, pa)
        db = bal_acc(y, pb) - bal_acc(y, pa)
        res[cond] = (a_lo, a_hi, b_lo, b_hi)
        print(f"  {cond:9} {roc_auc_score(y, pa):9.3f} {roc_auc_score(y, pb):14.3f}  {da:+.3f} [{a_lo:+.3f}, {a_hi:+.3f}]   "
              f"{bal_acc(y, pa):13.3f} {bal_acc(y, pb):16.3f}  {db:+.3f} [{b_lo:+.3f}, {b_hi:+.3f}]", flush=True)
    if not judge:
        return None
    key = ["Q95_420", "Q90_420"]
    auc_ok = all(res[c][0] > 0 for c in key) and not res["orig"][1] < 0
    bal_ok = all(res[c][2] > 0 for c in key) and not res["Q95_420"][1] < 0
    ok = {"AUC": auc_ok, "BAL": bal_ok, "BOTH": auc_ok and bal_ok}[kind]
    print(f"VERDICT {arm} on {gen}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


@torch.no_grad()
def colour_probe(net, items, dev, args):
    print("\n### Probe (no training): does the plain model rely on the colour detail the decoder invents?", flush=True)
    print("  normal = as decoded; cellmean = one colour value per 2x2 cell; gray = no colour at all", flush=True)
    for cond, q, sub in [("orig", None, 2), ("Q95_420", 95, 2), ("Q90_420", 90, 2), ("Q95_444", 95, 0)]:
        line = []
        for how in ("normal", "cellmean", "gray"):
            y, p = predict(net, items, dev, q, sub, args, colour=how)
            line.append(f"{how} {roc_auc_score(y, p):.3f}")
        print(f"  {cond:8} AUC  " + "   ".join(line), flush=True)


# ----------------------------------------------------------------------------- checks
def self_check(args, dev):
    print(VERSION, "| device", dev, flush=True)
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64, device=dev) * 255
    ref = make_net("plain", args, dev).eval()
    with torch.no_grad():
        a, _ = ref(x)
        for arm in ("ycnpr", "ycdsbn"):
            b, _ = make_net(arm, args, dev).eval()(x, domain=1, chroma_half=False)
            err = float((a - b).abs().max())
            print(f"self-check: {arm} with full-size colour reproduces the released NPR, max diff {err:.2e}", flush=True)
            if err > 1e-3 * max(1.0, float(a.abs().max())):
                raise SystemExit(f"self-check failed for {arm}: the first-layer conversion is wrong. Stop and report this.")
        c, _ = make_net("dsbn", args, dev).eval()(x, domain=1)
    if float((a - c).abs().max()) > 1e-4 * max(1.0, float(a.abs().max())):
        raise SystemExit("self-check failed: dsbn does not start from the released weights. Stop and report this.")
    print("self-check passed: all arms start from the released NPR\n", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="plain,ycnpr,dsbn,ycdsbn")
    ap.add_argument("--root", default="/content/data/GANGen-Detection", help="folder holding one sub-folder per generator")
    ap.add_argument("--gen", default="AttGAN", help="single mode: the training generator (name under --root, or a path)")
    ap.add_argument("--extra-gens", default="", help="single mode: other generators to score, e.g. BEGAN,SNGAN,STGAN")
    ap.add_argument("--train-gens", default="", help="held-out mode: e.g. AttGAN,SNGAN,STGAN")
    ap.add_argument("--test-gens", default="", help="held-out mode: generators never used in training, e.g. BEGAN")
    ap.add_argument("--npr", default="/content/NPR")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lam-feat", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=None, help="default 224 in single mode, 128 in held-out mode")
    ap.add_argument("--q-lo", type=int, default=75)
    ap.add_argument("--q-hi", type=int, default=95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh-crops", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.ckpt = args.ckpt or str(Path(args.npr) / "NPR.pth")

    def gen_path(g):
        return Path(g) if "/" in g else Path(args.root) / g

    held_out = bool(args.train_gens)
    if held_out:
        train_gens = [g.strip() for g in args.train_gens.split(",") if g.strip()]
        test_gens = [g.strip() for g in args.test_gens.split(",") if g.strip()]
        if not test_gens:
            raise SystemExit("held-out mode needs --test-gens")
        if set(train_gens) & set(test_gens):
            raise SystemExit("a generator cannot be in both --train-gens and --test-gens")
        args.crop = args.crop or 128
        args.train_label = "+".join(train_gens)
        tag = f"heldout_{'+'.join(test_gens)}"
    else:
        train_gens = [args.gen]
        test_gens = [g.strip() for g in args.extra_gens.split(",") if g.strip()]
        args.crop = args.crop or 224
        args.train_label = Path(args.gen).name
        tag = f"single_{Path(args.gen).name}"
    args.out = args.out or f"/content/npr_t2_{tag}_seed{args.seed}.csv"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self_check(args, dev)
    if dev.type != "cuda":
        print("WARNING: no GPU. In Colab: Runtime > Change runtime type > T4 GPU.\n", flush=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a}; choose from {list(ARMS)}")
    if "plain" not in arms:
        raise SystemExit("include 'plain': every arm is judged against it")
    arms = ["plain"] + [a for a in arms if a != "plain"]

    # ---- data
    seed_all(args.seed)
    train_items, in_tests = [], {}
    for g in train_gens:
        tr, te = list_split(gen_path(g), args.n_test, args.seed)
        shared = {group_key(p) for p, _ in tr} & {group_key(p) for p, _ in te}
        print(f"{Path(g).name}: train {len(tr)}  test {len(te)}  groups shared by train and test: {len(shared)}", flush=True)
        print("  " + describe_grouping(gen_path(g)), flush=True)
        train_items += tr
        in_tests[Path(g).name] = te
    if held_out:
        random.Random(args.seed).shuffle(train_items)
    out_tests = {}
    for g in test_gens:
        items = list_all(gen_path(g))
        if not items:
            print(f"  skip {g}: no images found in {gen_path(g)}", flush=True)
            continue
        out_tests[Path(g).name] = items
        print(f"{Path(g).name}: {len(items)} images, never used in training", flush=True)
        print("  " + describe_grouping(gen_path(g)), flush=True)
    if held_out and not out_tests:
        raise SystemExit("no held-out generator found; check --root and --test-gens")
    if "BEGAN" in out_tests:
        print("  note: BEGAN's real images already carry a JPEG 8x8 grid; read BEGAN rows with care.", flush=True)
    print(f"mode {'held-out' if held_out else 'single'} | training images {len(train_items)} | crop {args.crop} | seed {args.seed}\n",
          flush=True)

    # ---- arms
    store, rows, verdicts = {}, [], {}
    for arm in arms:
        seed_all(args.seed)
        net = make_net(arm, args, dev)
        print(f"training: {arm}", flush=True)
        train_arm(net, arm, train_items, dev, args)
        torch.save(net.state_dict(), str(Path(args.out).with_suffix("")) + f"_{arm}.pth")
        for g, items in in_tests.items():
            score(arm, net, items, dev, args, store, rows, g, CONDITIONS, "seen in training")
        for g, items in out_tests.items():
            score(arm, net, items, dev, args, store, rows, g, CONDITIONS if held_out else SHORT_CONDS, "never seen")
        if arm == "plain":
            first = next(iter(in_tests))
            colour_probe(net, in_tests[first], dev, args)
        else:
            # single mode: the verdict is on the training generator's held-out faces (as in t1);
            # held-out mode: the verdict is on each generator never seen in training.
            for g, items in in_tests.items():
                ok = compare(arm, g, items, store, args, CONDITIONS, judge=not held_out)
                if ok is not None:
                    verdicts[(arm, g)] = ok
            for g, items in out_tests.items():
                ok = compare(arm, g, items, store, args, CONDITIONS if held_out else SHORT_CONDS, judge=held_out)
                if ok is not None:
                    verdicts[(arm, g)] = ok
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- summary
    print("\n### Summary")
    print(f"  mode {'held-out' if held_out else 'single'}, trained on {args.train_label}, seed {args.seed}")
    if not held_out and "AttGAN" in in_tests:
        y, p = store[("AttGAN", "plain", "Q95_420")]
        print(f"  anchor: plain AUC on AttGAN at Q95_420 = {roc_auc_score(y, p):.3f} (t1 runs: 0.829, 0.829, 0.829)")
    print(f"  {'generator':10} {'role':12} " + " ".join(f"{a:>8}" for a in arms) + "   (AUC at Q95_420)")
    for g in list(in_tests) + list(out_tests):
        role = "seen" if g in in_tests else "never seen"
        vals = []
        for a in arms:
            y, p = store[(g, a, "Q95_420")]
            vals.append(f"{roc_auc_score(y, p):8.3f}")
        print(f"  {g:10} {role:12} " + " ".join(vals))
    for (arm, g), ok in verdicts.items():
        print(f"  VERDICT {arm:9} on {g:8} {'PASS' if ok else 'FAIL'}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
