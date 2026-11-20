"""npr_t4_colab.py  --  version t4: JA-NPR plus two control arms

t4 = t3 unchanged, plus two controls (new arms, nothing else changed):
  janpr_constq  JA-NPR, but every FiLM layer is fed the SAME constant vector instead of the file header.
                Same parameters as JA-NPR, no header information. Tests "is it just more parameters?"
  qtnorm_rgb    QT-Norm on the ORIGINAL RGB NPR residual (no chroma-aligned residual).
                Tests whether the chroma-aligned residual is needed inside JA-NPR.

(t3 description follows)


The final method has three parts. Each part is a change to the MODEL, not to the data, and each can be
switched off for an ablation:

  1. Chroma-aligned NPR residual (CA-NPR)
     NPR's own 2x2 residual, computed where a JPEG file actually stores information: brightness (Y) at
     full size, colour (Cb, Cr) at half size, because 4:2:0 JPEG keeps colour at half size only.
  2. Quantization-table-conditioned normalization (QT-Norm)
     Every batch-norm layer gets (a) separate running statistics for files with and without a JPEG
     table, and (b) a small FiLM layer that reads the file's own quantization table (how strongly it
     was compressed, and whether colour was subsampled) and rescales the features accordingly.
     Idea from domain-specific BN (Chang et al., CVPR 2019) and quality-factor conditioning in JPEG
     restoration (FBCNN, Jiang et al., ICCV 2021). The table is read from the file header, never guessed.
  3. Compression-adversarial head (CAdv)
     A one-layer head tries to tell clean from JPEG from the pooled features; a gradient-reversal layer
     (Ganin & Lempitsky, ICML 2015) pushes the backbone to make that impossible, so the features NPR
     passes to the classifier carry fake/real evidence but not "how compressed is this file".
     Used for compressed deepfakes in ODDN (Tao et al., AAAI 2025) and PLADA (Li et al., 2025).

Parts 2 and 3 work together: QT-Norm gives the network the information needed to undo compression
effects, CAdv gives it the reason to do so.

Arms (all start from the released NPR weights; every trained arm gets the SAME training protocol):
  released     the released NPR, no training (the base paper as published)
  plain        NPR fine-tuned with clean + JPEG views (the strongest honest baseline)
  ycdsbn       parts 1 + 2a  (best arm in t2)
  janpr_noadv  parts 1 + 2   (ablation: no adversarial head)
  janpr        parts 1 + 2 + 3  (the full method)

Modes: single (--gen, optional --extra-gens) and held-out (--train-gens, --test-gens), as in t2.
"""

import argparse
import csv
import io
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, JpegImagePlugin
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

VERSION = "npr_t4_colab t4"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
YC_SCALE = 0.226
IMG_EXT = {".png", ".jpg", ".jpeg"}
QDIM = 4


# ----------------------------------------------------------------------------- data
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def source_id(path):
    return Path(path).stem.split("_")[0]


@lru_cache(maxsize=None)
def folder_uses_ids(folder):
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


def qvec_of(im):
    """What the file header says: [has JPEG table, luma table strength, chroma table strength, 4:2:0].
    Read from the opened file itself, the way a deployed detector would read it."""
    q = getattr(im, "quantization", None)
    if not q:
        return [0.0, 0.0, 0.0, 0.0]
    luma = float(np.mean(q[0]))
    chroma = float(np.mean(q[1])) if 1 in q else luma
    try:
        s = JpegImagePlugin.get_sampling(im)
    except Exception:
        s = -1
    return [1.0, math.log1p(luma) / 5.0, math.log1p(chroma) / 5.0, 1.0 if s == 2 else 0.0]


def open_with_q(path):
    im = Image.open(path)
    q = qvec_of(im)
    return im.convert("RGB"), q


def jpeg_image(im, quality, subsampling):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    out = Image.open(buf)
    q = qvec_of(out)
    return out.convert("RGB"), q


def to255(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1)


class TrainSet(Dataset):
    def __init__(self, items, args):
        self.items, self.a = items, args

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a = self.a
        path, label = self.items[idx]
        im, qc = open_with_q(path)
        rng = random if a.fresh_crops else random.Random(a.seed + idx * 17 + int(torch.initial_seed() % 100000))
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = im.size
        if w < a.crop or h < a.crop:
            raise SystemExit(f"{path} is {w}x{h}, smaller than --crop {a.crop}. Use a smaller --crop.")
        x, y = rng.randrange(0, w - a.crop + 1), rng.randrange(0, h - a.crop + 1)
        im = im.crop((x, y, x + a.crop, y + a.crop))
        jim, qj = jpeg_image(im, random.randint(a.q_lo, a.q_hi), random.choice(a.subs))
        return to255(im), torch.tensor(qc), to255(jim), torch.tensor(qj), label


class EvalSet(Dataset):
    def __init__(self, items, quality, subsampling):
        self.items, self.quality, self.subsampling = items, quality, subsampling

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        im, q = open_with_q(path)
        if self.quality is not None:
            im, q = jpeg_image(im, self.quality, self.subsampling)
        return to255(im), torch.tensor(q), label


# ----------------------------------------------------------------------------- colour helpers
def rgb2ycc(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([0.299 * r + 0.587 * g + 0.114 * b,
                      128 - 0.168736 * r - 0.331264 * g + 0.5 * b,
                      128 + 0.5 * r - 0.418688 * g - 0.081312 * b], 1)


YCC2RGB = torch.tensor([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]])


def nn_residual(p):
    """NPR's own operation: each pixel minus the top-left pixel of its 2x2 cell."""
    return p - p[:, :, 0::2, 0::2].repeat_interleave(2, -1).repeat_interleave(2, -2)


# ----------------------------------------------------------------------------- model
class QTNorm(nn.Module):
    """Batch norm that knows about the file's JPEG table.
    dual: separate running statistics for 'no table' (0) and 'has table' (1).
    film: per-channel scale/shift computed from the table summary; starts at zero = identity."""

    def __init__(self, bn, dual, film):
        super().__init__()
        self.weight, self.bias = bn.weight, bn.bias
        self.eps, self.momentum = bn.eps, bn.momentum
        self.dual = dual
        for d in ((0, 1) if dual else (0,)):
            self.register_buffer(f"rm{d}", bn.running_mean.clone())
            self.register_buffer(f"rv{d}", bn.running_var.clone())
        self.film = None
        if film:
            self.film = nn.Linear(QDIM, 2 * bn.num_features)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.domain, self.q = 0, None

    def forward(self, x):
        d = self.domain if self.dual else 0
        y = F.batch_norm(x, getattr(self, f"rm{d}"), getattr(self, f"rv{d}"), self.weight, self.bias,
                         self.training, self.momentum, self.eps)
        if self.film is not None:
            g, b = self.film(self.q).chunk(2, 1)
            y = y * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return y


def swap_bn(module, dual, film):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, QTNorm(child, dual, film))
        else:
            swap_bn(child, dual, film)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


CONST_Q = torch.tensor([[1.0, 0.65, 0.72, 0.5]])   # fixed vector for the janpr_constq control


class Net(nn.Module):
    def __init__(self, base, residual="rgb", dual=False, film=False, adv=False, constq=False):
        super().__init__()
        self.base, self.mode, self.constq = base, residual, constq
        if dual or film:
            swap_bn(self.base, dual, film)
        if residual == "yc":
            # conv1 re-expressed so that with full-size colour the network equals the released NPR exactly
            w = self.base.conv1.weight.data
            m = (YCC2RGB / STD.view(3, 1)).to(w)
            self.base.conv1.weight.data = YC_SCALE * torch.einsum("ocij,ck->okij", w, m)
        self.adv = nn.Linear(self.base.fc1.in_features, 1) if adv else None

    def new_parameters(self):
        out = [p for m in self.modules() if isinstance(m, QTNorm) and m.film is not None for p in m.film.parameters()]
        if self.adv is not None:
            out += list(self.adv.parameters())
        return out

    def residual(self, x255, chroma_half=True):
        if self.mode == "rgb":
            x = (x255 / 255.0 - MEAN.to(x255.device)) / STD.to(x255.device)
            return x - self.base.interpolate(x, 0.5)
        ycc = rgb2ycc(x255)
        ry = nn_residual(ycc[:, 0:1])
        if chroma_half:
            c = F.avg_pool2d(ycc[:, 1:3], 2)
            rc = nn_residual(c).repeat_interleave(2, -1).repeat_interleave(2, -2)
        else:
            rc = nn_residual(ycc[:, 1:3])
        return torch.cat([ry, rc], 1) / (255.0 * YC_SCALE)

    def forward(self, x255, q, chroma_half=True):
        domain = int(q[:, 0].mean().item() > 0.5)
        qf = CONST_Q.to(q).expand_as(q) if self.constq else q   # control: FiLM never sees the header
        for m in self.modules():
            if isinstance(m, QTNorm):
                m.domain, m.q = domain, qf
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


ARMS = {
    "released": dict(residual="rgb", dual=False, film=False, adv=False, train=False),
    "plain": dict(residual="rgb", dual=False, film=False, adv=False, train=True),
    "ycdsbn": dict(residual="yc", dual=True, film=False, adv=False, train=True),
    "janpr_noadv": dict(residual="yc", dual=True, film=True, adv=False, train=True),
    "janpr": dict(residual="yc", dual=True, film=True, adv=True, train=True),
    "janpr_constq": dict(residual="yc", dual=True, film=True, adv=False, train=True, constq=True),
    "qtnorm_rgb": dict(residual="rgb", dual=True, film=True, adv=False, train=True),
}


def make_net(arm, args, dev):
    c = ARMS[arm]
    return Net(load_base(args.npr, args.ckpt), c["residual"], c["dual"], c["film"], c["adv"], c.get("constq", False)).to(dev)


def count_params(net):
    """Parameters actually used at test time (the adversarial head is training-only)."""
    used = [net.base.conv1, net.base.bn1, net.base.layer1, net.base.layer2, net.base.fc1]
    return sum(p.numel() for m in used for p in m.parameters())


def train_arm(net, arm, train_items, dev, args):
    cfg = ARMS[arm]
    new = net.new_parameters()
    new_ids = {id(p) for p in new}
    old = [p for p in net.parameters() if id(p) not in new_ids]
    groups = [{"params": old, "lr": args.lr}] + ([{"params": new, "lr": args.lr_new}] if new else [])
    opt = torch.optim.Adam(groups)
    loader = DataLoader(TrainSet(train_items, args), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    bce = nn.BCEWithLogitsLoss()
    steps, step = args.epochs * len(loader), 0
    net.train()
    for epoch in range(args.epochs):
        total, n, adv_ok, adv_n, t0 = 0.0, 0, 0, 0, time.time()
        for clean, qc, jpg, qj, y in loader:
            clean, qc, jpg, qj, y = clean.to(dev), qc.to(dev), jpg.to(dev), qj.to(dev), y.float().to(dev)
            lc, fc = net(clean, qc)
            lj, fj = net(jpg, qj)
            loss = bce(lc, y) + bce(lj, y)
            if cfg["adv"]:
                p = step / max(steps - 1, 1)
                lam = args.lam_adv * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)      # DANN schedule
                f = torch.cat([fc, fj])
                d = torch.cat([torch.zeros(len(fc)), torch.ones(len(fj))]).to(dev)
                logit_d = net.adv(GradReverse.apply(f, lam)).flatten()
                loss = loss + bce(logit_d, d)
                adv_ok += int(((logit_d > 0).float() == d).sum())
                adv_n += len(d)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            total += float(loss.detach()) * y.shape[0]
            n += y.shape[0]
        extra = f"  adversary accuracy {adv_ok / adv_n:.3f} (0.5 = features hide compression)" if adv_n else ""
        print(f"  epoch {epoch + 1}/{args.epochs}  loss {total / max(n, 1):.4f}  {time.time() - t0:.0f}s{extra}", flush=True)


# ----------------------------------------------------------------------------- evaluation
CONDITIONS = [("orig", None, 2), ("Q95_420", 95, 2), ("Q90_420", 90, 2), ("Q70_420", 70, 2),
              ("Q50_420", 50, 2), ("Q30_420", 30, 2), ("Q95_444", 95, 0), ("Q75_444", 75, 0),
              ("Q85_422", 85, 1)]
SHORT_CONDS = [c for c in CONDITIONS if c[0] in ("orig", "Q95_420", "Q90_420", "Q70_420", "Q50_420")]


@torch.no_grad()
def predict(net, items, dev, quality, sub, args):
    net.eval()
    loader = DataLoader(EvalSet(items, quality, sub), batch_size=args.eval_batch, shuffle=False, num_workers=args.workers)
    ys, ps = [], []
    for x, q, y in loader:
        logit, _ = net(x.to(dev), q.to(dev))
        ps.append(logit.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def clusters_of(items):
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
        lo, hi = single_ci(items, y, p, args.n_boot_info)
        tpr, tnr = float((p[y == 1] > 0).mean()), float((p[y == 0] <= 0).mean())
        rows.append({"seed": args.seed, "train": args.train_label, "protocol": f"Q{args.q_lo}-{args.q_hi} subs {args.train_subs}",
                     "generator": gen, "role": role, "arm": arm, "condition": cond, "auc": auc, "ci_lo": lo,
                     "ci_hi": hi, "bal_acc_0.5": (tpr + tnr) / 2, "tpr_0.5": tpr, "tnr_0.5": tnr})
        print(f"  {cond:8} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]  at 0.5: bal.acc {(tpr + tnr) / 2:.3f} "
              f"(fakes caught {tpr:.3f}, reals passed {tnr:.3f})", flush=True)


KEY = ["Q95_420", "Q90_420"]
GUARD = ["orig", "Q70_420", "Q50_420", "Q30_420", "Q95_444", "Q75_444"]
RULE_TEXT = ("PASS if, against plain: AUC clearly better on Q95_420 and Q90_420; balanced accuracy at 0.5 clearly "
             "better on Q95_420 and Q90_420; and AUC not clearly worse on any of " + ", ".join(GUARD) + ".")
JUDGED = ("ycdsbn", "janpr_noadv", "janpr", "janpr_constq", "qtnorm_rgb")


def compare(a, b, gen, items, store, args, conds, judge):
    """Paired comparison: b minus a. Returns PASS/FAIL if judged, else None."""
    head = f"\n### {b} vs {a} on {gen}"
    print(head + (f"\nRule (fixed before the run): {RULE_TEXT}" if judge else "  (information only, no verdict)"), flush=True)
    print(f"  {'condition':9} {'AUC ' + a:>16} {'AUC ' + b:>16}  {'dAUC [95% CI]':>24}   "
          f"{'bal.acc ' + a:>18} {'bal.acc ' + b:>18}  {'d bal.acc [95% CI]':>24}", flush=True)
    nb = args.n_boot if judge else args.n_boot_info
    res = {}
    for cond, _, _ in conds:
        y, pa = store[(gen, a, cond)]
        _, pb = store[(gen, b, cond)]
        a_lo, a_hi = paired_ci(items, y, pa, pb, roc_auc_score, nb)
        b_lo, b_hi = paired_ci(items, y, pa, pb, bal_acc, nb)
        res[cond] = (a_lo, a_hi, b_lo, b_hi)
        print(f"  {cond:9} {roc_auc_score(y, pa):16.3f} {roc_auc_score(y, pb):16.3f}  "
              f"{roc_auc_score(y, pb) - roc_auc_score(y, pa):+.3f} [{a_lo:+.3f}, {a_hi:+.3f}]   "
              f"{bal_acc(y, pa):18.3f} {bal_acc(y, pb):18.3f}  {bal_acc(y, pb) - bal_acc(y, pa):+.3f} "
              f"[{b_lo:+.3f}, {b_hi:+.3f}]", flush=True)
    if not judge:
        return None
    ok = (all(res[c][0] > 0 and res[c][2] > 0 for c in KEY)
          and all(not res[c][1] < 0 for c in GUARD if c in res))
    print(f"VERDICT {b} on {gen}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ----------------------------------------------------------------------------- checks
def self_check(args, dev):
    print(VERSION, "| device", dev, flush=True)
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64, device=dev) * 255
    q0 = torch.zeros(2, QDIM, device=dev)
    q1 = torch.tensor([[1.0, 0.3, 0.4, 1.0]] * 2, device=dev)
    with torch.no_grad():
        a, _ = make_net("released", args, dev).eval()(x, q0)
        for arm in ARMS:
            if arm == "released":
                continue
            net = make_net(arm, args, dev).eval()
            b, _ = net(x, q1, chroma_half=False)
            err = float((a - b).abs().max())
            if err > 1e-3 * max(1.0, float(a.abs().max())):
                raise SystemExit(f"self-check failed for {arm} (max diff {err:.2e}). Stop and report this.")
    print("self-check passed: every arm starts exactly at the released NPR (before training)", flush=True)
    im = Image.new("RGB", (32, 32), (120, 60, 200))
    _, qa = jpeg_image(im, 95, 2)
    _, qb = jpeg_image(im, 30, 0)
    print(f"self-check: header reading  Q95 4:2:0 -> {[round(v, 2) for v in qa]}   Q30 4:4:4 -> {[round(v, 2) for v in qb]}",
          flush=True)
    if not (qa[0] == 1 and qa[3] == 1 and qb[3] == 0 and qb[1] > qa[1]):
        raise SystemExit("self-check failed: JPEG header reading is wrong. Stop and report this.")
    print(flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="released,plain,ycdsbn,janpr_noadv,janpr")
    ap.add_argument("--root", default="/content/data/GANGen-Detection")
    ap.add_argument("--gen", default="AttGAN")
    ap.add_argument("--extra-gens", default="")
    ap.add_argument("--train-gens", default="")
    ap.add_argument("--test-gens", default="")
    ap.add_argument("--npr", default="/content/NPR")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-new", type=float, default=1e-3, help="learning rate of the new FiLM and adversary layers")
    ap.add_argument("--lam-adv", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--q-lo", type=int, default=30)
    ap.add_argument("--q-hi", type=int, default=95)
    ap.add_argument("--train-subs", default="2,0", help="JPEG colour subsampling used in training: 2=4:2:0, 0=4:4:4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-info", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh-crops", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.ckpt = args.ckpt or str(Path(args.npr) / "NPR.pth")
    args.subs = [int(s) for s in args.train_subs.split(",")]

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
    args.out = args.out or f"/content/npr_t4_{tag}_seed{args.seed}.csv"

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
    order = ["released", "plain"] + [a for a in ARMS if a not in ("released", "plain")]
    arms = [a for a in order if a in arms]

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
    print(f"mode {'held-out' if held_out else 'single'} | training images {len(train_items)} | crop {args.crop} | "
          f"JPEG quality {args.q_lo}-{args.q_hi}, subsampling {args.train_subs} | seed {args.seed}\n", flush=True)

    store, rows, verdicts, params = {}, [], {}, {}
    out_conds = CONDITIONS if held_out else SHORT_CONDS
    for arm in arms:
        net = make_net(arm, args, dev)
        seed_all(args.seed)   # after building the net, so every arm sees the same training order and crops
        params[arm] = count_params(net)
        print(f"{'training' if ARMS[arm]['train'] else 'scoring (no training)'}: {arm}  "
              f"[{params[arm] / 1e6:.3f}M parameters used at test time]", flush=True)
        if ARMS[arm]["train"]:
            train_arm(net, arm, train_items, dev, args)
            torch.save(net.state_dict(), str(Path(args.out).with_suffix("")) + f"_{arm}.pth")
        for g, items in in_tests.items():
            score(arm, net, items, dev, args, store, rows, g, CONDITIONS, "seen in training")
        for g, items in out_tests.items():
            score(arm, net, items, dev, args, store, rows, g, out_conds, "never seen")
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # comparisons: judged = new arms vs plain; information = baseline vs released, and the ablations
    tests = list(in_tests.items()) + list(out_tests.items())
    for g, items in tests:
        conds = CONDITIONS if g in in_tests or held_out else out_conds
        judged_here = (g in out_tests) if held_out else (g in in_tests)
        if "released" in arms:
            compare("released", "plain", g, items, store, args, conds, judge=False)
        for arm in arms:
            if arm in JUDGED:
                ok = compare("plain", arm, g, items, store, args, conds, judge=judged_here)
                if ok is not None:
                    verdicts[(arm, g)] = ok
        if "janpr" in arms and "janpr_noadv" in arms:
            compare("janpr_noadv", "janpr", g, items, store, args, conds, judge=False)
        if "janpr_noadv" in arms and "ycdsbn" in arms:
            compare("ycdsbn", "janpr_noadv", g, items, store, args, conds, judge=False)
        for ctrl in ("janpr_constq", "qtnorm_rgb"):
            if ctrl in arms and "janpr_noadv" in arms:
                compare(ctrl, "janpr_noadv", g, items, store, args, conds, judge=False)

    print("\n### Summary")
    print(f"  mode {'held-out' if held_out else 'single'}, trained on {args.train_label}, JPEG Q{args.q_lo}-{args.q_hi} "
          f"subsampling {args.train_subs}, seed {args.seed}")
    print("  parameters used at test time: " + ", ".join(f"{a} {params[a] / 1e6:.3f}M" for a in arms))
    show = ["orig", "Q95_420", "Q90_420", "Q70_420", "Q50_420"]
    for g in list(in_tests) + list(out_tests):
        print(f"\n  {g} ({'seen' if g in in_tests else 'never seen'}): AUC")
        print(f"    {'arm':12} " + " ".join(f"{c:>8}" for c in show))
        for a in arms:
            print(f"    {a:12} " + " ".join(f"{roc_auc_score(*store[(g, a, c)]):8.3f}" for c in show))
    print()
    for (arm, g), ok in verdicts.items():
        print(f"  VERDICT {arm:12} on {g:8} {'PASS' if ok else 'FAIL'}")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
