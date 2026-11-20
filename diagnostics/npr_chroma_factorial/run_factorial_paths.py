#!/usr/bin/env python3
"""Four-edge path integrals on the saved libjpeg-turbo decodes. Frozen NPR."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "experiments" / "npr_chroma_sensitivity"))
sys.path.insert(0, str(ROOT / "scripts"))

from chroma_ops import gl_integrate, gl_interval  # noqa: E402
from jpeg_codec import decode  # noqa: E402
from npr_eval_lib import NORM, device, load_npr  # noqa: E402
from torchvision import transforms

EDGES = (("444", "422"), ("444", "440"), ("422", "420"), ("440", "420"))
T5, W5 = gl_interval(5)
T9, W9 = gl_interval(9)
T17, W17 = gl_interval(17)
TIMES = np.unique(np.concatenate([np.array([0.0, 0.25, 0.5, 0.75, 1.0]), T5, T9, T17]))


def u_of(model, x):
    return (x - model.interpolate(x, 0.5)) * (2.0 / 3.0)


def forward_from_u(model, u):
    x = model.conv1(u)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.avgpool(x)
    return model.fc1(torch.flatten(x, 1)).reshape(-1)


def grad_at(model, u):
    leaf = u.detach().requires_grad_(True)
    z = forward_from_u(model, leaf)
    return torch.autograd.grad(z.sum(), leaf)[0].detach()


def project(g, delta):
    return (g * delta).sum(dim=(1, 2, 3))


def integrate(samples, times, nodes, weights):
    idx = [int(np.where(np.isclose(times, node))[0][0]) for node in nodes]
    taken = samples[idx]
    w = torch.tensor(weights, dtype=taken.dtype, device=taken.device).view(-1, 1)
    return (taken * w).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "NPR-DeepfakeDetection" / "NPR.pth")
    ap.add_argument("--manifest", type=Path, default=EXP / "results" / "jpeg_factorial_manifest.csv")
    ap.add_argument("--predictions", type=Path, default=EXP / "results" / "factorial_predictions.csv")
    ap.add_argument("--out", type=Path, default=EXP / "results" / "factorial_gradient_paths.csv")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = device()
    model = load_npr(str(args.checkpoint), dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    preds = list(csv.DictReader(args.predictions.open()))
    man = list(csv.DictReader(args.manifest.open()))
    loc = {(r["path"], r["arm"]): r["jpeg_path"] for r in man}
    done = set()
    if args.out.exists() and args.out.stat().st_size:
        done = {r["path"] for r in csv.DictReader(args.out.open())}
    fields = [
        "generator", "split", "path", "label", "face_id", "edge",
        "dz", "gl5", "gl9", "gl17", "abs_err_gl17",
        "gdir_0", "gdir_25", "gdir_50", "gdir_75", "gdir_1",
        "cos_0", "cos_1", "gnorm_0", "gnorm_1", "du_norm", "rel",
        "sign_changes", "sign_changes_quarter",
    ]
    new = not args.out.exists() or args.out.stat().st_size == 0
    out = args.out.open("a", newline="")
    w = csv.DictWriter(out, fieldnames=fields)
    if new:
        w.writeheader()
        out.flush()
    jobs = [r for r in preds if r["path"] not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"path nodes {len(TIMES)} todo {len(jobs)}", flush=True)

    def one(batch):
        images = {}
        for arm in ("444", "422", "440", "420"):
            raw = [decode(Path(loc[(rec["path"], arm)]).read_bytes()) for rec in batch]
            x = torch.stack([NORM(transforms.functional.to_tensor(im)) for im in raw]).to(dev)
            with torch.no_grad():
                images[arm] = u_of(model, x)
        for a, b in EDGES:
            delta = images[b] - images[a]
            series = []
            g_ends = {}
            for t in TIMES:
                tt = float(t)
                if tt == 0.0:
                    uu = images[a]
                elif tt == 1.0:
                    uu = images[b]
                else:
                    uu = images[a] + tt * delta
                g = grad_at(model, uu)
                if tt in (0.0, 1.0):
                    g_ends[tt] = g
                series.append(project(g, delta))
            stacked = torch.stack(series, 0)
            gl5 = integrate(stacked, TIMES, T5, W5)
            gl9 = integrate(stacked, TIMES, T9, W9)
            gl17 = integrate(stacked, TIMES, T17, W17)
            du = delta.flatten(1).norm(dim=1)
            ua = images[a].flatten(1).norm(dim=1)
            gn0 = g_ends[0.0].flatten(1).norm(dim=1)
            gn1 = g_ends[1.0].flatten(1).norm(dim=1)

            def at(t):
                return stacked[int(np.where(np.isclose(TIMES, t))[0][0])]

            p0, p1 = at(0.0), at(1.0)
            quarters = torch.stack([at(0.0), at(0.25), at(0.5), at(0.75), at(1.0)], 0)
            cos0 = p0 / (gn0 * du + 1e-12)
            cos1 = p1 / (gn1 * du + 1e-12)
            signs = torch.sign(stacked)
            changes = (signs[1:] * signs[:-1] < 0).sum(0)
            qsign = torch.sign(quarters)
            qchanges = (qsign[1:] * qsign[:-1] < 0).sum(0)
            for i, rec in enumerate(batch):
                dz = float(rec[f"z_{b}"]) - float(rec[f"z_{a}"])
                g17 = float(gl17[i].cpu())
                w.writerow({
                    "generator": rec["generator"], "split": rec["split"], "path": rec["path"],
                    "label": rec["label"], "face_id": rec["face_id"], "edge": f"{a}_to_{b}",
                    "dz": dz, "gl5": float(gl5[i].cpu()), "gl9": float(gl9[i].cpu()),
                    "gl17": g17, "abs_err_gl17": abs(dz - g17),
                    "gdir_0": float(p0[i].cpu()),
                    "gdir_25": float(at(0.25)[i].cpu()),
                    "gdir_50": float(at(0.5)[i].cpu()),
                    "gdir_75": float(at(0.75)[i].cpu()),
                    "gdir_1": float(p1[i].cpu()),
                    "cos_0": float(cos0[i].cpu()), "cos_1": float(cos1[i].cpu()),
                    "gnorm_0": float(gn0[i].cpu()), "gnorm_1": float(gn1[i].cpu()),
                    "du_norm": float(du[i].cpu()), "rel": float((du[i] / (ua[i] + 1e-12)).cpu()),
                    "sign_changes": int(changes[i].cpu()),
                    "sign_changes_quarter": int(qchanges[i].cpu()),
                })
        out.flush()
        print(f"paths {batch[-1]['split']} {Path(batch[-1]['path']).name}", flush=True)

    batch = []
    for rec in jobs:
        if batch and (batch[0]["width"], batch[0]["height"]) != (rec["width"], rec["height"]):
            one(batch)
            batch = []
        batch.append(rec)
        if len(batch) >= args.batch_size:
            one(batch)
            batch = []
    if batch:
        one(batch)
    print("done", flush=True)


if __name__ == "__main__":
    main()
