#!/usr/bin/env python3
"""Activation-space factorial effects for the frozen NPR checkpoint.

Spatial maps are reduced in the forward pass. The 512-d pooled vector, which
is the input of the linear classifier, is saved in full. Logits are checked
against the existing factorial CSV.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
FACT = ROOT / "experiments" / "npr_chroma_factorial"
sys.path.insert(0, str(FACT))
sys.path.insert(0, str(ROOT / "scripts"))

from jpeg_codec import decode, version  # noqa: E402
from npr_eval_lib import NORM, device, load_npr  # noqa: E402

ARMS = ("444", "422", "440", "420")
STAGES = ("npr", "npr_scaled", "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "pre_fc")
EPS = 1e-8
CKPT_SHA = "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward_stages(model, x):
    """Same module order as ResNet.forward. bn1 is cloned before the inplace ReLU."""
    npr = x - model.interpolate(x, 0.5)
    scaled = npr * (2.0 / 3.0)
    h = model.conv1(scaled)
    conv1 = h
    h = model.bn1(h)
    bn1 = h.clone()
    h = model.relu(h)
    relu = h
    h = model.maxpool(h)
    pool = h
    h = model.layer1(h)
    layer1 = h
    h = model.layer2(h)
    layer2 = h
    h = model.avgpool(h)
    pre = torch.flatten(h, 1)
    logit = model.fc1(pre).reshape(-1)
    return {
        "npr": npr,
        "npr_scaled": scaled,
        "conv1": conv1,
        "bn1": bn1,
        "relu": relu,
        "maxpool": pool,
        "layer1": layer1,
        "layer2": layer2,
        "pre_fc": pre,
    }, logit


def _flat(t):
    return t.reshape(t.shape[0], -1)


def _l2(t):
    return _flat(t).norm(dim=1)


def _cos(a, b):
    af, bf = _flat(a), _flat(b)
    return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1) + EPS)


def _mean_abs(t):
    return _flat(t).abs().mean(dim=1)


def _top_energy(diff, k=10):
    if diff.ndim == 2:
        energy = diff.pow(2)
    else:
        energy = diff.pow(2).sum(dim=tuple(range(2, diff.ndim)))
    total = energy.sum(1) + EPS
    kk = min(k, energy.shape[1])
    top = energy.topk(kk, dim=1).values.sum(1)
    # Participation ratio of channel energies. 1 = one channel, C = equal channels.
    pr = total.pow(2) / (energy.pow(2).sum(1) + EPS)
    return top / total, pr


def stage_rows(batch_meta, acts, logits, refs):
    rows = []
    pre = {}
    a00, a10, a01, a11 = (acts[a] for a in ARMS)
    for name in STAGES:
        t00, t10, t01, t11 = a00[name], a10[name], a01[name], a11[name]
        h0, h1 = t10 - t00, t11 - t01
        v0, v1 = t01 - t00, t11 - t10
        inter = t11 - t10 - t01 + t00
        phi_h = 0.5 * (h0 + h1)
        phi_v = 0.5 * (v0 + v1)
        ident = phi_h + phi_v - (t11 - t00)
        ident_i = (h1 - h0) - inter
        top, pr = _top_energy(inter)
        packs = {
            "l2_A00": _l2(t00),
            "l2_H0": _l2(h0),
            "l2_H1": _l2(h1),
            "l2_V0": _l2(v0),
            "l2_V1": _l2(v1),
            "l2_I": _l2(inter),
            "l2_Phi_H": _l2(phi_h),
            "l2_Phi_V": _l2(phi_v),
            "l2_total": _l2(t11 - t00),
            "l2_identity": _l2(ident),
            "l2_interaction_identity": _l2(ident_i),
            "mean_abs_I": _mean_abs(inter),
            "mean_abs_H0": _mean_abs(h0),
            "mean_abs_V0": _mean_abs(v0),
            "cos_H0_V0": _cos(h0, v0),
            "cos_I_H0": _cos(inter, h0),
            "cos_I_V0": _cos(inter, v0),
            "cos_PhiH_PhiV": _cos(phi_h, phi_v),
            "cos_00_10": _cos(t00, t10),
            "cos_00_01": _cos(t00, t01),
            "cos_00_11": _cos(t00, t11),
            "cos_10_01": _cos(t10, t01),
            "cos_10_11": _cos(t10, t11),
            "cos_01_11": _cos(t01, t11),
            "top10_energy_frac": top,
            "participation_ratio": pr,
        }
        cpu = {k: v.detach().float().cpu().numpy() for k, v in packs.items()}
        numel = int(t00[0].numel())
        for i, meta in enumerate(batch_meta):
            rec = {
                "generator": meta["generator"],
                "split": meta["split"],
                "path": meta["path"],
                "label": meta["label"],
                "face_id": meta["face_id"],
                "stage": name,
                "numel": numel,
                "logit_444": float(logits["444"][i].cpu()),
                "logit_422": float(logits["422"][i].cpu()),
                "logit_440": float(logits["440"][i].cpu()),
                "logit_420": float(logits["420"][i].cpu()),
                "ref_444": refs[i]["z_444"],
                "ref_422": refs[i]["z_422"],
                "ref_440": refs[i]["z_440"],
                "ref_420": refs[i]["z_420"],
            }
            for k, arr in cpu.items():
                rec[k] = float(arr[i])
            rec["rel_I"] = rec["l2_I"] / (rec["l2_H0"] + rec["l2_V0"] + EPS)
            rec["rel_I_vs_A00"] = rec["l2_I"] / (rec["l2_A00"] + EPS)
            rec["rel_H0_vs_A00"] = rec["l2_H0"] / (rec["l2_A00"] + EPS)
            rec["rel_V0_vs_A00"] = rec["l2_V0"] / (rec["l2_A00"] + EPS)
            rows.append(rec)
        if name == "pre_fc":
            pre = torch.stack([a00["pre_fc"], a10["pre_fc"], a01["pre_fc"], a11["pre_fc"]], 1)
            pre = pre.detach().float().cpu().numpy()
    return rows, pre


def load_jobs(pred_path, manifest_path):
    preds = list(csv.DictReader(pred_path.open()))
    loc = {}
    for row in csv.DictReader(manifest_path.open()):
        loc[(row["path"], row["arm"])] = row["jpeg_path"]
    jobs = []
    for row in preds:
        paths = {arm: loc[(row["path"], arm)] for arm in ARMS}
        for arm, jpeg in paths.items():
            if not Path(jpeg).is_file():
                raise SystemExit(f"missing JPEG {jpeg}")
        jobs.append({
            "generator": row["generator"],
            "split": row["split"],
            "path": row["path"],
            "label": int(row["label"]),
            "face_id": row["face_id"],
            "width": int(row["width"]),
            "height": int(row["height"]),
            "jpeg": paths,
            "ref": {f"z_{arm}": float(row[f"z_{arm}"]) for arm in ARMS},
        })
    return jobs


def tensor_from_jpeg(path, dev):
    im = decode(Path(path).read_bytes()).convert("RGB")
    x = NORM(transforms.functional.to_tensor(im))
    return x.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "NPR-DeepfakeDetection" / "NPR.pth")
    ap.add_argument("--predictions", type=Path, default=FACT / "results" / "factorial_predictions.csv")
    ap.add_argument("--manifest", type=Path, default=FACT / "results" / "jpeg_factorial_manifest.csv")
    ap.add_argument("--out-dir", type=Path, default=EXP / "results")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    digest = sha256(args.checkpoint)
    if digest != CKPT_SHA:
        raise SystemExit(f"checkpoint hash mismatch {digest}")
    dev = device()
    model = load_npr(str(args.checkpoint), dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != 1437761:
        raise SystemExit(f"parameter count {n_params}")
    weight = model.fc1.weight.detach().float().cpu().numpy().reshape(-1)
    bias = float(model.fc1.bias.detach().float().cpu().numpy().reshape(-1)[0])
    if weight.shape != (512,):
        raise SystemExit(f"fc1 weight shape {weight.shape}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / "pre_fc_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "checkpoint_sha256": digest,
        "n_params": n_params,
        "encoder_decoder": version(),
        "decode": "djpeg -dct int, default smooth upsampling, saved factorial JPEG bytes",
        "normalization": "ImageNet mean/std, full image, no crop",
        "arms": list(ARMS),
        "stages": list(STAGES),
        "eps": EPS,
        "fc1_bias": bias,
        "note": "Spatial tensors are reduced before they are discarded. pre_fc is the flattened adaptive average pool and the input of fc1.",
    }
    (args.out_dir / "extract_config.json").write_text(json.dumps(cfg, indent=2))
    np.save(args.out_dir / "fc1_weight.npy", weight)

    jobs = load_jobs(args.predictions, args.manifest)
    if args.limit:
        jobs = jobs[:args.limit]
    csv_path = args.out_dir / "stage_effects.csv"
    done = set()
    if csv_path.exists() and csv_path.stat().st_size:
        done = {r["path"] for r in csv.DictReader(csv_path.open())}
    todo = [j for j in jobs if j["path"] not in done]
    print(f"checkpoint ok params={n_params} device={dev} todo {len(todo)} already {len(done)}", flush=True)

    fields = None
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    out = csv_path.open("a", newline="")
    writer = None
    checked = False
    max_gap = 0.0

    def flush_batch(batch):
        nonlocal writer, fields, checked, max_gap
        if not batch:
            return
        acts = {arm: None for arm in ARMS}
        logits = {}
        xs = {arm: torch.stack([tensor_from_jpeg(rec["jpeg"][arm], dev) for rec in batch]) for arm in ARMS}
        with torch.no_grad():
            for arm in ARMS:
                stages, logit = forward_stages(model, xs[arm])
                official = model(xs[arm]).reshape(-1)
                err = (logit - official).abs().max().item()
                if err > 1e-4:
                    raise SystemExit(f"instrumented forward differs from model.forward by {err}")
                acts[arm] = stages
                logits[arm] = logit
        rows, pre = stage_rows(batch, acts, logits, [rec["ref"] for rec in batch])
        for rec in rows:
            if rec["stage"] != "pre_fc":
                continue
            for arm, key in (("444", "logit_444"), ("422", "logit_422"), ("440", "logit_440"), ("420", "logit_420")):
                gap = abs(rec[key] - rec[f"ref_{arm}"])
                max_gap = max(max_gap, gap)
            if not checked:
                checked = True
                print(f"first logit gaps 444 {abs(rec['logit_444']-rec['ref_444']):.3e}", flush=True)
            if max(abs(rec[f"logit_{arm}"] - rec[f"ref_{arm}"]) for arm in ARMS) > 1e-3:
                raise SystemExit(f"logit mismatch on {rec['path']}")
        if writer is None:
            fields = list(rows[0].keys())
            writer = csv.DictWriter(out, fieldnames=fields)
            if new_file:
                writer.writeheader()
        by_path = {}
        for rec in rows:
            by_path.setdefault(rec["path"], []).append(rec)
        for i, rec in enumerate(batch):
            for row in by_path[rec["path"]]:
                writer.writerow(row)
            np.savez_compressed(
                shard_dir / f"{hashlib.sha256(rec['path'].encode()).hexdigest()[:16]}.npz",
                path=rec["path"],
                pre_fc=pre[i],
            )
        out.flush()
        print(f"activations {batch[-1]['split']} {Path(batch[-1]['path']).name} max_logit_gap {max_gap:.3e}", flush=True)

    batch = []
    for rec in todo:
        if batch and (batch[0]["width"], batch[0]["height"]) != (rec["width"], rec["height"]):
            flush_batch(batch)
            batch = []
        batch.append(rec)
        if len(batch) >= args.batch_size:
            flush_batch(batch)
            batch = []
    if batch:
        flush_batch(batch)
    out.close()
    print(f"done max_logit_gap {max_gap:.3e}", flush=True)


if __name__ == "__main__":
    main()
