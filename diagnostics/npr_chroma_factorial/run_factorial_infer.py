#!/usr/bin/env python3
"""Four-arm libjpeg-turbo factorial. Frozen NPR. Append-only outputs.

Arms are 4:4:4, 4:2:2, 4:4:0, 4:2:0 at quality 95. 4:4:0 is the vertical-only
control. Pillow is scored only as a bridge for the three arms it can encode.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "experiments" / "npr_chroma_sensitivity"))
sys.path.insert(0, str(ROOT / "scripts"))

from chroma_ops import mix, rgb_to_ycbcr  # noqa: E402
from jpeg_codec import (  # noqa: E402
    EXPECTED_SOF, decode, dqt_hash, encode, parse_markers, qtable_ids, sof_tuple, version,
)
from npr_eval_lib import NORM, celeb_id, device, jpeg_pil, list_images, load_npr  # noqa: E402
from run_jpeg_consistency import list_split  # noqa: E402

ARMS = ("444", "422", "440", "420")
EDGES = (("444", "422"), ("444", "440"), ("440", "420"), ("422", "420"))
PIL_SUB = {"444": 0, "422": 1, "420": 2}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def as_batch(images, dev):
    xs = [NORM(transforms.functional.to_tensor(im)) for im in images]
    return torch.stack(xs, 0).to(dev)


def logits_of(model, images, dev):
    x = as_batch(images, dev)
    with torch.no_grad():
        z = model(x).reshape(-1)
    return z.detach().float().cpu().numpy()


def pixel_err(a, b):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {
        "max": int(d.max()),
        "mean": float(d.mean()),
        "frac_nonzero": float(np.mean(d.max(axis=-1) > 0)),
        "mean_r": float(d[..., 0].mean()),
        "mean_g": float(d[..., 1].mean()),
        "mean_b": float(d[..., 2].mean()),
    }


class Table:
    def __init__(self, path, fields):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = (not path.exists()) or path.stat().st_size == 0
        self.f = path.open("a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        self.w.writerow({k: row.get(k, "") for k in self.w.fieldnames})
        self.f.flush()


def done_paths(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open() as f:
        return {r["path"] for r in csv.DictReader(f)}


def select_items(data_root, limit):
    _, test = list_split(data_root / "AttGAN", 400, 0, group=True)
    jobs = [("AttGAN", "attgan_test", p, lab) for p, lab in test]
    rng = np.random.default_rng(1)
    for gen in ("BEGAN", "SNGAN", "STGAN"):
        items = list_images(data_root / gen)
        picked = []
        for lab in (0, 1):
            pool = [t for t in items if t[1] == lab]
            idx = rng.choice(len(pool), size=min(150, len(pool)), replace=False)
            picked.extend(pool[int(i)] for i in idx)
        jobs.extend((gen, f"{gen}_sub150", p, lab) for p, lab in picked)
    return jobs[:limit] if limit else jobs


def jpeg_dest(root, gen, path, arm):
    p = Path(path)
    folder = p.parent.name
    return root / gen / folder / f"{p.stem}_{arm}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "NPR-DeepfakeDetection" / "NPR.pth")
    ap.add_argument("--out-dir", type=Path, default=EXP / "results")
    ap.add_argument("--jpeg-dir", type=Path, default=EXP / "data" / "jpeg")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    digest = sha256(args.checkpoint)
    if digest != "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53":
        raise SystemExit(f"checkpoint hash mismatch {digest}")
    dev = device()
    model = load_npr(str(args.checkpoint), dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != 1437761:
        raise SystemExit(f"parameter count {n_params}")
    probe = Image.open(next(p for p, lab in list_images(args.data_root / "AttGAN") if lab == 0)).convert("RGB")
    x = as_batch([probe], dev)
    with torch.no_grad():
        err = (model(x).reshape(-1) - forward_from_u(model, u_of(model, x))).abs().max().item()
    if err > 1e-4:
        raise SystemExit(f"forward mismatch {err}")
    print(f"checkpoint ok params={n_params} device={dev} encoder={version()} wrap_err={err:.3e}", flush=True)

    pred_fields = [
        "generator", "split", "path", "label", "face_id", "width", "height",
        "z_444", "z_422", "z_440", "z_420",
        "pil_444", "pil_422", "pil_420",
        "dqt_hash", "jpeg_sha_444", "jpeg_sha_422", "jpeg_sha_440", "jpeg_sha_420",
    ]
    man_fields = [
        "generator", "split", "path", "label", "face_id", "arm", "width", "height",
        "jpeg_path", "jpeg_sha256", "sof", "qslots", "dqt_hash", "process",
        "rt_max", "rt_mean", "pil_max", "pil_mean",
    ]
    edge_fields = [
        "generator", "split", "path", "label", "face_id", "edge",
        "phi_y", "phi_c", "z00", "z10", "z01", "z11", "sum_err",
        "raw_gap", "rt_max_control", "rt_max_target",
    ]
    pred_path = args.out_dir / "factorial_predictions.csv"
    done = done_paths(pred_path)
    tables = {
        "pred": Table(pred_path, pred_fields),
        "man": Table(args.out_dir / "jpeg_factorial_manifest.csv", man_fields),
        "edge": Table(args.out_dir / "factorial_channel_shapley.csv", edge_fields),
    }
    jobs = [j for j in select_items(args.data_root, args.limit) if j[2] not in done]
    print(f"todo {len(jobs)} already {len(done)}", flush=True)
    expected_dqt = None
    batch = []

    def consume(batch):
        if not batch:
            return
        for i, rec in enumerate(batch):
            z = rec["_z"]
            tables["pred"].write({
                "generator": rec["gen"], "split": rec["split"], "path": rec["path"],
                "label": rec["label"], "face_id": rec["face"],
                "width": rec["w"], "height": rec["h"],
                "z_444": z["444"], "z_422": z["422"], "z_440": z["440"], "z_420": z["420"],
                "pil_444": z["pil_444"], "pil_422": z["pil_422"], "pil_420": z["pil_420"],
                "dqt_hash": rec["dqt"],
                "jpeg_sha_444": rec["sha"]["444"], "jpeg_sha_422": rec["sha"]["422"],
                "jpeg_sha_440": rec["sha"]["440"], "jpeg_sha_420": rec["sha"]["420"],
            })
            for arm in ARMS:
                tables["man"].write({
                    "generator": rec["gen"], "split": rec["split"], "path": rec["path"],
                    "label": rec["label"], "face_id": rec["face"], "arm": arm,
                    "width": rec["w"], "height": rec["h"],
                    "jpeg_path": rec["jpeg_path"][arm],
                    "jpeg_sha256": rec["sha"][arm],
                    "sof": rec["sof"][arm], "qslots": rec["qslots"][arm],
                    "dqt_hash": rec["dqt"], "process": "SOF0",
                    "rt_max": rec["rt"][arm]["max"], "rt_mean": rec["rt"][arm]["mean"],
                    "pil_max": rec["pil_err"][arm]["max"] if arm in rec["pil_err"] else "",
                    "pil_mean": rec["pil_err"][arm]["mean"] if arm in rec["pil_err"] else "",
                })
            for edge, (a, b) in rec["edges"].items():
                phi_y, phi_c, z00, z10, z01, z11 = a
                tables["edge"].write({
                    "generator": rec["gen"], "split": rec["split"], "path": rec["path"],
                    "label": rec["label"], "face_id": rec["face"], "edge": edge,
                    "phi_y": phi_y, "phi_c": phi_c, "z00": z00, "z10": z10, "z01": z01, "z11": z11,
                    "sum_err": (phi_y + phi_c) - (z11 - z00),
                    "raw_gap": z[b[1]] - z[b[0]],
                    "rt_max_control": rec["rt"][b[0]]["max"],
                    "rt_max_target": rec["rt"][b[1]]["max"],
                })
        print(f"wrote {batch[-1]['gen']} n={len(batch)} {Path(batch[-1]['path']).name}", flush=True)

    for gen, split, path, lab in jobs:
        clean = Image.open(path).convert("RGB")
        w, h = clean.size
        decoded = {}
        meta_sof, meta_slot, meta_sha, meta_path = {}, {}, {}, {}
        dqt = None
        for arm in ARMS:
            blob = encode(clean, arm, 95)
            parsed = parse_markers(blob)
            got = sof_tuple(parsed)
            if got != EXPECTED_SOF[arm]:
                raise SystemExit(f"SOF {arm} {got} != {EXPECTED_SOF[arm]} on {path}")
            this_dqt = dqt_hash(parsed)
            if expected_dqt is None:
                expected_dqt = this_dqt
                print(f"dqt hash {expected_dqt} qslots {qtable_ids(parsed)}", flush=True)
                if this_dqt != "28d506d794713d12460be674da01ebd14fe2262038e902c081a6946578ac98d3":
                    raise SystemExit(f"DQT hash is not the validated Q95 table {this_dqt}")
            if this_dqt != expected_dqt:
                raise SystemExit(f"DQT changed on {arm} {path}")
            dqt = this_dqt
            dest = jpeg_dest(args.jpeg_dir, gen, path, arm)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            decoded[arm] = decode(blob)
            meta_sof[arm] = str(got)
            meta_slot[arm] = str(qtable_ids(parsed))
            meta_sha[arm] = hashlib.sha256(blob).hexdigest()
            meta_path[arm] = str(dest)
        ycc = {arm: rgb_to_ycbcr(np.asarray(decoded[arm], np.float64)) for arm in ARMS}
        views = dict(decoded)
        rt_err = {}
        for arm in ARMS:
            back = Image.fromarray(mix(ycc[arm], ycc[arm], "both"), "RGB")
            views[f"rt_{arm}"] = back
            rt_err[arm] = pixel_err(decoded[arm], back)
        pil = {}
        pil_err = {}
        for arm, sub in PIL_SUB.items():
            pil[arm] = jpeg_pil(clean, 95, sub)
            views[f"pil_{arm}"] = pil[arm]
            pil_err[arm] = pixel_err(decoded[arm], pil[arm])
        edges = {}
        for a, b in EDGES:
            views[f"y_{a}_{b}"] = Image.fromarray(mix(ycc[a], ycc[b], "Y"), "RGB")
            views[f"c_{a}_{b}"] = Image.fromarray(mix(ycc[a], ycc[b], "C"), "RGB")
            edges[f"{a}_to_{b}"] = ((a, b),)
        rec = {
            "gen": gen, "split": split, "path": path, "label": int(lab),
            "face": celeb_id(path, int(lab)), "w": w, "h": h,
            "views": views, "rt": rt_err, "pil_err": pil_err, "edges": {},
            "sof": meta_sof, "qslots": meta_slot, "sha": meta_sha,
            "jpeg_path": meta_path, "dqt": dqt, "shape": (w, h),
            "edge_names": edges,
        }
        if batch and batch[0]["shape"] != rec["shape"]:
            _fill_edges(batch, model, dev)
            consume(batch)
            batch = []
        batch.append(rec)
        if len(batch) >= args.batch_size:
            _fill_edges(batch, model, dev)
            consume(batch)
            batch = []
    if batch:
        _fill_edges(batch, model, dev)
        consume(batch)
    print("done", flush=True)


def _fill_edges(batch, model, dev):
    # Logits are computed in consume; edge Shapley needs those logits.
    # Compute them here so consume can stay a writer. Both call the model.
    names = list(batch[0]["views"])
    got = {name: logits_of(model, [rec["views"][name] for rec in batch], dev) for name in names}
    for i, rec in enumerate(batch):
        z = {name: float(got[name][i]) for name in names}
        rec["z"] = z
        for edge, ((a, b),) in rec["edge_names"].items():
            z00, z11 = z[f"rt_{a}"], z[f"rt_{b}"]
            z10, z01 = z[f"y_{a}_{b}"], z[f"c_{a}_{b}"]
            phi_y = 0.5 * ((z10 - z00) + (z11 - z01))
            phi_c = 0.5 * ((z01 - z00) + (z11 - z10))
            rec["edges"][edge] = ((phi_y, phi_c, z00, z10, z01, z11), (a, b))
    # Stash logits onto the record for consume without a second forward.
    for i, rec in enumerate(batch):
        rec["_z"] = {name: float(got[name][i]) for name in names}


if __name__ == "__main__":
    main()
