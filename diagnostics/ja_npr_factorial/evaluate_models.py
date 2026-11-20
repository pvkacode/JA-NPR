#!/usr/bin/env python3
"""Score released NPR and the shipped JA-NPR weights on the existing four JPEG arms."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from runtime import (  # noqa: E402
    ARMS, CKPT_SHA, FACT, NPR_CKPT, WEIGHTS, check_jpeg, forward_ja, load_ja_module,
    load_janpr, npr_input, sha256, x255_of,
)
from npr_eval_lib import device, load_npr  # noqa: E402

MODELS = (
    ("npr", None),
    ("ja_attgan_s0", WEIGHTS / "janpr_AttGAN_seed0.pth"),
    ("ja_attgan_s1", WEIGHTS / "janpr_AttGAN_seed1.pth"),
    ("ja_attgan_s2", WEIGHTS / "janpr_AttGAN_seed2.pth"),
    ("ja_attgan_sngan_s0", WEIGHTS / "janpr_AttGAN+SNGAN_seed0.pth"),
)
EPS = 1e-8


def effects(a00, a10, a01, a11):
    def flat(t):
        return t.reshape(t.shape[0], -1)

    def l2(t):
        return flat(t).norm(dim=1)

    def cos(a, b):
        af, bf = flat(a), flat(b)
        return (af * bf).sum(1) / (af.norm(dim=1) * bf.norm(dim=1) + EPS)

    h0, h1 = a10 - a00, a11 - a01
    v0, v1 = a01 - a00, a11 - a10
    inter = a11 - a10 - a01 + a00
    phi_h, phi_v = 0.5 * (h0 + h1), 0.5 * (v0 + v1)
    return {
        "l2_A00": l2(a00), "l2_H0": l2(h0), "l2_V0": l2(v0), "l2_I": l2(inter),
        "l2_Phi_H": l2(phi_h), "l2_Phi_V": l2(phi_v),
        "l2_identity": l2(phi_h + phi_v - (a11 - a00)),
        "cos_H0_V0": cos(h0, v0), "cos_I_H0": cos(inter, h0), "cos_00_11": cos(a00, a11),
        "numel": int(a00[0].numel()),
    }


def load_jobs():
    preds = list(csv.DictReader((FACT / "results" / "factorial_predictions.csv").open()))
    loc = {(r["path"], r["arm"]): r["jpeg_path"] for r in csv.DictReader((FACT / "results" / "jpeg_factorial_manifest.csv").open())}
    jobs = []
    for row in preds:
        jobs.append({
            "generator": row["generator"], "split": row["split"], "path": row["path"],
            "label": int(row["label"]), "face_id": row["face_id"],
            "width": int(row["width"]), "height": int(row["height"]),
            "jpeg": {arm: loc[(row["path"], arm)] for arm in ARMS},
            "ref": {arm: float(row[f"z_{arm}"]) for arm in ARMS},
        })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=EXP / "results")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if sha256(NPR_CKPT) != CKPT_SHA:
        raise SystemExit("NPR checkpoint hash mismatch")
    for _, weight in MODELS:
        if weight is not None and not weight.is_file():
            raise SystemExit(f"missing JA-NPR weight {weight}")
    mod = load_ja_module()
    dev = device()
    npr = load_npr(str(NPR_CKPT), dev)
    npr.eval()
    nets = {"npr": npr}
    weight_sha = {}
    for name, weight in MODELS:
        if weight is None:
            continue
        nets[name] = load_janpr(mod, weight, dev)
        weight_sha[name] = sha256(weight)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_config.json").write_text(json.dumps({
        "npr_sha256": CKPT_SHA,
        "ja_weights": weight_sha,
        "plain_npr_checkpoint": "not in the JA-NPR zip; not evaluated",
        "pixels": "existing factorial JPEGs, djpeg -dct int, smooth upsample",
        "q": "Pillow qvec_of on those files, unchanged",
        "chroma_half": True,
        "ja_stages": list(__import__("runtime").JA_STAGES),
    }, indent=2))
    torch.save(
        {"weight": nets["ja_attgan_s0"].base.fc1.weight.detach().float().cpu(),
         "bias": nets["ja_attgan_s0"].base.fc1.bias.detach().float().cpu()},
        args.out_dir / "ja_s0_fc1.pt",
    )

    jobs = load_jobs()
    if args.limit:
        jobs = jobs[:args.limit]
    logit_path = args.out_dir / "logits.csv"
    stage_path = args.out_dir / "ja_stage_effects.csv"
    done = set()
    if logit_path.exists() and logit_path.stat().st_size:
        done = {r["path"] for r in csv.DictReader(logit_path.open())}
    todo = [j for j in jobs if j["path"] not in done]
    print(f"device {dev} todo {len(todo)} already {len(done)}", flush=True)
    shard_dir = args.out_dir / "ja_pre_fc_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    logit_fields, stage_fields = None, None
    logit_new = not logit_path.exists() or logit_path.stat().st_size == 0
    stage_new = not stage_path.exists() or stage_path.stat().st_size == 0
    logit_f = logit_path.open("a", newline="")
    stage_f = stage_path.open("a", newline="")
    logit_w = stage_w = None
    max_npr_gap = 0.0

    class Hold:
        pass

    state = Hold()
    state.acc = {}
    state.acc_n = {}
    state.box = {}

    def run_batch(batch):
        nonlocal logit_w, stage_w, logit_fields, stage_fields, max_npr_gap
        decoded, qvecs = {}, {}
        for arm in ARMS:
            images, qs = [], []
            for rec in batch:
                q, im = check_jpeg(rec["jpeg"][arm], arm, mod)
                images.append(im)
                qs.append(q)
            decoded[arm] = images
            qvecs[arm] = torch.tensor(qs, dtype=torch.float32, device=dev)
        rows, stage_rows, pres = [], [], []
        with torch.no_grad():
            npr_logits = {}
            for arm in ARMS:
                x = torch.stack([npr_input(im, dev) for im in decoded[arm]])
                npr_logits[arm] = npr(x).reshape(-1).float().cpu()
            ja_logits = {name: {} for name, w in MODELS if w is not None}
            ja_stages = None
            film_rows = {arm: None for arm in ARMS}
            for arm in ARMS:
                x = torch.stack([x255_of(im, dev) for im in decoded[arm]])
                q = qvecs[arm]
                for name in ja_logits:
                    if name == "ja_attgan_s0":
                        stages, logit, film = forward_ja(mod, nets[name], x, q)
                        ja_stages = stages if arm == "444" else ja_stages
                        state.box[arm] = stages
                        film_rows[arm] = {k: v.detach().float().cpu().numpy() for k, v in film.items()}
                        ja_logits[name][arm] = logit.float().cpu()
                    else:
                        logit, _ = nets[name](x, q, chroma_half=True)
                        ja_logits[name][arm] = logit.reshape(-1).float().cpu()
        for i, rec in enumerate(batch):
            gap = max(abs(float(npr_logits[arm][i]) - rec["ref"][arm]) for arm in ARMS)
            max_npr_gap = max(max_npr_gap, gap)
            if gap > 1e-3:
                raise SystemExit(f"NPR logit mismatch {gap} on {rec['path']}")
            base = {
                "generator": rec["generator"], "split": rec["split"], "path": rec["path"],
                "label": rec["label"], "face_id": rec["face_id"],
            }
            def pack(model, zs):
                row = dict(base)
                row["model"] = model
                for arm in ARMS:
                    row[f"z_{arm}"] = float(zs[arm][i])
                    row[f"q3_{arm}"] = float(qvecs[arm][i, 3].cpu())
                row["q_luma"] = float(qvecs["444"][i, 1].cpu())
                row["q_chroma"] = float(qvecs["444"][i, 2].cpu())
                for arm in ARMS:
                    if model == "ja_attgan_s0":
                        row[f"film_gamma_{arm}"] = float(film_rows[arm]["gamma_abs"][i])
                        row[f"film_beta_{arm}"] = float(film_rows[arm]["beta_abs"][i])
                    else:
                        row[f"film_gamma_{arm}"] = ""
                        row[f"film_beta_{arm}"] = ""
                return row
            rows.append(pack("npr", npr_logits))
            for name in ja_logits:
                rows.append(pack(name, ja_logits[name]))
            box = state.box
            for stage, _ in ((s, None) for s in (
                "ycc", "y_npr", "chroma_pooled", "chroma_npr_up", "residual", "residual_scaled",
                "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "pre_fc",
            )):
                got = effects(box["444"][stage][i:i + 1], box["422"][stage][i:i + 1],
                              box["440"][stage][i:i + 1], box["420"][stage][i:i + 1])
                rec_s = dict(base)
                rec_s["stage"] = stage
                rec_s["model"] = "ja_attgan_s0"
                for k, v in got.items():
                    rec_s[k] = float(v[0].detach().cpu()) if torch.is_tensor(v) else int(v) if k == "numel" else v
                rec_s["rel_I"] = rec_s["l2_I"] / (rec_s["l2_H0"] + rec_s["l2_V0"] + EPS)
                rec_s["rel_I_vs_A00"] = rec_s["l2_I"] / (rec_s["l2_A00"] + EPS)
                rec_s["rel_H0_vs_A00"] = rec_s["l2_H0"] / (rec_s["l2_A00"] + EPS)
                rec_s["rel_V0_vs_A00"] = rec_s["l2_V0"] / (rec_s["l2_A00"] + EPS)
                stage_rows.append(rec_s)
                if stage in ("y_npr", "chroma_npr_up", "residual", "bn1", "pre_fc"):
                    inter = (box["420"][stage][i] - box["422"][stage][i]
                             - box["440"][stage][i] + box["444"][stage][i])
                    vec = inter.detach().reshape(-1).float().cpu().double()
                    key = (rec["split"], stage, int(rec["label"]))
                    if key not in state.acc:
                        state.acc[key] = vec
                        state.acc_n[key] = 1
                    else:
                        state.acc[key] += vec
                        state.acc_n[key] += 1
            pre = np.stack([box[arm]["pre_fc"][i].detach().float().cpu().numpy() for arm in ARMS], 0)
            pres.append((rec["path"], pre))
        if logit_w is None:
            logit_fields = list(rows[0].keys())
            logit_w = csv.DictWriter(logit_f, fieldnames=logit_fields, extrasaction="ignore")
            if logit_new:
                logit_w.writeheader()
            stage_fields = list(stage_rows[0].keys())
            stage_w = csv.DictWriter(stage_f, fieldnames=stage_fields)
            if stage_new:
                stage_w.writeheader()
        for row in rows:
            logit_w.writerow(row)
        for row in stage_rows:
            stage_w.writerow(row)
        for path, pre in pres:
            np.savez_compressed(shard_dir / f"{hashlib.sha256(path.encode()).hexdigest()[:16]}.npz", path=path, pre_fc=pre)
        logit_f.flush()
        stage_f.flush()
        print(f"scored {batch[-1]['split']} {Path(batch[-1]['path']).name} npr_gap {max_npr_gap:.3e}", flush=True)

    batch = []
    for rec in todo:
        if batch and (batch[0]["width"], batch[0]["height"]) != (rec["width"], rec["height"]):
            run_batch(batch)
            batch = []
        batch.append(rec)
        if len(batch) >= args.batch_size:
            run_batch(batch)
            batch = []
    if batch:
        run_batch(batch)
    class_rows = []
    splits = sorted({k[0] for k in state.acc})
    stages = sorted({k[1] for k in state.acc})
    for split in splits:
        for stage in stages:
            if (split, stage, 0) not in state.acc or (split, stage, 1) not in state.acc:
                continue
            real = state.acc[(split, stage, 0)] / state.acc_n[(split, stage, 0)]
            fake = state.acc[(split, stage, 1)] / state.acc_n[(split, stage, 1)]
            denom = real.norm() * fake.norm()
            class_rows.append({
                "split": split, "stage": stage,
                "n_real": state.acc_n[(split, stage, 0)],
                "n_fake": state.acc_n[(split, stage, 1)],
                "l2_mean_I_real": float(real.norm()),
                "l2_mean_I_fake": float(fake.norm()),
                "l2_mean_I_gap": float((fake - real).norm()),
                "cosine_class_mean_I": float(torch.dot(real, fake) / (denom + EPS)),
            })
    with (args.out_dir / "class_mean_interaction.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        wri.writeheader()
        wri.writerows(class_rows)
    print(f"done npr_gap {max_npr_gap:.3e}", flush=True)


if __name__ == "__main__":
    main()
