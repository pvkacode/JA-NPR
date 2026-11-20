#!/usr/bin/env python3
"""Paired factorial comparison of released NPR and JA-NPR. No model."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NPR_STAGES = EXP.parents[1] / "experiments" / "npr_activation_factorial" / "results" / "stage_effects.csv"
ARMS = ("444", "422", "440", "420")


def load_logits():
    rows = list(csv.DictReader((RES / "logits.csv").open()))
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    return by


def effects(r):
    z = {a: float(r[f"z_{a}"]) for a in ARMS}
    h0, h1 = z["422"] - z["444"], z["420"] - z["440"]
    v0, v1 = z["440"] - z["444"], z["420"] - z["422"]
    inter = z["420"] - z["422"] - z["440"] + z["444"]
    return {
        "H0": h0, "H1": h1, "V0": v0, "V1": v1, "I": inter,
        "Phi_H": 0.5 * (h0 + h1), "Phi_V": 0.5 * (v0 + v1),
        "total": z["420"] - z["444"],
        **{f"z_{a}": z[a] for a in ARMS},
    }


def clusters(rows):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(int(r["label"]), str(r["face_id"]))].append(i)
    return groups


def resample(groups, y, rng):
    pos = [k for k in groups if k[0] == 1]
    neg = [k for k in groups if k[0] == 0]
    if len(pos) <= 1:
        return (
            rng.choice(np.where(y == 0)[0], int((y == 0).sum()), replace=True),
            rng.choice(np.where(y == 1)[0], int((y == 1).sum()), replace=True),
            "image",
        )
    real, fake = [], []
    for j in rng.choice(len(neg), len(neg), replace=True):
        real.extend(groups[neg[j]])
    for j in rng.choice(len(pos), len(pos), replace=True):
        fake.extend(groups[pos[j]])
    return np.array(real), np.array(fake), "face"


def auc_of(z, y, ri, fi):
    zz = np.concatenate([z[ri], z[fi]])
    yy = np.concatenate([np.zeros(len(ri)), np.ones(len(fi))])
    return float(roc_auc_score(yy, zz))


def bal_at_zero(z, y):
    pred = z > 0
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((~pred)[y == 0].mean()) if (y == 0).any() else float("nan")
    return 0.5 * (tpr + tnr)


def ci(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bars(path, title, labels, series, names, note, ylabel):
    im = Image.new("RGB", (980, 560), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, 8), title, fill="black")
    dr.text((12, 28), note, fill="black")
    colors = [(31, 78, 121), (176, 0, 32), (70, 130, 90), (180, 140, 40)]
    left, right, top, bottom = 60, 940, 60, 430
    dr.rectangle([left, top, right, bottom], outline="black")
    vals = [v for s in series for v, _, _ in s]
    lo, hi = min(0.0, min(vals)), max(vals)
    if hi == lo:
        hi = lo + 1
    gw = (right - left - 16) / len(labels)
    for i, lab in enumerate(labels):
        for j, s in enumerate(series):
            val, clo, chi = s[i]
            bw = gw * 0.7 / len(series)
            x = left + 8 + i * gw + j * bw

            def yy(v, lo=lo, hi=hi):
                return int(bottom - (v - lo) / (hi - lo) * (bottom - top - 8))

            y0, y1 = yy(0), yy(val)
            dr.rectangle([x, min(y0, y1), x + bw - 1, max(y0, y1)], fill=colors[j])
            dr.line([(x + bw / 2, yy(clo)), (x + bw / 2, yy(chi))], fill="black")
        dr.text((left + 8 + i * gw, 440), lab, fill="black")
    dr.text((12, 480), ylabel, fill="black")
    dr.text((12, 510), "Series: " + ", ".join(names) + ". Ticks are 95% intervals.", fill="black")
    im.save(path)


def main():
    by = load_logits()
    stages = list(csv.DictReader((RES / "ja_stage_effects.csv").open()))
    npr_act = list(csv.DictReader(NPR_STAGES.open()))
    summary, paired = [], []
    rng = np.random.default_rng(0)
    models = list(by)
    for model in models:
        splits = []
        for r in by[model]:
            if r["split"] not in splits:
                splits.append(r["split"])
        for split in splits:
            sub = [r for r in by[model] if r["split"] == split]
            y = np.array([int(r["label"]) for r in sub])
            groups = clusters(sub)
            eff = [effects(r) for r in sub]
            keys = ["H0", "H1", "V0", "V1", "I", "Phi_H", "Phi_V", "total"]
            arr = {k: np.array([e[k] for e in eff]) for k in keys}
            z = {a: np.array([e[f"z_{a}"] for e in eff]) for a in ARMS}
            point_auc = {a: float(roc_auc_score(y, z[a])) for a in ARMS}
            point_ap = {a: float(average_precision_score(y, z[a])) for a in ARMS}
            point_bal = {a: bal_at_zero(z[a], y) for a in ARMS}
            means = {}
            for lab, name in ((None, "all"), (0, "real"), (1, "fake")):
                mask = np.ones(len(sub), dtype=bool) if lab is None else (y == lab)
                means[name] = {k: float(arr[k][mask].mean()) for k in keys}
            means["gap"] = {k: means["fake"][k] - means["real"][k] for k in keys}
            ident = max(abs((e["Phi_H"] + e["Phi_V"]) - e["total"]) for e in eff)
            draws = {a: [] for a in list(ARMS) + ["auc_I"]}
            edraw = {f"{name}_{k}": [] for name in ("all", "real", "fake", "gap") for k in ("I", "Phi_H", "Phi_V")}
            for _ in range(2000):
                ri, fi, scheme = resample(groups, y, rng)
                aucs = {a: auc_of(z[a], y, ri, fi) for a in ARMS}
                for a, v in aucs.items():
                    draws[a].append(v)
                draws["auc_I"].append(aucs["420"] - aucs["422"] - aucs["440"] + aucs["444"])
                idx = np.concatenate([ri, fi])
                for k in ("I", "Phi_H", "Phi_V"):
                    edraw[f"all_{k}"].append(float(arr[k][idx].mean()))
                    edraw[f"real_{k}"].append(float(arr[k][ri].mean()))
                    edraw[f"fake_{k}"].append(float(arr[k][fi].mean()))
                    edraw[f"gap_{k}"].append(float(arr[k][fi].mean() - arr[k][ri].mean()))
            scheme = "image" if sum(k[0] == 1 for k in groups) <= 1 else "face"
            for a in ARMS:
                est, lo, hi = ci(draws[a], point_auc[a])
                summary.append({
                    "model": model, "split": split, "quantity": f"auc_{a}",
                    "estimate": est, "lo": lo, "hi": hi, "ap": point_ap[a],
                    "bal_acc_logit_gt_0": point_bal[a], "scheme": scheme, "n": len(sub),
                    "identity_max": ident,
                    "n_fake_clusters": sum(k[0] == 1 for k in groups),
                })
            est, lo, hi = ci(draws["auc_I"], point_auc["420"] - point_auc["422"] - point_auc["440"] + point_auc["444"])
            summary.append({
                "model": model, "split": split, "quantity": "auc_I",
                "estimate": est, "lo": lo, "hi": hi, "ap": "", "bal_acc_logit_gt_0": "",
                "scheme": scheme, "n": len(sub), "identity_max": ident,
                "n_fake_clusters": sum(k[0] == 1 for k in groups),
            })
            for name in ("all", "real", "fake", "gap"):
                for k in ("I", "Phi_H", "Phi_V"):
                    est, lo, hi = ci(edraw[f"{name}_{k}"], means[name][k])
                    summary.append({
                        "model": model, "split": split, "quantity": f"{name}_{k}",
                        "estimate": est, "lo": lo, "hi": hi, "ap": "", "bal_acc_logit_gt_0": "",
                        "scheme": scheme, "n": len(sub), "identity_max": ident,
                        "n_fake_clusters": sum(k0[0] == 1 for k0 in groups),
                    })
            if split == "attgan_test":
                print(model, {a: round(point_auc[a], 3) for a in ARMS}, "I", round(means["all"]["I"], 2),
                      "gapI", round(means["gap"]["I"], 2), "realI", round(means["real"]["I"], 2),
                      "fakeI", round(means["fake"]["I"], 2), flush=True)

    # Paired AttGAN AUC and interaction, JA seed 0 minus NPR, same draws.
    npr = by["npr"]
    ja = by["ja_attgan_s0"]
    # same order
    assert [r["path"] for r in npr] == [r["path"] for r in ja]
    for split in ("attgan_test", "BEGAN_sub150", "SNGAN_sub150", "STGAN_sub150"):
        a = [r for r in npr if r["split"] == split]
        b = [r for r in ja if r["split"] == split]
        y = np.array([int(r["label"]) for r in a])
        groups = clusters(a)
        za = {arm: np.array([float(r[f"z_{arm}"]) for r in a]) for arm in ARMS}
        zb = {arm: np.array([float(r[f"z_{arm}"]) for r in b]) for arm in ARMS}
        ia = np.array([effects(r)["I"] for r in a])
        ib = np.array([effects(r)["I"] for r in b])
        rng = np.random.default_rng(1)
        bucket = {arm: [] for arm in list(ARMS) + ["I", "gapI"]}
        for _ in range(2000):
            ri, fi, scheme = resample(groups, y, rng)
            for arm in ARMS:
                bucket[arm].append(auc_of(zb[arm], y, ri, fi) - auc_of(za[arm], y, ri, fi))
            idx = np.concatenate([ri, fi])
            bucket["I"].append(float(ib[idx].mean() - ia[idx].mean()))
            bucket["gapI"].append(float((ib[fi].mean() - ib[ri].mean()) - (ia[fi].mean() - ia[ri].mean())))
        scheme = "image" if sum(k[0] == 1 for k in groups) <= 1 else "face"
        for key, point in (
            *[(arm, float(roc_auc_score(y, zb[arm]) - roc_auc_score(y, za[arm]))) for arm in ARMS],
            ("I", float(ib.mean() - ia.mean())),
            ("gapI", float((ib[y == 1].mean() - ib[y == 0].mean()) - (ia[y == 1].mean() - ia[y == 0].mean()))),
        ):
            est, lo, hi = ci(bucket[key], point)
            paired.append({"split": split, "quantity": f"ja_s0_minus_npr_{key}", "estimate": est, "lo": lo, "hi": hi, "scheme": scheme})
            if split == "attgan_test":
                print(f"paired {key} {est:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

    # Activation ratios.
    def stage_mean(rows, split, stage, metric="rel_I"):
        sub = [r for r in rows if r["split"] == split and r["stage"] == stage]
        return sub

    act_rows = []
    ja_s = stage_mean(stages, "attgan_test", "residual")
    npr_s = [r for r in npr_act if r["split"] == "attgan_test" and r["stage"] == "npr"]
    # align by path
    npr_map = {r["path"]: r for r in npr_s}
    ja_map = {r["path"]: r for r in ja_s}
    paths = [p for p in ja_map if p in npr_map]
    y = np.array([int(ja_map[p]["label"]) for p in paths])
    face = [ja_map[p]["face_id"] for p in paths]
    fake_rows = [{"label": int(ja_map[p]["label"]), "face_id": ja_map[p]["face_id"]} for p in paths]
    groups = clusters(fake_rows)
    ja_rel = np.array([float(ja_map[p]["rel_I"]) for p in paths])
    npr_rel = np.array([float(npr_map[p]["rel_I"]) for p in paths])
    ja_pre = {r["path"]: r for r in stages if r["split"] == "attgan_test" and r["stage"] == "pre_fc"}
    npr_pre = {r["path"]: r for r in npr_act if r["split"] == "attgan_test" and r["stage"] == "pre_fc"}
    ja_pre_v = np.array([float(ja_pre[p]["rel_I"]) for p in paths])
    npr_pre_v = np.array([float(npr_pre[p]["rel_I"]) for p in paths])
    rng = np.random.default_rng(2)
    buckets = {k: [] for k in (
        "ja_res", "npr_res", "diff_res", "ja_pre", "npr_pre", "diff_pre",
        "ja_res_real", "ja_res_fake", "npr_res_real", "npr_res_fake",
    )}
    for _ in range(2000):
        ri, fi, _ = resample(groups, y, rng)
        idx = np.concatenate([ri, fi])
        buckets["ja_res"].append(float(ja_rel[idx].mean()))
        buckets["npr_res"].append(float(npr_rel[idx].mean()))
        buckets["diff_res"].append(float((ja_rel - npr_rel)[idx].mean()))
        buckets["ja_pre"].append(float(ja_pre_v[idx].mean()))
        buckets["npr_pre"].append(float(npr_pre_v[idx].mean()))
        buckets["diff_pre"].append(float((ja_pre_v - npr_pre_v)[idx].mean()))
        buckets["ja_res_real"].append(float(ja_rel[ri].mean()))
        buckets["ja_res_fake"].append(float(ja_rel[fi].mean()))
        buckets["npr_res_real"].append(float(npr_rel[ri].mean()))
        buckets["npr_res_fake"].append(float(npr_rel[fi].mean()))
    points = {
        "ja_residual_rel_I": (float(ja_rel.mean()), "ja_res"),
        "npr_residual_rel_I": (float(npr_rel.mean()), "npr_res"),
        "ja_minus_npr_residual_rel_I": (float((ja_rel - npr_rel).mean()), "diff_res"),
        "ja_pre_rel_I": (float(ja_pre_v.mean()), "ja_pre"),
        "npr_pre_rel_I": (float(npr_pre_v.mean()), "npr_pre"),
        "ja_minus_npr_pre_rel_I": (float((ja_pre_v - npr_pre_v).mean()), "diff_pre"),
        "ja_residual_rel_I_real": (float(ja_rel[y == 0].mean()), "ja_res_real"),
        "ja_residual_rel_I_fake": (float(ja_rel[y == 1].mean()), "ja_res_fake"),
        "npr_residual_rel_I_real": (float(npr_rel[y == 0].mean()), "npr_res_real"),
        "npr_residual_rel_I_fake": (float(npr_rel[y == 1].mean()), "npr_res_fake"),
    }
    for key, (point, bkey) in points.items():
        est, lo, hi = ci(buckets[bkey], point)
        act_rows.append({"split": "attgan_test", "quantity": key, "estimate": est, "lo": lo, "hi": hi})
        print(f"{key} {est:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    # Stage means for the figure.
    stage_order = ["ycc", "y_npr", "chroma_pooled", "chroma_npr_up", "residual", "residual_scaled",
                   "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "pre_fc"]
    stage_labels = ["YCbCr", "Y NPR", "C pool", "C NPR", "resid", "scaled",
                    "conv1", "bn1", "relu", "pool", "layer1", "layer2", "pooled"]
    for stage in stage_order:
        vals = [float(r["rel_I"]) for r in stages if r["split"] == "attgan_test" and r["stage"] == stage]
        print(f"stage {stage} rel_I {np.mean(vals):.3f}", flush=True)

    # Classifier.
    order_paths = [r["path"] for r in by["ja_attgan_s0"] if r["split"] == "attgan_test"]
    shards = {}
    for path in (RES / "ja_pre_fc_shards").glob("*.npz"):
        blob = np.load(path, allow_pickle=True)
        shards[str(blob["path"])] = blob["pre_fc"]
    pre = np.stack([shards[p] for p in order_paths])
    blob = torch.load(RES / "ja_s0_fc1.pt", map_location="cpu", weights_only=False)
    w = blob["weight"].numpy().reshape(-1)
    inter = pre[:, 3] - pre[:, 1] - pre[:, 2] + pre[:, 0]
    contrib = inter * w[None, :]
    logits_i = np.array([effects(r)["I"] for r in by["ja_attgan_s0"] if r["split"] == "attgan_test"])
    recon = float(np.max(np.abs(contrib.sum(1) - logits_i)))
    mean_c = contrib.mean(0)
    order = np.argsort(-np.abs(mean_c))
    abs_cume = np.cumsum(np.abs(mean_c[order]))
    abs_total = np.abs(mean_c).sum()
    k50 = int(np.searchsorted(abs_cume / abs_total, 0.5) + 1)
    k80 = int(np.searchsorted(abs_cume / abs_total, 0.8) + 1)
    y_att = np.array([int(r["label"]) for r in by["ja_attgan_s0"] if r["split"] == "attgan_test"])
    I_real, I_fake = inter[y_att == 0].mean(0), inter[y_att == 1].mean(0)
    cos_I = float(np.dot(I_real, I_fake) / (np.linalg.norm(I_real) * np.linalg.norm(I_fake) + 1e-12))
    meta = {
        "reconstruction_max_abs": recon,
        "mean_logit_interaction": float(mean_c.sum()),
        "positive_sum": float(mean_c[mean_c > 0].sum()),
        "negative_sum": float(mean_c[mean_c < 0].sum()),
        "top20_signed_share": float(mean_c[order[:20]].sum() / mean_c.sum()),
        "channels_for_half_abs": k50,
        "channels_for_80_abs": k80,
        "cosine_class_mean_I": cos_I,
        "l2_mean_I_real": float(np.linalg.norm(I_real)),
        "l2_mean_I_fake": float(np.linalg.norm(I_fake)),
        "l2_mean_I_gap": float(np.linalg.norm(I_fake - I_real)),
        "top20": [int(c) for c in order[:20]],
    }
    print(json.dumps(meta, indent=2), flush=True)
    (RES / "classifier_summary.json").write_text(json.dumps(meta, indent=2))
    with (RES / "classifier_channels.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["channel", "mean_contribution", "abs_rank"])
        wri.writeheader()
        rank = {int(c): i + 1 for i, c in enumerate(order)}
        for c in range(len(mean_c)):
            wri.writerow({"channel": c, "mean_contribution": float(mean_c[c]), "abs_rank": rank[c]})
    with (RES / "factorial_summary.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wri.writeheader()
        wri.writerows(summary)
    with (RES / "paired_differences.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        wri.writeheader()
        wri.writerows(paired)
    with (RES / "activation_comparison.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(act_rows[0].keys()))
        wri.writeheader()
        wri.writerows(act_rows)

    def series(model, qty):
        out = []
        for arm in ARMS:
            rec = next(r for r in summary if r["model"] == model and r["split"] == "attgan_test" and r["quantity"] == f"auc_{arm}")
            out.append((float(rec["estimate"]), float(rec["lo"]), float(rec["hi"])))
        return out

    note = "AttGAN test, 400 real + 406 fake, 29 fake-face clusters. Same images and JPEGs."
    bars(FIG / "auc_four_arms.png", "Raw-logit AUC on the four Q95 arms",
         list(ARMS), [series("npr", None), series("ja_attgan_s0", None)],
         ["released NPR", "JA-NPR AttGAN seed 0"], note, "AUC")
    # interaction means
    def iser(model):
        out = []
        for qty in ("real_I", "fake_I", "gap_I", "all_I"):
            rec = next(r for r in summary if r["model"] == model and r["split"] == "attgan_test" and r["quantity"] == qty)
            out.append((float(rec["estimate"]), float(rec["lo"]), float(rec["hi"])))
        return out
    bars(FIG / "logit_interaction.png", "Logit interaction I_HV",
         ["real", "fake", "fake-real", "all"], [iser("npr"), iser("ja_attgan_s0")],
         ["released NPR", "JA-NPR seed 0"], note, "Logits. Negative means the second factor reverses the first.")
    npr_res = next(r for r in act_rows if r["quantity"] == "npr_residual_rel_I")
    ja_res = next(r for r in act_rows if r["quantity"] == "ja_residual_rel_I")
    npr_pre = next(r for r in act_rows if r["quantity"] == "npr_pre_rel_I")
    ja_pre_row = next(r for r in act_rows if r["quantity"] == "ja_pre_rel_I")

    def triple(rec):
        return (float(rec["estimate"]), float(rec["lo"]), float(rec["hi"]))

    bars(FIG / "residual_interaction_ratio.png", "Interaction ratio at the residual and the pool",
         ["residual", "pooled 512"],
         [[triple(npr_res), triple(npr_pre)], [triple(ja_res), triple(ja_pre_row)]],
         ["released NPR", "JA-NPR seed 0"],
         note + " Ratio ||I||/(||H0||+||V0||). Operators differ, so residual L2 norms are not compared.",
         "Dimensionless ratio")
    # The bars() function expects series aligned to labels. I passed mismatched shapes. Fix by a simpler image below.
    stage_vals = []
    for stage in stage_order:
        vals = [float(r["rel_I"]) for r in stages if r["split"] == "attgan_test" and r["stage"] == stage]
        stage_vals.append((float(np.mean(vals)), float(np.mean(vals)), float(np.mean(vals))))
    ja_real = next(r for r in act_rows if r["quantity"] == "ja_residual_rel_I_real")
    ja_fake = next(r for r in act_rows if r["quantity"] == "ja_residual_rel_I_fake")
    npr_real = next(r for r in act_rows if r["quantity"] == "npr_residual_rel_I_real")
    npr_fake = next(r for r in act_rows if r["quantity"] == "npr_residual_rel_I_fake")
    bars(FIG / "real_fake_residual_ratio.png", "Residual interaction ratio by class",
         ["real", "fake"],
         [[triple(npr_real), triple(npr_fake)], [triple(ja_real), triple(ja_fake)]],
         ["released NPR", "JA-NPR seed 0"], note, "||I|| / (||H0|| + ||V0||)")
    bars(FIG / "pooled_interaction_ratio.png", "Pooled representation interaction ratio",
         ["pre-classifier 512"],
         [[triple(npr_pre)], [triple(ja_pre_row)]],
         ["released NPR", "JA-NPR seed 0"], note, "||I|| / (||H0|| + ||V0||)")
    bars(FIG / "stage_interaction_ratio.png", "JA-NPR seed 0 interaction ratio by stage",
         stage_labels, [stage_vals], ["JA-NPR default chroma-aligned residual, then the trunk"],
         note + " Point means. Residual 0.620 and pooled 3.578 have intervals in activation_comparison.csv.",
         "||I|| / (||H0|| + ||V0||)")
    # channel hist
    im = Image.new("RGB", (900, 480), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, 8), "JA-NPR seed 0 mean channel contribution to the logit interaction", fill="black")
    dr.text((12, 28), f"AttGAN n={len(order_paths)}. Sum of the 512 values is the mean logit interaction.", fill="black")
    counts, edges = np.histogram(mean_c, bins=40)
    left, right, top, bottom = 50, 860, 60, 400
    dr.rectangle([left, top, right, bottom], outline="black")
    peak = max(int(counts.max()), 1)
    bw = (right - left) / len(counts)
    for i, c in enumerate(counts):
        h = int(c / peak * (bottom - top - 8))
        x = left + i * bw
        color = (176, 0, 32) if edges[i] < 0 else (31, 78, 121)
        dr.rectangle([x + 1, bottom - h, x + bw - 1, bottom], fill=color)
    zero = left + (0 - edges[0]) / (edges[-1] - edges[0]) * (right - left)
    dr.line([(zero, top), (zero, bottom)], fill="black")
    for frac in (0, 0.5, 1):
        val = edges[0] + frac * (edges[-1] - edges[0])
        x = left + frac * (right - left)
        dr.text((x - 16, bottom + 4), f"{val:.2f}", fill="black")
    dr.text((12, 430), "Channel contribution in logits. Red bins are negative, blue bins are positive. Zero is the black line.", fill="black")
    dr.text((12, 450), "Descriptive. Channels are not independent samples.", fill="black")
    im.save(FIG / "classifier_contributions.png")
    print("wrote summaries", flush=True)


if __name__ == "__main__":
    main()
