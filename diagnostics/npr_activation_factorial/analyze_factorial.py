#!/usr/bin/env python3
"""Bootstrap and classifier decomposition from saved activation summaries. No model."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STAGES = ("npr", "npr_scaled", "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "pre_fc")
METRICS = (
    "rel_I", "rel_I_vs_A00", "rel_H0_vs_A00", "rel_V0_vs_A00",
    "l2_I", "l2_H0", "l2_V0", "l2_identity", "l2_interaction_identity",
    "cos_H0_V0", "cos_I_H0", "cos_I_V0", "cos_PhiH_PhiV",
    "cos_00_10", "cos_00_01", "cos_00_11", "cos_10_11", "cos_01_11",
    "mean_abs_I", "top10_energy_frac", "participation_ratio",
)


def load_rows():
    return list(csv.DictReader((RES / "stage_effects.csv").open()))


def clusters(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(int(row["label"]), str(row["face_id"]))].append(i)
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


def ci(samples, point):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def load_pre_fc(order):
    shards = {}
    for path in (RES / "pre_fc_shards").glob("*.npz"):
        blob = np.load(path, allow_pickle=True)
        shards[str(blob["path"])] = blob["pre_fc"]
    missing = [p for p in order if p not in shards]
    if missing:
        raise SystemExit(f"missing pre_fc shards: {len(missing)}")
    return np.stack([shards[p] for p in order], 0)


def bars(path, title, labels, series, ylabel, note):
    im = Image.new("RGB", (1100, 620), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 8), title, fill="black")
    dr.text((16, 28), note, fill="black")
    colors = [(31, 78, 121), (176, 0, 32), (70, 130, 90), (180, 140, 40)]
    left, right, top, bottom = 70, 1060, 70, 500
    dr.rectangle([left, top, right, bottom], outline="black")
    values = [v for s in series for v, _, _ in s]
    lo = min(0.0, min(values))
    hi = max(values) if values else 1.0
    if hi == lo:
        hi = lo + 1.0
    n = len(labels)
    group_w = (right - left - 20) / max(n, 1)
    for i, lab in enumerate(labels):
        for j, series_j in enumerate(series):
            val, clo, chi = series_j[i]
            x = left + 10 + i * group_w + j * (group_w * 0.8 / max(len(series), 1))
            bw = group_w * 0.7 / max(len(series), 1)

            def yy(v):
                return int(bottom - (v - lo) / (hi - lo) * (bottom - top - 10))

            y0, y1 = yy(0), yy(val)
            dr.rectangle([x, min(y0, y1), x + bw, max(y0, y1)], fill=colors[j % 4])
            dr.line([(x + bw / 2, yy(clo)), (x + bw / 2, yy(chi))], fill="black")
        dr.text((left + 8 + i * group_w, 510), lab, fill="black")
    dr.text((16, 560), ylabel, fill="black")
    dr.text((16, 585), "Black ticks are 95% cluster intervals. Blue real, red fake when both are drawn.", fill="black")
    im.save(path)


def hist(path, values, title, xlabel, note):
    im = Image.new("RGB", (900, 520), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 8), title, fill="black")
    dr.text((16, 28), note, fill="black")
    vals = np.asarray(values, np.float64)
    counts, edges = np.histogram(vals, bins=40)
    left, right, top, bottom = 60, 860, 60, 430
    dr.rectangle([left, top, right, bottom], outline="black")
    peak = max(int(counts.max()), 1)
    bw = (right - left) / len(counts)
    for i, c in enumerate(counts):
        h = int(c / peak * (bottom - top - 8))
        x = left + i * bw
        dr.rectangle([x + 1, bottom - h, x + bw - 1, bottom], fill=(31, 78, 121))
    dr.text((16, 450), xlabel, fill="black")
    dr.text((16, 480), f"n={vals.size}  min={vals.min():.3g}  max={vals.max():.3g}  mean={vals.mean():.3g}", fill="black")
    im.save(path)


def scatter(path, x, y, labels, title, xlabel, ylabel, note):
    im = Image.new("RGB", (820, 700), "white")
    dr = ImageDraw.Draw(im)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    dr.text((16, 8), title, fill="black")
    dr.text((16, 28), note, fill="black")
    left, right, top, bottom = 70, 780, 60, 600
    dr.rectangle([left, top, right, bottom], outline="black")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    def px(v, lo, hi, a, b):
        if hi == lo:
            return (a + b) // 2
        return int(a + (v - lo) / (hi - lo) * (b - a))

    for i in range(len(x)):
        col = (176, 0, 32) if int(labels[i]) == 1 else (31, 78, 121)
        cx, cy = px(x[i], xmin, xmax, left, right), px(y[i], ymin, ymax, bottom, top)
        dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)
    dr.text((left, 620), xlabel, fill="black")
    dr.text((left, 650), ylabel, fill="black")
    im.save(path)


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("no stage rows")
    ident = max(abs(float(r["l2_identity"])) for r in rows)
    ident_i = max(abs(float(r["l2_interaction_identity"])) for r in rows)
    logit_gap = max(
        max(abs(float(r[f"logit_{arm}"]) - float(r[f"ref_{arm}"])) for arm in ("444", "422", "440", "420"))
        for r in rows if r["stage"] == "pre_fc"
    )
    print(f"max identity L2 {ident:.3e} interaction-identity L2 {ident_i:.3e} logit gap {logit_gap:.3e}", flush=True)

    images = []
    seen = set()
    for r in rows:
        if r["stage"] == "pre_fc" and r["path"] not in seen:
            images.append(r)
            seen.add(r["path"])
    pre = load_pre_fc([r["path"] for r in images])
    weight = np.load(RES / "fc1_weight.npy")
    inter = pre[:, 3] - pre[:, 1] - pre[:, 2] + pre[:, 0]
    contrib = inter * weight[None, :]
    logit_i = np.array([
        float(r["logit_420"]) - float(r["logit_422"]) - float(r["logit_440"]) + float(r["logit_444"])
        for r in images
    ])
    recon = contrib.sum(1)
    recon_err = float(np.max(np.abs(recon - logit_i)))
    print(f"classifier reconstruction max abs {recon_err:.3e}", flush=True)

    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["stage"])].append(r)

    summary = []
    rng = np.random.default_rng(0)
    n_boot = 2000
    focus_splits = []
    for split, stage in by:
        if split not in focus_splits:
            focus_splits.append(split)
    for split in focus_splits:
        for stage in STAGES:
            sub = by[(split, stage)]
            y = np.array([int(r["label"]) for r in sub])
            groups = clusters(sub)
            scheme = "image" if sum(k[0] == 1 for k in groups) <= 1 else "face"
            arrays = {m: np.array([float(r[m]) for r in sub]) for m in METRICS}
            point = {}
            point["all"] = {m: float(arrays[m].mean()) for m in METRICS}
            point["real"] = {m: float(arrays[m][y == 0].mean()) for m in METRICS}
            point["fake"] = {m: float(arrays[m][y == 1].mean()) for m in METRICS}
            point["gap"] = {m: point["fake"][m] - point["real"][m] for m in METRICS}
            draws = {kind: {m: [] for m in METRICS} for kind in ("all", "real", "fake", "gap")}
            for _ in range(n_boot):
                ri, fi, scheme = resample(groups, y, rng)
                idx = np.concatenate([ri, fi])
                for m in METRICS:
                    draws["all"][m].append(float(arrays[m][idx].mean()))
                    draws["real"][m].append(float(arrays[m][ri].mean()))
                    draws["fake"][m].append(float(arrays[m][fi].mean()))
                    draws["gap"][m].append(float(arrays[m][fi].mean() - arrays[m][ri].mean()))
            for kind in draws:
                for m in METRICS:
                    est, lo, hi = ci(draws[kind][m], point[kind][m])
                    summary.append({
                        "split": split, "stage": stage, "group": kind, "metric": m,
                        "estimate": est, "lo": lo, "hi": hi, "scheme": scheme,
                        "n": len(sub),
                        "n_fake_clusters": sum(k[0] == 1 for k in groups),
                        "n_real_clusters": sum(k[0] == 0 for k in groups),
                    })
            if split == "attgan_test":
                print(
                    f"AttGAN {stage:11} rel_I real {point['real']['rel_I']:.3f} "
                    f"fake {point['fake']['rel_I']:.3f} gap {point['gap']['rel_I']:.3f} "
                    f"vsA00 {point['all']['rel_I_vs_A00']:.3f}",
                    flush=True,
                )

    with (RES / "stage_bootstrap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Classifier channels. AttGAN indices into `images`, which follow prediction order.
    att_idx = [i for i, r in enumerate(images) if r["split"] == "attgan_test"]
    att_rows = [images[i] for i in att_idx]
    att_c = contrib[att_idx]
    att_y = np.array([int(r["label"]) for r in att_rows])
    mean_c = att_c.mean(0)
    order = np.argsort(-np.abs(mean_c))
    top = order[:20]
    total = float(mean_c.sum())
    top_share = float(mean_c[top].sum() / total) if total != 0 else float("nan")
    abs_mass = np.abs(mean_c).sum()
    abs_share = float(np.abs(mean_c[top]).sum() / abs_mass)
    real_mean = att_c[att_y == 0].mean(0)
    fake_mean = att_c[att_y == 1].mean(0)
    real_top = set(np.argsort(-np.abs(real_mean))[:20].tolist())
    fake_top = set(np.argsort(-np.abs(fake_mean))[:20].tolist())
    overlap = len(real_top & fake_top)

    groups = clusters(att_rows)
    y = att_y
    rng = np.random.default_rng(1)
    hits = np.zeros(512)
    jaccards = []
    point_set = set(top.tolist())
    for _ in range(2000):
        ri, fi, _ = resample(groups, y, rng)
        idx = np.concatenate([ri, fi])
        # att_idx positions: resample indices are into att_rows, so into att_c.
        boot_mean = att_c[idx].mean(0)
        boot_top = set(np.argsort(-np.abs(boot_mean))[:20].tolist())
        jaccards.append(len(point_set & boot_top) / len(point_set | boot_top))
        for c in boot_top:
            hits[c] += 1
    hits /= 2000.0
    channel_rows = []
    for rank, c in enumerate(order, start=1):
        channel_rows.append({
            "channel": int(c),
            "rank": rank,
            "mean_contribution": float(mean_c[c]),
            "mean_real": float(real_mean[c]),
            "mean_fake": float(fake_mean[c]),
            "weight": float(weight[c]),
            "top20_bootstrap_frequency": float(hits[c]),
        })
    with (RES / "classifier_channels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
        w.writeheader()
        w.writerows(channel_rows)
    pos = float(mean_c[mean_c > 0].sum())
    neg = float(mean_c[mean_c < 0].sum())
    meta = {
        "reconstruction_max_abs": recon_err,
        "logit_gap_max_abs": logit_gap,
        "identity_l2_max": ident,
        "interaction_identity_l2_max": ident_i,
        "attgan_mean_logit_interaction": total,
        "top20_signed_share_of_mean": top_share,
        "top20_abs_share": abs_share,
        "positive_channel_sum": pos,
        "negative_channel_sum": neg,
        "real_fake_top20_overlap": overlap,
        "top20_jaccard_mean": float(np.mean(jaccards)),
        "top20_jaccard_lo": float(np.percentile(jaccards, 2.5)),
        "top20_jaccard_hi": float(np.percentile(jaccards, 97.5)),
        "top20": [int(c) for c in top],
        "n_attgan": len(att_idx),
        "n_fake_clusters": 29,
    }
    # Class-mean interaction vector alignment at pre_fc.
    I_att = inter[att_idx]
    cos_means = float(
        np.dot(real_mean, fake_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(fake_mean) + 1e-12)
    )
    # real_mean here is mean contribution, which is w * mean I. Cosine of contribution vectors
    # equals cosine of mean I only if compared in the weighted space. Also report raw I means.
    I_real = I_att[att_y == 0].mean(0)
    I_fake = I_att[att_y == 1].mean(0)
    cos_I = float(np.dot(I_real, I_fake) / (np.linalg.norm(I_real) * np.linalg.norm(I_fake) + 1e-12))
    meta["cosine_mean_interaction_vectors"] = cos_I
    meta["cosine_mean_contribution_vectors"] = cos_means
    meta["l2_mean_I_real"] = float(np.linalg.norm(I_real))
    meta["l2_mean_I_fake"] = float(np.linalg.norm(I_fake))
    meta["l2_mean_I_gap"] = float(np.linalg.norm(I_fake - I_real))
    (RES / "classifier_summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in (
        "reconstruction_max_abs", "attgan_mean_logit_interaction", "top20_signed_share_of_mean",
        "top20_abs_share", "real_fake_top20_overlap", "top20_jaccard_mean",
        "cosine_mean_interaction_vectors", "l2_mean_I_gap",
    )}, indent=2), flush=True)

    def series(metric, group):
        out = []
        for stage in STAGES:
            rec = next(r for r in summary if r["split"] == "attgan_test" and r["stage"] == stage and r["group"] == group and r["metric"] == metric)
            out.append((float(rec["estimate"]), float(rec["lo"]), float(rec["hi"])))
        return out

    note = "AttGAN test, 400 real and 406 fake images, 29 fake-face clusters. Means over images."
    labels = list(STAGES)
    bars(
        FIG / "stage_interaction_magnitude.png",
        "Relative interaction magnitude ||I_HV|| / ||A_444||",
        labels, [series("rel_I_vs_A00", "all")],
        "Mean ratio. Dimensionless. Not a cross-layer causal importance.",
        note + " One series: all images.",
    )
    bars(
        FIG / "stage_horizontal_vertical_magnitude.png",
        "Relative simple effects ||H0|| / ||A_444|| and ||V0|| / ||A_444||",
        labels, [series("rel_H0_vs_A00", "all"), series("rel_V0_vs_A00", "all")],
        "Blue H0, red V0. Mean ratio to the 4:4:4 activation at the same stage.",
        note,
    )
    bars(
        FIG / "stage_interaction_ratio.png",
        "Interaction ratio ||I_HV|| / (||H0|| + ||V0||)",
        labels, [series("rel_I", "all")],
        "Dimensionless ratio specified for this experiment.",
        note,
    )
    bars(
        FIG / "stage_real_fake_interaction.png",
        "Relative interaction ||I_HV|| / ||A_444|| by class",
        labels, [series("rel_I_vs_A00", "real"), series("rel_I_vs_A00", "fake")],
        "Blue real, red fake. A difference here is not by itself a ranking effect.",
        note,
    )
    hist(
        FIG / "classifier_contribution_distribution.png",
        mean_c,
        "AttGAN mean channel contribution to the logit interaction",
        "w_c times the mean interaction in channel c. Sum of bars' underlying values equals the mean logit interaction.",
        f"512 channels, averaged over {len(att_idx)} images. Descriptive, not 512 tests.",
    )
    im = Image.new("RGB", (1000, 560), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 8), "Top 20 channels by |mean contribution| to the AttGAN logit interaction", fill="black")
    dr.text((16, 28), f"n={len(att_idx)} images. Signed mean. Bootstrap frequency is printed under each bar.", fill="black")
    vals = mean_c[top]
    lo, hi = float(min(vals.min(), 0)), float(max(vals.max(), 0))
    if hi == lo:
        hi = lo + 1
    for i, c in enumerate(top):
        val = float(mean_c[c])
        x = 40 + i * 48
        y0 = 300
        y1 = int(300 - val / (hi - lo) * 200)
        col = (176, 0, 32) if val < 0 else (31, 78, 121)
        dr.line([(x + 16, y0), (x + 16, y1)], fill=col, width=16)
        dr.text((x, 80), str(int(c)), fill="black")
        dr.text((x, 470), f"{hits[c]:.2f}", fill="black")
        dr.text((x, 500), f"{val:.2f}", fill="black")
    dr.text((16, 530), "Numbers under the bars: fraction of 2,000 face-cluster bootstraps in which the channel is in the top 20.", fill="black")
    im.save(FIG / "classifier_top20_channels.png")
    scatter(
        FIG / "representation_vs_logit_interaction.png",
        np.linalg.norm(I_att, axis=1),
        logit_i[att_idx],
        att_y,
        "Per-image pre-fc interaction norm versus logit interaction",
        "||I_HV|| of the 512-d pooled vector",
        "Logit interaction I_HV",
        f"AttGAN n={len(att_idx)}. Blue real, red fake. w·I equals the logit interaction; max residual {recon_err:.2e}.",
    )
    print("wrote figures", flush=True)


if __name__ == "__main__":
    main()
