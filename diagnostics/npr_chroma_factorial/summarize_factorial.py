#!/usr/bin/env python3
"""Contrasts, cluster bootstrap, and pairwise margins from saved logits. No model."""

import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
FIG = EXP / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    with (RES / name).open() as f:
        return list(csv.DictReader(f))


def effects(z00, z10, z01, z11):
    h0 = z10 - z00
    h1 = z11 - z01
    v0 = z01 - z00
    v1 = z11 - z10
    phi_h = 0.5 * (h0 + h1)
    phi_v = 0.5 * (v0 + v1)
    inter = z11 - z10 - z01 + z00
    return {
        "H0": h0, "H1": h1, "V0": v0, "V1": v1,
        "Phi_H": phi_h, "Phi_V": phi_v, "I_HV": inter,
        "total": z11 - z00,
    }


def clusters(rows):
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((int(r["label"]), str(r["face_id"])), []).append(i)
    return groups


def resample(groups, rng):
    pos = [k for k in groups if k[0] == 1]
    neg = [k for k in groups if k[0] == 0]
    image_level = len(pos) <= 1
    if image_level:
        pos_idx = [i for k in pos for i in groups[k]]
        neg_idx = [i for k in neg for i in groups[k]]
        return (
            rng.choice(neg_idx, len(neg_idx), replace=True),
            rng.choice(pos_idx, len(pos_idx), replace=True),
            "image",
        )
    real, fake = [], []
    for j in rng.choice(len(neg), len(neg), replace=True):
        real.extend(groups[neg[j]])
    for j in rng.choice(len(pos), len(pos), replace=True):
        fake.extend(groups[pos[j]])
    return np.array(real), np.array(fake), "face"


def class_gap(v, y, real_idx=None, fake_idx=None):
    if real_idx is None:
        return float(v[y == 1].mean() - v[y == 0].mean())
    return float(v[fake_idx].mean() - v[real_idx].mean())


def auc_of(z, y, real_idx, fake_idx):
    zz = np.concatenate([z[real_idx], z[fake_idx]])
    yy = np.concatenate([np.zeros(len(real_idx)), np.ones(len(fake_idx))])
    return float(roc_auc_score(yy, zz))


def pair_auc(zr, zf):
    m = zf[:, None] - zr[None, :]
    return float(np.mean((m > 0) + 0.5 * (m == 0)))


def transition(m0, m1):
    c0, c1 = m0 > 0, m1 > 0
    t0, t1 = m0 == 0, m1 == 0
    w0, w1 = m0 < 0, m1 < 0
    return {
        "stay_correct": float(np.mean(c0 & c1)),
        "correct_to_wrong": float(np.mean(c0 & w1)),
        "wrong_to_correct": float(np.mean(w0 & c1)),
        "to_tie": float(np.mean(~t0 & t1)),
        "stay_wrong": float(np.mean(w0 & w1)),
        "mean_margin_change": float(np.mean(m1 - m0)),
    }


def scatter(path, xs, ys, title, xlab, ylab, identity=False):
    y = np.array(ys)
    x = np.array(xs)
    W, H = 720, 620
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    left, right, top, bottom = 70, 690, 40, 540
    dr.rectangle([left, top, right, bottom], outline="black")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if identity:
        lo, hi = min(xmin, ymin), max(xmax, ymax)
        xmin = ymin = lo
        xmax = ymax = hi
    def px(v, lo, hi, a, b):
        if hi == lo:
            return (a + b) // 2
        return int(a + (float(v) - lo) / (hi - lo) * (b - a))
    # Caller passes parallel class labels via attribute if present.
    labs = getattr(scatter, "labs", None)
    for i in range(len(x)):
        col = (176, 0, 32) if labs is not None and labs[i] == 1 else (31, 78, 121)
        cx, cy = px(x[i], xmin, xmax, left, right), px(y[i], ymin, ymax, bottom, top)
        dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)
    if identity:
        dr.line([(left, bottom), (right, top)], fill=(120, 120, 120))
    dr.text((left, 8), title, fill="black")
    dr.text((left, 560), xlab, fill="black")
    dr.text((left, 585), ylab, fill="black")
    im.save(path)


def bars(path, title, groups, series):
    im = Image.new("RGB", (860, 460), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 10), title, fill="black")
    colors = [(40, 90, 140), (70, 130, 90), (180, 140, 40), (160, 50, 50)]
    for gi, (gname, values) in enumerate(groups):
        for j, val in enumerate(values):
            bh = int(max(0.0, min(1.0, val)) * 280)
            x = 150 + gi * 180 + j * 32
            dr.rectangle([x, 360 - bh, x + 24, 360], fill=colors[j % 4])
            dr.text((x - 4, 364), f"{val:.2f}", fill="black")
        dr.text((150 + gi * 180, 400), gname, fill="black")
    dr.text((16, 430), "bars: " + ", ".join(series), fill="black")
    im.save(path)


def main():
    rows = load("factorial_predictions.csv")
    edges = load("factorial_channel_shapley.csv")
    hybrid_gap = [abs((float(r["z11"]) - float(r["z00"])) - float(r["raw_gap"])) for r in edges]
    print(
        f"rows {len(rows)} edges {len(edges)} max hybrid-vs-decoded {max(hybrid_gap):.4f} "
        f"max sum_err {max(abs(float(r['sum_err'])) for r in edges):.3e}",
        flush=True,
    )
    n_boot = 2000
    contrast_rows = []
    keys = ["H0", "H1", "V0", "V1", "Phi_H", "Phi_V", "I_HV", "total"]
    by_split = {}
    for r in rows:
        z = {k: float(r[k]) for k in ("z_444", "z_422", "z_440", "z_420")}
        e = effects(z["z_444"], z["z_422"], z["z_440"], z["z_420"])
        ident = abs((e["Phi_H"] + e["Phi_V"]) - e["total"])
        inter_ok = abs(e["I_HV"] - (e["H1"] - e["H0"]))
        rec = {
            "generator": r["generator"], "split": r["split"], "path": r["path"],
            "label": r["label"], "face_id": r["face_id"],
            **{k: e[k] for k in keys},
            "identity_err": ident, "interaction_identity_err": inter_ok,
            "pil_gap_444": float(r["z_444"]) - float(r["pil_444"]),
            "pil_gap_422": float(r["z_422"]) - float(r["pil_422"]),
            "pil_gap_420": float(r["z_420"]) - float(r["pil_420"]),
        }
        contrast_rows.append(rec)
        by_split.setdefault(r["split"], []).append(rec)
    with (RES / "factorial_logit_contrasts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(contrast_rows[0].keys()))
        w.writeheader()
        w.writerows(contrast_rows)
    orders = []
    for r in contrast_rows:
        route_a = float(r["H0"]) + float(r["V1"])
        route_b = float(r["V0"]) + float(r["H1"])
        orders.append({
            "generator": r["generator"], "split": r["split"], "path": r["path"],
            "label": r["label"], "face_id": r["face_id"],
            "routeA_H": r["H0"], "routeA_V": r["V1"], "routeA_total": route_a,
            "routeB_V": r["V0"], "routeB_H": r["H1"], "routeB_total": route_b,
            "shapley_H": r["Phi_H"], "shapley_V": r["Phi_V"],
            "route_total_err": abs(route_a - route_b),
            "interaction": r["I_HV"],
        })
    with (RES / "factorial_path_orders.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(orders[0].keys()))
        w.writeheader()
        w.writerows(orders)
    max_id = max(abs(float(r["identity_err"])) for r in contrast_rows)
    max_pil = max(max(abs(float(r["pil_gap_444"])), abs(float(r["pil_gap_422"])), abs(float(r["pil_gap_420"]))) for r in contrast_rows)
    print(f"identity max {max_id:.3e} pillow-turbo logit max abs {max_pil:.3e}", flush=True)

    boot_rows = []
    auc_rows = []
    trans_rows = []
    rng = np.random.default_rng(0)
    for split, sub in by_split.items():
        y = np.array([int(r["label"]) for r in sub])
        groups = clusters(sub)
        scheme = "face" if sum(k[0] == 1 for k in groups) > 1 else "image"
        arrays = {k: np.array([float(r[k]) for r in sub]) for k in keys}
        zmap = {
            "444": np.array([float(r["z_444"]) for r in rows if r["split"] == split]),
            "422": np.array([float(r["z_422"]) for r in rows if r["split"] == split]),
            "440": np.array([float(r["z_440"]) for r in rows if r["split"] == split]),
            "420": np.array([float(r["z_420"]) for r in rows if r["split"] == split]),
        }
        # rows and sub are the same order if we filtered in file order. Rebuild from sub's paths.
        order = {r["path"]: i for i, r in enumerate(sub)}
        src = [r for r in rows if r["split"] == split]
        assert [r["path"] for r in src] == [r["path"] for r in sub]
        point_gap = {k: class_gap(arrays[k], y) for k in keys}
        auc_point = {arm: float(roc_auc_score(y, zmap[arm])) for arm in zmap}
        ap_point = {arm: float(average_precision_score(y, zmap[arm])) for arm in zmap}
        print(split, {k: round(v, 4) for k, v in auc_point.items()}, "G_H", round(point_gap["Phi_H"], 2), "G_V", round(point_gap["Phi_V"], 2), "G_I", round(point_gap["I_HV"], 2), flush=True)
        draws = {k: [] for k in list(keys) + ["auc_444", "auc_422", "auc_440", "auc_420", "auc_H", "auc_V", "auc_I"]}
        # Pairwise on the original sample, AttGAN and others.
        real = np.where(y == 0)[0]
        fake = np.where(y == 1)[0]
        margins = {arm: zmap[arm][fake][:, None] - zmap[arm][real][None, :] for arm in zmap}
        base_auc = {arm: pair_auc(zmap[arm][real], zmap[arm][fake]) for arm in zmap}
        for arm in zmap:
            if abs(base_auc[arm] - auc_point[arm]) > 1e-6:
                print("pairwise AUC mismatch", split, arm, base_auc[arm], auc_point[arm])
        comparisons = [
            ("444_to_422", "444", "422"),
            ("444_to_440", "444", "440"),
            ("422_to_420", "422", "420"),
            ("440_to_420", "440", "420"),
            ("444_to_420", "444", "420"),
        ]
        point_trans = {}
        for name, a, b in comparisons:
            point_trans[name] = transition(margins[a], margins[b])
        trans_draws = {name: {k: [] for k in point_trans[name]} for name, _, _ in comparisons}
        for _ in range(n_boot):
            ri, fi, scheme = resample(groups, rng)
            for k in keys:
                draws[k].append(class_gap(arrays[k], y, ri, fi))
            aucs = {arm: auc_of(zmap[arm], y, ri, fi) for arm in zmap}
            for arm, val in aucs.items():
                draws[f"auc_{arm}"].append(val)
            draws["auc_H"].append(0.5 * ((aucs["422"] - aucs["444"]) + (aucs["420"] - aucs["440"])))
            draws["auc_V"].append(0.5 * ((aucs["440"] - aucs["444"]) + (aucs["420"] - aucs["422"])))
            draws["auc_I"].append(aucs["420"] - aucs["422"] - aucs["440"] + aucs["444"])
            bm = {arm: zmap[arm][fi][:, None] - zmap[arm][ri][None, :] for arm in zmap}
            for name, a, b in comparisons:
                got = transition(bm[a], bm[b])
                for kk, vv in got.items():
                    trans_draws[name][kk].append(vv)
        def ci(samples, point):
            lo, hi = np.percentile(samples, [2.5, 97.5])
            return float(point), float(lo), float(hi)
        for k in keys:
            est, lo, hi = ci(draws[k], point_gap[k])
            boot_rows.append({
                "split": split, "quantity": {"Phi_H": "G_H", "Phi_V": "G_V", "I_HV": "G_HV"}.get(k, k),
                "estimate": est, "lo": lo, "hi": hi, "scheme": scheme,
                "n": len(sub), "n_fake_clusters": sum(k0 == 1 for k0, _ in groups),
                "n_real_clusters": sum(k0 == 0 for k0, _ in groups),
            })
        for arm in zmap:
            est, lo, hi = ci(draws[f"auc_{arm}"], auc_point[arm])
            auc_rows.append({"split": split, "quantity": f"auc_{arm}", "estimate": est, "lo": lo, "hi": hi, "ap": ap_point[arm], "scheme": scheme, "n": len(sub)})
        for q in ("auc_H", "auc_V", "auc_I"):
            point = {
                "auc_H": 0.5 * ((auc_point["422"] - auc_point["444"]) + (auc_point["420"] - auc_point["440"])),
                "auc_V": 0.5 * ((auc_point["440"] - auc_point["444"]) + (auc_point["420"] - auc_point["422"])),
                "auc_I": auc_point["420"] - auc_point["422"] - auc_point["440"] + auc_point["444"],
            }[q]
            est, lo, hi = ci(draws[q], point)
            auc_rows.append({"split": split, "quantity": q, "estimate": est, "lo": lo, "hi": hi, "ap": "", "scheme": scheme, "n": len(sub)})
        for name, _, _ in comparisons:
            for kk, point in point_trans[name].items():
                est, lo, hi = ci(trans_draws[name][kk], point)
                trans_rows.append({"split": split, "comparison": name, "quantity": kk, "estimate": est, "lo": lo, "hi": hi, "scheme": scheme, "n_pairs": int(len(real) * len(fake))})

    # Class-conditional means of the contrasts.
    cond_rows = []
    for split, sub in by_split.items():
        for lab, name in ((0, "real"), (1, "fake")):
            part = [r for r in sub if int(r["label"]) == lab]
            for k in keys:
                v = np.array([float(r[k]) for r in part])
                cond_rows.append({
                    "split": split, "class": name, "quantity": k, "n": len(part),
                    "mean": float(v.mean()), "median": float(np.median(v)),
                    "std": float(v.std()),
                    "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                })
    # Edge shapley gaps.
    edge_boot = []
    for split in by_split:
        for edge in ("444_to_422", "444_to_440", "440_to_420", "422_to_420"):
            sub = [r for r in edges if r["split"] == split and r["edge"] == edge]
            y = np.array([int(r["label"]) for r in sub])
            groups = clusters(sub)
            fields = {
                "phi_y": np.array([float(r["phi_y"]) for r in sub]),
                "phi_c": np.array([float(r["phi_c"]) for r in sub]),
            }
            fields["phi_c_minus_y"] = fields["phi_c"] - fields["phi_y"]
            point = {k: class_gap(v, y) for k, v in fields.items()}
            bucket = {k: [] for k in fields}
            rng_e = np.random.default_rng(1)
            scheme = "face"
            for _ in range(n_boot):
                ri, fi, scheme = resample(groups, rng_e)
                for k, v in fields.items():
                    bucket[k].append(class_gap(v, y, ri, fi))
            for k, v in fields.items():
                lo, hi = np.percentile(bucket[k], [2.5, 97.5])
                edge_boot.append({
                    "split": split, "edge": edge, "quantity": k, "estimate": point[k],
                    "lo": float(lo), "hi": float(hi), "scheme": scheme, "n": len(sub),
                    "mean_real": float(v[y == 0].mean()), "mean_fake": float(v[y == 1].mean()),
                    "sum_err_max": max(abs(float(r["sum_err"])) for r in sub),
                })
            print(split, edge, "G_C", round(point["phi_c"], 2), "G_Y", round(point["phi_y"], 2), flush=True)

    def dump(name, rows_):
        with (RES / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
            w.writeheader()
            w.writerows(rows_)
    dump("factorial_bootstrap.csv", boot_rows)
    dump("factorial_class_conditional.csv", cond_rows)
    dump("factorial_auc.csv", auc_rows)
    dump("pairwise_margin_transitions.csv", trans_rows)
    dump("edge_channel_shapley_bootstrap.csv", edge_boot)
    # Margin distribution summaries, not every pair.
    dist = []
    for split in by_split:
        sub = by_split[split]
        y = np.array([int(r["label"]) for r in sub])
        src = [r for r in rows if r["split"] == split]
        for arm, col in (("444", "z_444"), ("422", "z_422"), ("440", "z_440"), ("420", "z_420")):
            zmap[arm] = np.array([float(r[col]) for r in src])
        real, fake = np.where(y == 0)[0], np.where(y == 1)[0]
        for arm in zmap:
            m = (zmap[arm][fake][:, None] - zmap[arm][real][None, :]).ravel()
            dist.append({
                "split": split, "arm": arm, "n_pairs": m.size,
                "mean": float(m.mean()), "median": float(np.median(m)),
                "q10": float(np.percentile(m, 10)), "q90": float(np.percentile(m, 90)),
                "frac_positive": float(np.mean(m > 0)),
                "frac_negative": float(np.mean(m < 0)),
                "frac_abs_gt_20": float(np.mean(np.abs(m) > 20)),
                "frac_abs_lt_5": float(np.mean(np.abs(m) < 5)),
            })
    dump("pairwise_margin_distributions.csv", dist)

    # Figures from AttGAN.
    att = by_split["attgan_test"]
    labs = np.array([int(r["label"]) for r in att])
    scatter.labs = labs
    scatter(FIG / "horizontal_vertical_effects.png",
            [float(r["Phi_H"]) for r in att], [float(r["Phi_V"]) for r in att],
            f"AttGAN test n={len(att)}  blue real, red fake", "Phi_H horizontal", "Phi_V vertical")
    scatter(FIG / "horizontal_vertical_interaction.png",
            [float(r["Phi_H"]) + float(r["Phi_V"]) for r in att], [float(r["I_HV"]) for r in att],
            f"AttGAN test n={len(att)} interaction vs total is not the same axis",
            "Phi_H + Phi_V = z420 - z444", "I_HV interaction")
    # Edge chroma gaps as text bars using bootstrap point estimates.
    att_edges = [r for r in edge_boot if r["split"] == "attgan_test" and r["quantity"] == "phi_c"]
    im = Image.new("RGB", (780, 420), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 12), "AttGAN chroma class gap by edge. Negative means fakes drop more than reals.", fill="black")
    for i, r in enumerate(att_edges):
        val = float(r["estimate"])
        x = 40 + i * 180
        y0 = 200
        y1 = int(200 - val)
        dr.line([(x + 40, 200), (x + 40, y1)], fill=(160, 50, 50), width=18)
        dr.text((x, 40), r["edge"], fill="black")
        dr.text((x, 340), f"{val:.1f}", fill="black")
        dr.text((x, 360), f"[{float(r['lo']):.1f}, {float(r['hi']):.1f}]", fill="black")
    im.save(FIG / "edge_channel_shapley.png")
    im.save(FIG / "horizontal_vertical_channel_shapley.png")
    order = ["attgan_test", "BEGAN_sub150", "SNGAN_sub150", "STGAN_sub150"]
    groups = []
    for split in order:
        vals = []
        for q in ("auc_444", "auc_422", "auc_440", "auc_420"):
            vals.append(float(next(r["estimate"] for r in auc_rows if r["split"] == split and r["quantity"] == q)))
        groups.append((split.replace("_sub150", "").replace("_test", ""), vals))
    bars(FIG / "factorial_auc_comparison.png", "Raw-logit AUC. AttGAN is the full test. Others are 150 per class.", groups, ["4:4:4", "4:2:2", "4:4:0", "4:2:0"])
    bars(FIG / "generator_factorial_comparison.png", "Same four AUCs, one group per generator.", groups, ["4:4:4", "4:2:2", "4:4:0", "4:2:0"])
    # Transition heatmap-like text figure for AttGAN.
    im = Image.new("RGB", (900, 360), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, 8), "AttGAN pairwise transitions. Share of real x fake pairs.", fill="black")
    att_t = [r for r in trans_rows if r["split"] == "attgan_test" and r["quantity"] in ("stay_correct", "correct_to_wrong", "wrong_to_correct")]
    comps = []
    for r in att_t:
        if r["comparison"] not in comps:
            comps.append(r["comparison"])
    dr.text((180, 36), "stay correct", fill="black")
    dr.text((360, 36), "correct to wrong", fill="black")
    dr.text((560, 36), "wrong to correct", fill="black")
    for i, comp in enumerate(comps):
        dr.text((12, 70 + i * 40), comp, fill="black")
        for j, q in enumerate(("stay_correct", "correct_to_wrong", "wrong_to_correct")):
            rec = next(r for r in att_t if r["comparison"] == comp and r["quantity"] == q)
            dr.text((180 + j * 190, 70 + i * 40), f"{float(rec['estimate']):.3f}", fill="black")
    im.save(FIG / "pairwise_ranking_transitions.png")
    im.save(FIG / "pairwise_ranking_transition_matrix.png")
    man = RES / "jpeg_factorial_manifest.csv"
    if man.exists():
        dest = EXP / "manifests"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(man, dest / "jpeg_factorial_manifest.csv")
    print("wrote summaries", flush=True)


if __name__ == "__main__":
    main()
