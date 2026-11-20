#!/usr/bin/env python3
"""Summarize saved path integrals. No model."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RES = Path(__file__).resolve().parent / "results"
FIG = Path(__file__).resolve().parent / "figures"


def r2(y, yhat):
    y = np.asarray(y, np.float64)
    yhat = np.asarray(yhat, np.float64)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def fit_stats(y, yhat):
    y = np.asarray(y, np.float64)
    yhat = np.asarray(yhat, np.float64)
    err = yhat - y
    if y.std() == 0 or yhat.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y, yhat)[0, 1])
    return {
        "n": int(y.size),
        "r2": r2(y, yhat),
        "corr": corr,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "mean_signed_err": float(np.mean(err)),
    }


def main():
    rows = list(csv.DictReader((RES / "factorial_gradient_paths.csv").open()))
    if not rows:
        raise SystemExit("no path rows")
    detail = []
    summary = []
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["edge"], r["label"])].append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["split"], r["edge"])].append(r)
    for (split, edge), sub in groups.items():
        for rule, pred in (("tangent_gdir0", "gdir_0"), ("gl5", "gl5"), ("gl9", "gl9"), ("gl17", "gl17")):
            for lab, name in ((None, "all"), ("0", "real"), ("1", "fake")):
                part = sub if lab is None else [r for r in sub if r["label"] == lab]
                if len(part) < 3:
                    continue
                y = np.array([float(r["dz"]) for r in part])
                yhat = np.array([float(r[pred]) for r in part])
                stats = fit_stats(y, yhat)
                summary.append({"split": split, "edge": edge, "class": name, "predictor": rule, **stats})
        for lab, name in (("0", "real"), ("1", "fake")):
            part = [r for r in sub if r["label"] == lab]
            rec = {
                "split": split, "edge": edge, "class": name, "n": len(part),
                "mean_dz": float(np.mean([float(r["dz"]) for r in part])),
                "mean_gdir_0": float(np.mean([float(r["gdir_0"]) for r in part])),
                "mean_gdir_25": float(np.mean([float(r["gdir_25"]) for r in part])),
                "mean_gdir_50": float(np.mean([float(r["gdir_50"]) for r in part])),
                "mean_gdir_75": float(np.mean([float(r["gdir_75"]) for r in part])),
                "mean_gdir_1": float(np.mean([float(r["gdir_1"]) for r in part])),
                "mean_cos_0": float(np.mean([float(r["cos_0"]) for r in part])),
                "mean_cos_1": float(np.mean([float(r["cos_1"]) for r in part])),
                "mean_gnorm_0": float(np.mean([float(r["gnorm_0"]) for r in part])),
                "mean_gnorm_1": float(np.mean([float(r["gnorm_1"]) for r in part])),
                "frac_quarter_sign_change": float(np.mean([int(r["sign_changes_quarter"]) > 0 for r in part])),
                "mean_quarter_sign_changes": float(np.mean([int(r["sign_changes_quarter"]) for r in part])),
                "mean_rel": float(np.mean([float(r["rel"]) for r in part])),
            }
            detail.append(rec)
            print(
                f"{split} {edge} {name} dz {rec['mean_dz']:.1f} "
                f"g0 {rec['mean_gdir_0']:.1f} g1 {rec['mean_gdir_1']:.1f} "
                f"sign {rec['frac_quarter_sign_change']:.2f}",
                flush=True,
            )
    def dump(name, items):
        with (RES / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            w.writeheader()
            w.writerows(items)
    dump("factorial_gradient_sensitivity.csv", summary)
    dump("factorial_path_integrals.csv", detail)
    # AttGAN mean profiles.
    im = Image.new("RGB", (980, 640), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 8), "AttGAN mean directional derivative along each edge. Blue real, red fake.", fill="black")
    edges = ["444_to_422", "444_to_440", "422_to_420", "440_to_420"]
    ts = [0, 0.25, 0.5, 0.75, 1]
    for i, edge in enumerate(edges):
        x0, y0, x1, y1 = 40 + (i % 2) * 470, 40 + (i // 2) * 280, 470 + (i % 2) * 470, 280 + (i // 2) * 280
        dr.rectangle([x0, y0, x1, y1], outline="black")
        dr.text((x0 + 8, y0 + 6), edge, fill="black")
        series = []
        for lab, col in (("0", (31, 78, 121)), ("1", (176, 0, 32))):
            rec = next(r for r in detail if r["split"] == "attgan_test" and r["edge"] == edge and r["class"] == ("real" if lab == "0" else "fake"))
            series.append(([rec[f"mean_gdir_{t}"] for t in ("0", "25", "50", "75", "1")], col))
        vals = [v for s, _ in series for v in s]
        lo, hi = min(vals), max(vals)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        for s, col in series:
            pts = []
            for t, v in zip(ts, s):
                px = int(x0 + 30 + t * (x1 - x0 - 50))
                py = int(y1 - 30 - (v - lo) / (hi - lo) * (y1 - y0 - 60))
                pts.append((px, py))
            dr.line(pts, fill=col, width=2)
        dr.text((x0 + 8, y1 - 22), f"[{lo:.0f}, {hi:.0f}]", fill="black")
    im.save(FIG / "factorial_path_profiles.png")
    im = Image.new("RGB", (860, 420), "white")
    dr = ImageDraw.Draw(im)
    dr.text((16, 10), "AttGAN fraction of images whose directional derivative changes sign on 0, 0.25, 0.5, 0.75, 1.", fill="black")
    for i, edge in enumerate(edges):
        for j, (cls, col) in enumerate((("real", (31, 78, 121)), ("fake", (176, 0, 32)))):
            rec = next(r for r in detail if r["split"] == "attgan_test" and r["edge"] == edge and r["class"] == cls)
            h = int(rec["frac_quarter_sign_change"] * 250)
            x = 80 + i * 190 + j * 40
            dr.rectangle([x, 320 - h, x + 28, 320], fill=col)
            dr.text((x - 4, 330), f"{rec['frac_quarter_sign_change']:.2f}", fill="black")
        dr.text((80 + i * 190, 360), edge, fill="black")
    dr.text((16, 390), "Blue real, red fake. A sign change means the local slope reverses along that edge.", fill="black")
    im.save(FIG / "factorial_gradient_sign_changes.png")
    print("wrote path summaries", flush=True)


if __name__ == "__main__":
    main()
