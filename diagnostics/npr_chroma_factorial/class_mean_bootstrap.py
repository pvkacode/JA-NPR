#!/usr/bin/env python3
"""Face-cluster intervals for class means. Separate from the class-gap bootstrap."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
KEYS = ["H0", "H1", "V0", "V1", "Phi_H", "Phi_V", "I_HV", "total"]


def main():
    rows = list(csv.DictReader((RES / "factorial_logit_contrasts.csv").open()))
    out_rows = []
    rng_seed = 0
    for split in ("attgan_test", "BEGAN_sub150", "SNGAN_sub150", "STGAN_sub150"):
        sub = [r for r in rows if r["split"] == split]
        groups = defaultdict(list)
        for i, r in enumerate(sub):
            groups[(int(r["label"]), r["face_id"])].append(i)
        pos = [k for k in groups if k[0] == 1]
        neg = [k for k in groups if k[0] == 0]
        image_level = len(pos) <= 1
        arr = {k: np.array([float(r[k]) for r in sub]) for k in KEYS}
        y = np.array([int(r["label"]) for r in sub])
        rng = np.random.default_rng(rng_seed)
        store = {(lab, k): [] for lab in (0, 1) for k in KEYS}
        for _ in range(2000):
            if image_level:
                ri = rng.choice(np.where(y == 0)[0], int((y == 0).sum()), replace=True)
                fi = rng.choice(np.where(y == 1)[0], int((y == 1).sum()), replace=True)
            else:
                ri, fi = [], []
                for j in rng.choice(len(neg), len(neg), replace=True):
                    ri.extend(groups[neg[j]])
                for j in rng.choice(len(pos), len(pos), replace=True):
                    fi.extend(groups[pos[j]])
                ri, fi = np.array(ri), np.array(fi)
            for lab, idx in ((0, ri), (1, fi)):
                for k in KEYS:
                    store[(lab, k)].append(float(arr[k][idx].mean()))
        for lab, name in ((0, "real"), (1, "fake")):
            for k in KEYS:
                point = float(arr[k][y == lab].mean())
                lo, hi = np.percentile(store[(lab, k)], [2.5, 97.5])
                out_rows.append({
                    "split": split, "class": name, "quantity": k,
                    "estimate": point, "lo": float(lo), "hi": float(hi),
                    "scheme": "image" if image_level else "face",
                    "n_boot": 2000, "seed": rng_seed,
                    "n_fake_clusters": len(pos), "n_real_clusters": len(neg),
                })
    pred = [r for r in csv.DictReader((RES / "factorial_predictions.csv").open()) if r["split"] == "attgan_test"]
    qrows = []
    for name, a, b in (
        ("444_to_422", "444", "422"), ("444_to_440", "444", "440"),
        ("422_to_420", "422", "420"), ("440_to_420", "440", "420"),
        ("444_to_420", "444", "420"),
    ):
        def col(arm, lab, aa=a):
            return np.array([float(r[f"z_{arm}"]) for r in pred if int(r["label"]) == lab])
        d = (col(b, 1)[:, None] - col(b, 0)[None, :]) - (col(a, 1)[:, None] - col(a, 0)[None, :])
        d = d.ravel()
        q10, q50, q90 = np.percentile(d, [10, 50, 90])
        qrows.append({
            "split": "attgan_test", "comparison": name, "n_pairs": int(d.size),
            "mean": float(d.mean()), "q10": float(q10), "q50": float(q50), "q90": float(q90),
            "frac_abs_gt_20": float(np.mean(np.abs(d) > 20)),
            "frac_abs_lt_5": float(np.mean(np.abs(d) < 5)),
        })
    def dump(name, rows):
        with (RES / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    dump("factorial_class_mean_bootstrap.csv", out_rows)
    dump("pairwise_margin_change_quantiles.csv", qrows)
    print("wrote class means and margin quantiles", flush=True)


if __name__ == "__main__":
    main()
