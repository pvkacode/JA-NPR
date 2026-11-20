"""Builds every number, table and figure used in the paper directly from the CSV files in results/csv.
Run:  python results/make_tables.py   (needs pandas, matplotlib)"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "csv")
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)

single = pd.concat([pd.read_csv(os.path.join(CSV, f"t3_single_AttGAN_seed{s}.csv")) for s in (0, 1, 2)])
held = pd.read_csv(os.path.join(CSV, "t3_heldout_STGAN+BEGAN_seed0.csv"))
single["arm"] = single["arm"].replace({"janpr_noadv": "JA-NPR", "janpr": "JA-NPR+CAdv"})
held["arm"] = held["arm"].replace({"janpr_noadv": "JA-NPR"})

CONDS = ["orig", "Q95_420", "Q90_420", "Q70_420", "Q50_420", "Q30_420", "Q95_444", "Q75_444", "Q85_422"]
lines = []

# Table: AttGAN, 3 seeds, mean (min-max) for plain and JA-NPR; released (seed 0 only)
att = single[single.generator == "AttGAN"]
def cell(arm, cond, col="auc"):
    v = att[(att.arm == arm) & (att.condition == cond)][col]
    return v.mean(), v.min(), v.max(), len(v)
lines.append("== AttGAN AUC, mean over seeds (min, max, n) ==")
for arm in ["released", "plain", "ycdsbn", "JA-NPR+CAdv", "JA-NPR"]:
    row = []
    for c in CONDS:
        m, lo, hi, n = cell(arm, c)
        row.append(f"{c}:{m:.3f}({lo:.3f}-{hi:.3f},n={n})")
    lines.append(arm + "  " + "  ".join(row))
lines.append("== AttGAN balanced accuracy at 0.5, mean over seeds ==")
for arm in ["released", "plain", "ycdsbn", "JA-NPR+CAdv", "JA-NPR"]:
    lines.append(arm + "  " + "  ".join(f"{c}:{cell(arm, c, 'bal_acc_0.5')[0]:.3f}" for c in CONDS))
lines.append("== per-seed AUC plain vs JA-NPR ==")
for s in (0, 1, 2):
    d = att[att.seed == s]
    for c in CONDS:
        p = d[(d.arm == "plain") & (d.condition == c)].auc.values[0]
        j = d[(d.arm == "JA-NPR") & (d.condition == c)].auc.values[0]
        lines.append(f"seed{s} {c}: plain {p:.3f} JA-NPR {j:.3f} diff {j - p:+.3f}")
lines.append("== held-out (trained AttGAN+SNGAN) AUC ==")
for g in ["AttGAN", "SNGAN", "STGAN", "BEGAN"]:
    for arm in ["plain", "JA-NPR"]:
        d = held[(held.generator == g) & (held.arm == arm)]
        lines.append(f"{g} {arm}: " + "  ".join(f"{c}:{d[d.condition == c].auc.values[0]:.3f}" for c in CONDS))
lines.append("== single seed0, other generators (trained AttGAN only) Q95_420 AUC ==")
s0 = single[single.seed == 0]
for g in ["BEGAN", "SNGAN", "STGAN"]:
    lines.append(g + ": " + "  ".join(
        f"{a}:{s0[(s0.generator == g) & (s0.arm == a) & (s0.condition == 'Q95_420')].auc.values[0]:.3f}"
        for a in ["released", "plain", "ycdsbn", "JA-NPR+CAdv", "JA-NPR"]))
open(os.path.join(OUT, "numbers.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# Figure 1: AUC vs JPEG quality on AttGAN (4:2:0), mean and range over seeds
qs = [95, 90, 70, 50, 30]
plt.figure(figsize=(3.4, 2.5))
for arm, style, lab in [("released", "k:o", "Released NPR [1]"), ("plain", "b--s", "NPR + JPEG fine-tune"),
                        ("JA-NPR", "r-^", "JA-NPR (ours)")]:
    ms, los, his = zip(*[cell(arm, f"Q{q}_420")[:3] for q in qs])
    plt.plot(qs, ms, style, label=lab, markersize=4, linewidth=1.2)
    if arm != "released":
        plt.fill_between(qs, los, his, alpha=0.2, color=style[0])
plt.axhline(0.5, color="gray", linewidth=0.6)
plt.gca().invert_xaxis()
plt.xlabel("JPEG quality (4:2:0)")
plt.ylabel("AUC")
plt.ylim(0.15, 1.0)
plt.legend(fontsize=6.5, loc="center left", bbox_to_anchor=(0.0, 0.47))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_quality_curve.pdf"))

# Figure 2: held-out generators
fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.3), sharey=True)
for ax, g in zip(axes, ["STGAN", "BEGAN"]):
    for arm, style, lab in [("plain", "b--s", "NPR + JPEG fine-tune"), ("JA-NPR", "r-^", "JA-NPR (ours)")]:
        d = held[(held.generator == g) & (held.arm == arm)]
        ys = [d[d.condition == f"Q{q}_420"].auc.values[0] for q in qs]
        lo = [d[d.condition == f"Q{q}_420"].ci_lo.values[0] for q in qs]
        hi = [d[d.condition == f"Q{q}_420"].ci_hi.values[0] for q in qs]
        ax.plot(qs, ys, style, label=lab, markersize=4, linewidth=1.2)
        ax.fill_between(qs, lo, hi, alpha=0.2, color=style[0])
    ax.axhline(0.5, color="gray", linewidth=0.6)
    ax.invert_xaxis()
    ax.set_title(f"{g} (unseen)", fontsize=8)
    ax.set_xlabel("JPEG quality", fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].set_ylabel("AUC", fontsize=7)
axes[1].legend(fontsize=5.5, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_heldout.pdf"))
print("figures written to", OUT)
