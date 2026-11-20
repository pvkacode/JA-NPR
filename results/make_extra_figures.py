"""Draws the ablation, seven-generator and gain-decomposition figures of the paper.
Numbers are copied from results/logs (t5 runs) and results/RESULTS.md. Run: python results/make_extra_figures.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(out, exist_ok=True)
arms = ['NPR+JPEG FT', 'fast-BN', 'CA-NPR+dual BN', 'QT-Norm RGB', 'JA-NPR const-q', 'JA-NPR']
q95 = [0.794, 0.839, 0.801, 0.906, 0.938, 0.940]; q70 = [0.773, 0.783, 0.692, 0.885, 0.855, 0.849]
x = np.arange(len(arms)); w = 0.38
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(x - w / 2, q95, w, label='Q95', color='#4C72B0'); ax.bar(x + w / 2, q70, w, label='Q70', color='#DD8452')
ax.set_xticks(x); ax.set_xticklabels(arms, rotation=30, ha='right', fontsize=6); ax.set_ylim(0.6, 1.0)
ax.set_ylabel('AUC (4:2:0)', fontsize=7); ax.tick_params(axis='y', labelsize=6); ax.legend(fontsize=6, loc='upper left')
ax.axhline(0.794, color='gray', lw=0.6, ls=':')
plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_ablation.pdf'))
gens = ['STGAN', 'BEGAN', 'CramerGAN', 'InfoMaxGAN', 'MMDGAN', 'RelGAN', 'S3GAN']
p = [0.519, 0.281, 0.609, 0.629, 0.560, 0.397, 0.532]; c = [0.642, 0.546, 0.631, 0.637, 0.648, 0.753, 0.540]
j = [0.802, 0.619, 0.647, 0.520, 0.654, 0.762, 0.487]
x = np.arange(len(gens)); w = 0.27
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.bar(x - w, p, w, label='NPR + JPEG FT', color='#4C72B0'); ax.bar(x, c, w, label='JA-NPR const-q', color='#55A868')
ax.bar(x + w, j, w, label='JA-NPR', color='#C44E52')
ax.axhline(0.5, color='gray', lw=0.6); ax.set_xticks(x); ax.set_xticklabels(gens, rotation=30, ha='right', fontsize=6)
ax.set_ylabel('AUC at Q95 4:2:0', fontsize=7); ax.tick_params(axis='y', labelsize=6); ax.set_ylim(0.2, 0.9)
ax.legend(fontsize=5.5, loc='upper right')
plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_seven.pdf'))
fig, ax = plt.subplots(figsize=(3.4, 1.9))
labels = ['Released NPR', '+ JPEG fine-tune\n(data)', '+ JA-NPR\n(model)']; vals = [0.233, 0.799, 0.922]
ax.bar(labels, vals, color=['#999999', '#4C72B0', '#C44E52'], width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=7)
ax.text(1, 0.45, '+0.566', ha='center', color='white', fontsize=7); ax.text(2, 0.45, '+0.123', ha='center', color='white', fontsize=7)
ax.axhline(0.5, color='gray', lw=0.6); ax.set_ylim(0, 1.05); ax.set_ylabel('AUC at Q95 4:2:0', fontsize=7); ax.tick_params(labelsize=6.5)
plt.tight_layout(); plt.savefig(os.path.join(out, 'fig_decomp.pdf'))
print('figures written to', out)
