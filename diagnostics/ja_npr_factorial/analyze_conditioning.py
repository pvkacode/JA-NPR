#!/usr/bin/env python3
"""What the JA-NPR header actually changes across the four arms."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from runtime import WEIGHTS, load_ja_module, load_janpr  # noqa: E402
from npr_eval_lib import device  # noqa: E402

RES = EXP / "results"
FIG = EXP / "figures"
ARMS = ("444", "422", "440", "420")


def main():
    rows = [r for r in csv.DictReader((RES / "logits.csv").open()) if r["model"] == "ja_attgan_s0" and r["split"] == "attgan_test"]
    keys = []
    for r in rows:
        keys.append((r["q_luma"], r["q_chroma"], r["q3_444"], r["q3_422"], r["q3_440"], r["q3_420"]))
    counts = Counter(keys)
    logged = {}
    for arm in ARMS:
        g = sorted({round(float(r[f"film_gamma_{arm}"]), 8) for r in rows})
        b = sorted({round(float(r[f"film_beta_{arm}"]), 8) for r in rows})
        logged[arm] = {"n_unique_gamma_abs": len(g), "n_unique_beta_abs": len(b), "gamma_abs": g[:3], "beta_abs": b[:3]}
    sample = rows[0]
    q_by_arm = {
        arm: [1.0, float(sample["q_luma"]), float(sample["q_chroma"]), float(sample[f"q3_{arm}"])]
        for arm in ARMS
    }
    mod = load_ja_module()
    net = load_janpr(mod, WEIGHTS / "janpr_AttGAN_seed0.pth", device())
    layers = []
    with torch.no_grad():
        for name, layer in net.named_modules():
            if isinstance(layer, mod.QTNorm) and layer.film is not None:
                outs = {}
                for arm, vec in q_by_arm.items():
                    q = torch.tensor([vec], dtype=torch.float32, device=next(layer.film.parameters()).device)
                    gamma, beta = layer.film(q).chunk(2, 1)
                    outs[arm] = (gamma.reshape(-1).cpu(), beta.reshape(-1).cpu())
                same_422 = torch.max((outs["422"][0] - outs["444"][0]).abs()).item() + torch.max((outs["422"][1] - outs["444"][1]).abs()).item()
                same_440 = torch.max((outs["440"][0] - outs["444"][0]).abs()).item() + torch.max((outs["440"][1] - outs["444"][1]).abs()).item()
                d420_g = (outs["420"][0] - outs["444"][0]).norm().item()
                d420_b = (outs["420"][1] - outs["444"][1]).norm().item()
                layers.append({
                    "module": name,
                    "channels": int(outs["444"][0].numel()),
                    "max_abs_422_minus_444": same_422,
                    "max_abs_440_minus_444": same_440,
                    "l2_gamma_420_minus_444": d420_g,
                    "l2_beta_420_minus_444": d420_b,
                    "mean_abs_gamma": {arm: float(outs[arm][0].abs().mean()) for arm in ARMS},
                    "mean_abs_beta": {arm: float(outs[arm][1].abs().mean()) for arm in ARMS},
                })
    out = {
        "n_attgan_images": len(rows),
        "n_distinct_q_patterns": len(counts),
        "q_by_arm": q_by_arm,
        "q_patterns": [{"luma_chroma_q3s": list(k), "count": v} for k, v in counts.most_common(5)],
        "bn1_logged_abs_mean": logged,
        "film_layers": layers,
        "interpretation": (
            "Every arm has a quantization table, so QT-Norm stays in the JPEG domain. "
            "The luma and chroma table features match because the DQT is the same. "
            "q[3] is 1 only when Pillow get_sampling returns 2, which is 4:2:0. "
            "4:4:4, 4:2:2, and 4:4:0 therefore produce the same q and the same FiLM "
            "gamma and beta. The header does not distinguish horizontal-only from vertical-only subsampling."
        ),
    }
    (RES / "conditioning.json").write_text(json.dumps(out, indent=2))
    FIG.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (980, 520), "white")
    dr = ImageDraw.Draw(im)
    dr.text((12, 8), "JA-NPR q vector on the four Q95 arms", fill="black")
    dr.text((12, 28), f"AttGAN n={len(rows)}. One header pattern. 4:4:4, 4:2:2, and 4:4:0 share q[3]=0.", fill="black")
    labels = ["has table", "luma table", "chroma table", "4:2:0 bit"]
    colors = [(31, 78, 121), (70, 130, 90), (180, 140, 40), (176, 0, 32)]
    left, top, bottom = 80, 70, 400
    dr.rectangle([left, top, 940, bottom], outline="black")
    for i, arm in enumerate(ARMS):
        for j, val in enumerate(q_by_arm[arm]):
            x = left + 30 + i * 210 + j * 40
            h = int(val / 1.05 * (bottom - top - 20))
            dr.rectangle([x, bottom - h, x + 30, bottom], fill=colors[j])
        dr.text((left + 40 + i * 210, 410), arm, fill="black")
    dr.text((12, 460), "Bars within a group: " + ", ".join(labels) + ".", fill="black")
    dr.text((12, 485), "FiLM is an affine function of this vector, so three of the four arms get identical conditioning.", fill="black")
    im.save(FIG / "conditioning_q.png")
    print(json.dumps({k: out[k] for k in ("q_by_arm", "n_distinct_q_patterns")}, indent=2))
    print("film layers", len(layers), "max 422-444", max(r["max_abs_422_minus_444"] for r in layers),
          "max 440-444", max(r["max_abs_440_minus_444"] for r in layers))


if __name__ == "__main__":
    main()
