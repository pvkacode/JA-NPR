"""Load JA-NPR exactly as shipped and read q the way qvec_of does. No behavior changes."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

ROOT = Path(__file__).resolve().parents[2]
JA_SRC = Path("/tmp/janpr/JA-NPR/npr_t3_colab.py")
WEIGHTS = Path("/tmp/janpr/JA-NPR/weights")
NPR_ROOT = ROOT / "NPR-DeepfakeDetection"
NPR_CKPT = NPR_ROOT / "NPR.pth"
FACT = ROOT / "experiments" / "npr_chroma_factorial"
KNOWN_DQT = "28d506d794713d12460be674da01ebd14fe2262038e902c081a6946578ac98d3"
CKPT_SHA = "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53"
ARMS = ("444", "422", "440", "420")
# Pillow's get_sampling map has no 4:4:0 entry, so code -1 is the library result.
EXPECT_PIL_SAMPLING = {"444": 0, "422": 1, "440": -1, "420": 2}
EXPECT_Q3 = {"444": 0.0, "422": 0.0, "440": 0.0, "420": 1.0}

sys.path.insert(0, str(FACT))
sys.path.insert(0, str(ROOT / "scripts"))
from jpeg_codec import EXPECTED_SOF, decode, dqt_hash, parse_markers, sof_tuple  # noqa: E402
from npr_eval_lib import NORM, load_npr  # noqa: E402


def load_ja_module():
    spec = importlib.util.spec_from_file_location("ja_npr_t3", JA_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q_of_file(path, mod):
    im = Image.open(path)
    sampling = JpegImagePlugin.get_sampling(im)
    q = mod.qvec_of(im)
    return sampling, q


def check_jpeg(path, arm, mod):
    blob = Path(path).read_bytes()
    parsed = parse_markers(blob)
    sof = sof_tuple(parsed)
    if sof != EXPECTED_SOF[arm]:
        raise SystemExit(f"SOF {arm} {sof} on {path}")
    digest = dqt_hash(parsed)
    if digest != KNOWN_DQT:
        raise SystemExit(f"DQT changed on {path}")
    sampling, q = q_of_file(path, mod)
    if sampling != EXPECT_PIL_SAMPLING[arm]:
        raise SystemExit(f"Pillow sampling {sampling} on {arm} {path}")
    if abs(q[3] - EXPECT_Q3[arm]) > 1e-8:
        raise SystemExit(f"q3 {q[3]} on {arm} {path}")
    return q, decode(blob).convert("RGB")


def load_janpr(mod, weight_path, dev):
    class Args:
        npr = str(NPR_ROOT)
        ckpt = str(NPR_CKPT)
    net = mod.make_net("janpr_noadv", Args(), dev)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state, strict=True)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def x255_of(image, dev):
    arr = np.asarray(image, np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).to(dev)


def npr_input(image, dev):
    t = torch.from_numpy(np.asarray(image, np.float32) / 255.0).permute(2, 0, 1)
    return NORM(t).to(dev)


EPS = 1e-8
JA_STAGES = (
    "ycc", "y_npr", "chroma_pooled", "chroma_npr_up", "residual", "residual_scaled",
    "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "pre_fc",
)


def forward_ja(mod, net, x255, q):
    """Default path: chroma_half=True. bn1 is cloned before the inplace ReLU."""
    domain = int(q[:, 0].mean().item() > 0.5)
    for m in net.modules():
        if isinstance(m, mod.QTNorm):
            m.domain = domain
            m.q = q
    ycc = mod.rgb2ycc(x255)
    y_npr = mod.nn_residual(ycc[:, 0:1])
    chroma_pooled = torch.nn.functional.avg_pool2d(ycc[:, 1:3], 2)
    chroma_npr_up = mod.nn_residual(chroma_pooled).repeat_interleave(2, -1).repeat_interleave(2, -2)
    residual = torch.cat([y_npr, chroma_npr_up], 1) / (255.0 * mod.YC_SCALE)
    scaled = residual * (2.0 / 3.0)
    h = net.base.conv1(scaled)
    conv1 = h
    h = net.base.bn1(h)
    bn1 = h.clone()
    g, b = net.base.bn1.film(q).chunk(2, 1)
    h = net.base.relu(h)
    relu = h
    h = net.base.maxpool(h)
    pool = h
    h = net.base.layer1(h)
    layer1 = h
    h = net.base.layer2(h)
    layer2 = h
    pre = torch.flatten(net.base.avgpool(h), 1)
    logit = net.base.fc1(pre).reshape(-1)
    official, official_pre = net(x255, q, chroma_half=True)
    err = (logit - official.reshape(-1)).abs().max().item()
    if err > 1e-4:
        raise SystemExit(f"instrumented JA-NPR forward differs by {err}")
    if (pre - official_pre).abs().max().item() > 1e-4:
        raise SystemExit("instrumented pooled vector differs from net.forward")
    stages = {
        "ycc": ycc, "y_npr": y_npr, "chroma_pooled": chroma_pooled,
        "chroma_npr_up": chroma_npr_up, "residual": residual, "residual_scaled": scaled,
        "conv1": conv1, "bn1": bn1, "relu": relu, "maxpool": pool,
        "layer1": layer1, "layer2": layer2, "pre_fc": pre,
    }
    film = {"gamma_abs": g.abs().mean(1), "beta_abs": b.abs().mean(1)}
    return stages, logit, film
