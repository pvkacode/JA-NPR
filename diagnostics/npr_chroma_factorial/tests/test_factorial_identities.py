"""Identities that do not need the AttGAN test labels."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from jpeg_codec import EXPECTED_SOF, decode, dqt_hash, encode, parse_markers, qtable_ids, sof_tuple  # noqa: E402


def effects(z00, z10, z01, z11):
    h0, h1 = z10 - z00, z11 - z01
    v0, v1 = z01 - z00, z11 - z10
    phi_h = 0.5 * (h0 + h1)
    phi_v = 0.5 * (v0 + v1)
    interaction = z11 - z10 - z01 + z00
    return h0, h1, v0, v1, phi_h, phi_v, interaction


def test_factorial_identities():
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = rng.normal(size=4)
        h0, h1, v0, v1, phi_h, phi_v, inter = effects(*z)
        assert abs((phi_h + phi_v) - (z[3] - z[0])) < 1e-9
        assert abs(inter - (h1 - h0)) < 1e-9
        assert abs(inter - (v1 - v0)) < 1e-9


def test_channel_shapley_identity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        z00, z10, z01, z11 = rng.normal(size=4)
        phi_y = 0.5 * ((z10 - z00) + (z11 - z01))
        phi_c = 0.5 * ((z01 - z00) + (z11 - z10))
        assert abs((phi_y + phi_c) - (z11 - z00)) < 1e-9


def test_pairwise_auc_matches_ranking():
    # Three reals, three fakes, no ties.
    real = np.array([0.0, 1.0, 2.0])
    fake = np.array([0.5, 1.5, 3.0])
    wins = 0.0
    for f in fake:
        for r in real:
            wins += 1.0 if f > r else (0.5 if f == r else 0.0)
    auc = wins / (len(real) * len(fake))
    # 0.5 beats one real, 1.5 beats two, 3.0 beats three: 6/9.
    assert abs(auc - (6 / 9)) < 1e-12


def test_encoder_writes_four_sampling_modes_and_one_dqt():
    rng = np.random.default_rng(2)
    im = Image.fromarray(rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8), "RGB")
    hashes = set()
    slots = set()
    for arm, expected in EXPECTED_SOF.items():
        blob = encode(im, arm, quality=95)
        parsed = parse_markers(blob)
        assert sof_tuple(parsed) == expected, (arm, sof_tuple(parsed))
        assert parsed["process"] == 0xC0
        hashes.add(dqt_hash(parsed))
        slots.add(qtable_ids(parsed))
        back = np.asarray(decode(blob))
        assert back.shape == (32, 48, 3)
    assert len(hashes) == 1, hashes
    assert len(slots) == 1


def test_gauss_legendre_integrates_a_cubic():
    import sys
    sys.path.insert(0, str(EXP.parents[0] / "npr_chroma_sensitivity"))
    from chroma_ops import gl_interval
    t, w = gl_interval(5)
    assert abs(float(np.dot(t ** 3, w)) - 0.25) < 1e-12


def test_frozen_checkpoint_checksum():
    import hashlib
    path = EXP.parents[1] / "NPR-DeepfakeDetection" / "NPR.pth"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53"


def test_saved_factorial_identities_and_pairing():
    import csv
    res = EXP / "results"
    contrasts = res / "factorial_logit_contrasts.csv"
    manifest = res / "jpeg_factorial_manifest.csv"
    if not contrasts.exists() or not manifest.exists():
        return
    rows = list(csv.DictReader(contrasts.open()))
    err = max(abs(float(r["identity_err"])) for r in rows)
    inter = max(abs(float(r["interaction_identity_err"])) for r in rows)
    assert err < 1e-6 and inter < 1e-6
    arms = {}
    hashes = set()
    for r in csv.DictReader(manifest.open()):
        arms.setdefault(r["path"], set()).add(r["arm"])
        hashes.add(r["dqt_hash"])
    assert all(v == {"444", "422", "440", "420"} for v in arms.values())
    assert hashes == {"28d506d794713d12460be674da01ebd14fe2262038e902c081a6946578ac98d3"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
