"""libjpeg-turbo CLI encode/decode and JPEG marker checks.

cjpeg/djpeg from libjpeg-turbo 3.1.3. No PyTurboJPEG. Pillow is not used to
encode the factorial arms, because it has no 4:4:0 switch.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

CJPE = "cjpeg"
DJPEG = "djpeg"

# Component order for a YCbCr file: Y, Cb, Cr.
SAMPLES = {
    "444": "1x1,1x1,1x1",
    "422": "2x1,1x1,1x1",
    "440": "1x2,1x1,1x1",
    "420": "2x2,1x1,1x1",
}
EXPECTED_SOF = {
    "444": ((1, 1, 1), (2, 1, 1), (3, 1, 1)),
    "422": ((1, 2, 1), (2, 1, 1), (3, 1, 1)),
    "440": ((1, 1, 2), (2, 1, 1), (3, 1, 1)),
    "420": ((1, 2, 2), (2, 1, 1), (3, 1, 1)),
}


def version():
    proc = subprocess.run([CJPE, "-version"], capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    return text.strip()


def ppm_bytes(im):
    im = im.convert("RGB")
    # Pillow's PPM writer. cjpeg reads binary PPM.
    import io
    buf = io.BytesIO()
    im.save(buf, format="PPM")
    return buf.getvalue()


def encode(im, arm, quality=95):
    if arm not in SAMPLES:
        raise ValueError(arm)
    proc = subprocess.run(
        [CJPE, "-quality", str(quality), "-dct", "int", "-sample", SAMPLES[arm]],
        input=ppm_bytes(im),
        capture_output=True,
        check=True,
    )
    return proc.stdout


def decode(jpeg):
    proc = subprocess.run(
        [DJPEG, "-dct", "int", "-pnm"],
        input=jpeg,
        capture_output=True,
        check=True,
    )
    im = Image.open(io_bytes(proc.stdout)).convert("RGB")
    return im


def io_bytes(data):
    import io
    return io.BytesIO(data)


def parse_markers(data):
    """Return SOF components, DQT bytes keyed by table id, and the process marker."""
    sof = []
    dqt = {}
    process = None
    i = 0
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        if marker == 0x00 or (0xD0 <= marker <= 0xD7):
            i += 2
            continue
        length = int.from_bytes(data[i + 2:i + 4], "big")
        body = data[i + 4:i + 2 + length]
        if marker in (0xC0, 0xC1, 0xC2):
            process = marker
            n = body[5]
            comps = []
            for k in range(n):
                cid = body[6 + 3 * k]
                hv = body[7 + 3 * k]
                tq = body[8 + 3 * k]
                comps.append((int(cid), int(hv >> 4), int(hv & 0x0F), int(tq)))
            sof = comps
        elif marker == 0xDB:
            j = 0
            while j < len(body):
                info = body[j]
                prec = info >> 4
                tid = info & 0x0F
                ncoef = 64 * (2 if prec else 1)
                dqt[int(tid)] = bytes(body[j:j + 1 + ncoef])
                j += 1 + ncoef
        if marker == 0xDA:
            break
        i += 2 + length
    return {"sof": sof, "dqt": dqt, "process": process}


def dqt_hash(parsed):
    blob = b"".join(parsed["dqt"][k] for k in sorted(parsed["dqt"]))
    return hashlib.sha256(blob).hexdigest()


def sof_tuple(parsed):
    return tuple((cid, h, v) for cid, h, v, _tq in parsed["sof"])


def qtable_ids(parsed):
    return tuple((cid, tq) for cid, _h, _v, tq in parsed["sof"])
