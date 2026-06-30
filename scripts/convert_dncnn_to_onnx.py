#!/usr/bin/env python3
"""Convert KAIR's DnCNN (Zhang et al., TIP 2017) blind color-denoising weights to ONNX.

Produces models/denoise.onnx -- the image-to-image denoiser the noise-reduction slider
uses when available (MLDenoiser feeds an image in and gets a denoised image back, same
contract as the Auto WB image-to-image model).

torch is only needed for this one-time export; the app runs the ONNX with onnxruntime alone.

    pip install torch onnx onnxscript          # build-time only
    python scripts/convert_dncnn_to_onnx.py

Source weights/arch are fetched from the official repo (cszn/KAIR).
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

ARCH_URL = "https://raw.githubusercontent.com/cszn/KAIR/master/models/network_dncnn.py"
BLOCK_URL = "https://raw.githubusercontent.com/cszn/KAIR/master/models/basicblock.py"
WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
OUT = Path(__file__).resolve().parents[1] / "models" / "denoise.onnx"


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> int:
    try:
        import torch
        import onnx
    except Exception as exc:
        print(f"Need torch + onnx for export: {exc}")
        print("  pip install torch onnx onnxscript")
        return 1

    work = Path(tempfile.mkdtemp(prefix="dncnn_"))
    models_pkg = work / "models"
    models_pkg.mkdir(parents=True, exist_ok=True)
    (models_pkg / "__init__.py").write_text("")
    print("Fetching architecture + weights...")
    _fetch(ARCH_URL, models_pkg / "network_dncnn.py")
    _fetch(BLOCK_URL, models_pkg / "basicblock.py")
    _fetch(WEIGHTS_URL, work / "dncnn_color_blind.pth")

    sys.path.insert(0, str(work))
    from models.network_dncnn import DnCNN  # type: ignore

    # dncnn_color_blind: 3-channel, 20 conv layers, BN already merged into conv weights
    # by KAIR's release process -- hence act_mode='R' (Conv+ReLU only, no BatchNorm layer).
    net = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R")
    net.load_state_dict(torch.load(str(work / "dncnn_color_blind.pth"), map_location="cpu"), strict=True)
    net.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, 256, 256)
    print("Exporting ONNX...")
    torch.onnx.export(
        net, dummy, str(OUT),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "h", 3: "w"}},
        opset_version=13,
    )
    # Newer exporters externalize weights -- consolidate into one self-contained file.
    model = onnx.load(str(OUT))
    onnx.save(model, str(OUT), save_as_external_data=False)
    sidecar = OUT.with_suffix(OUT.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    print(f"Wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print("Validate with: python scripts/download_denoise_model.py --skip-download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
