#!/usr/bin/env python3
"""Convert Deep White-Balance (Afifi & Brown, CVPR 2020) AWB weights to a single ONNX file.

Produces models/white_balance.onnx -- the image-to-image AWB model the "Auto AI" white-balance
button uses (WhiteBalanceEstimator derives WB gains from the model's input->output color shift).

torch is only needed for this one-time export; the app runs the ONNX with onnxruntime alone.

    pip install torch onnx onnxscript          # build-time only
    python scripts/convert_deepwb_to_onnx.py

Source weights/arch are fetched from the official repo (mahmoudnafifi/Deep_White_Balance).
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/mahmoudnafifi/Deep_White_Balance/master/PyTorch"
OUT = Path(__file__).resolve().parents[1] / "models" / "white_balance.onnx"


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

    work = Path(tempfile.mkdtemp(prefix="deepwb_"))
    arch = work / "arch"
    arch.mkdir(parents=True, exist_ok=True)
    (arch / "__init__.py").write_text("")  # avoid importing modules we don't fetch
    print("Fetching architecture + weights...")
    _fetch(f"{REPO}/arch/deep_wb_single_task.py", arch / "deep_wb_single_task.py")
    _fetch(f"{REPO}/arch/deep_wb_blocks.py", arch / "deep_wb_blocks.py")
    _fetch(f"{REPO}/models/net_awb.pth", work / "net_awb.pth")

    sys.path.insert(0, str(work))
    from arch import deep_wb_single_task  # type: ignore

    net = deep_wb_single_task.deepWBnet()
    net.load_state_dict(torch.load(str(work / "net_awb.pth"), map_location="cpu"))
    net.eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The U-Net has 4 downsamples, so spatial dims must be multiples of 16; export dynamic.
    dummy = torch.rand(1, 3, 512, 512)
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
    print("Validate with: python scripts/download_wb_model.py --skip-download")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
