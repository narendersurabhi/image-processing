#!/usr/bin/env python3
"""Download and validate the optional SAM ViT-B instance-segmentation models.

The app uses these for learned per-person masks and SAM-refined Face boundaries. Two ONNX files
are needed (the standard SAM split): a heavy image **encoder** (run once per image) and a
lightweight **decoder** (run per prompt). Both are fetched to models/ and validated -- each is
loaded in onnxruntime and run on a dummy input so a truncated/wrong download is caught here rather
than silently falling back at runtime.

Usage:
    python scripts/download_sam_model.py                       # default mirror
    python scripts/download_sam_model.py --url-encoder <URL> --url-decoder <URL>
    python scripts/download_sam_model.py --skip-download       # validate existing files only

The default URLs point at a commonly-mirrored SAM ViT-B ONNX export. Override with the --url-*
flags if you prefer a source you trust; the validation below works regardless of origin, as long
as the decoder follows the standard SAM ONNX signature (image_embeddings / point_coords /
point_labels / mask_input / has_mask_input / orig_im_size -> masks, iou_predictions).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENCODER_OUT = REPO_ROOT / "models" / "sam_vit_b_encoder.onnx"
DEFAULT_DECODER_OUT = REPO_ROOT / "models" / "sam_vit_b_decoder.onnx"

# A commonly-mirrored SAM ViT-B ONNX export (encoder + decoder), standard official decoder
# signature. Override with --url-* if you prefer a different trusted source.
DEFAULT_ENCODER_URL = "https://huggingface.co/rajlab/sam_vit_b/resolve/main/encoder.onnx"
DEFAULT_DECODER_URL = "https://huggingface.co/rajlab/sam_vit_b/resolve/main/decoder.onnx"

_SAM_INPUT_SIZE = 1024


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"Downloading:\n  {url}\n-> {out}")

    def _hook(block_num, block_size, total_size):
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = done * 100 // total_size
            sys.stdout.write(f"\r  {done/1e6:6.1f} / {total_size/1e6:6.1f} MB ({pct:3d}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    sys.stdout.write("\n")
    tmp.replace(out)


def _validate_encoder(path: Path) -> "tuple[bool, object]":
    """Confirm the encoder loads and emits a 4D image embedding on a dummy 1024x1024 frame.
    Returns (ok, embedding) so the embedding can be reused to validate the decoder."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"  ! Cannot validate (missing dep): {exc}")
        return False, None
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  ! Failed to load encoder as ONNX: {exc}")
        return False, None

    name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, _SAM_INPUT_SIZE, _SAM_INPUT_SIZE), dtype=np.float32)
    try:
        emb = sess.run(None, {name: dummy})[0]
    except Exception as exc:
        print(f"  ! Encoder inference failed: {exc}")
        return False, None
    emb = np.asarray(emb)
    if emb.ndim != 4:
        print(f"  ! Encoder output is not a 4D embedding (got shape {emb.shape}).")
        return False, None
    print(f"  OK: encoder input '{name}', embedding {emb.shape}")
    return True, emb


def _validate_decoder(path: Path, embedding) -> bool:
    """Confirm the decoder follows the standard SAM ONNX signature and returns a mask + iou."""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover
        print(f"  ! Cannot validate (missing dep): {exc}")
        return False
    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"  ! Failed to load decoder as ONNX: {exc}")
        return False

    if embedding is None:
        embedding = np.zeros((1, 256, 64, 64), dtype=np.float32)
    feed_all = {
        "image_embeddings": np.asarray(embedding, dtype=np.float32),
        "point_coords": np.array([[[512.0, 512.0], [0.0, 0.0]]], dtype=np.float32),
        "point_labels": np.array([[1.0, -1.0]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros((1,), dtype=np.float32),
        "orig_im_size": np.array([720.0, 1280.0], dtype=np.float32),
    }
    wanted = {i.name for i in sess.get_inputs()}
    missing = {"image_embeddings", "point_coords", "point_labels"} - wanted
    if missing:
        print(f"  ! Decoder is missing expected inputs {missing} (not a standard SAM ONNX export).")
        return False
    feed = {k: v for k, v in feed_all.items() if k in wanted}
    try:
        outputs = sess.run(None, feed)
    except Exception as exc:
        print(f"  ! Decoder inference failed: {exc}")
        return False
    masks = np.asarray(outputs[0])
    if masks.ndim not in (3, 4):
        print(f"  ! Decoder mask output has unexpected shape {masks.shape}.")
        return False
    print(f"  OK: decoder inputs {sorted(wanted)}, masks {masks.shape}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download + validate the SAM ViT-B instance models")
    parser.add_argument("--url-encoder", default=DEFAULT_ENCODER_URL, help="SAM ViT-B encoder ONNX URL")
    parser.add_argument("--url-decoder", default=DEFAULT_DECODER_URL, help="SAM ViT-B decoder ONNX URL")
    parser.add_argument("--out-encoder", default=str(DEFAULT_ENCODER_OUT), help="Encoder output path")
    parser.add_argument("--out-decoder", default=str(DEFAULT_DECODER_OUT), help="Decoder output path")
    parser.add_argument("--skip-download", action="store_true", help="Only validate existing files")
    args = parser.parse_args(argv)

    enc_out = Path(args.out_encoder).expanduser().resolve()
    dec_out = Path(args.out_decoder).expanduser().resolve()

    if not args.skip_download:
        for url, out in ((args.url_encoder, enc_out), (args.url_decoder, dec_out)):
            try:
                _download(url, out)
            except Exception as exc:
                print(f"Download failed: {exc}")
                print(f"Tip: pass --url-encoder/--url-decoder with sources you trust, or download manually to {out}")
                return 1
    else:
        for out in (enc_out, dec_out):
            if not out.exists():
                print(f"No file at {out} to validate.")
                return 1

    print(f"Validating encoder {enc_out.name} ({enc_out.stat().st_size/1e6:.1f} MB)...")
    enc_ok, embedding = _validate_encoder(enc_out)
    print(f"Validating decoder {dec_out.name} ({dec_out.stat().st_size/1e6:.1f} MB)...")
    dec_ok = _validate_decoder(dec_out, embedding)

    if not (enc_ok and dec_ok):
        print("\nValidation FAILED. The app will keep using the non-SAM mask fallbacks.")
        return 2

    print("\nDone. The app will now use SAM ViT-B for Person instance masks and Face boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
