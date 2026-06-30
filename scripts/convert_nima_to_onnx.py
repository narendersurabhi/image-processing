#!/usr/bin/env python3
"""Convert NIMA (Talebi & Milanfar, TIP 2018) MobileNet aesthetic-scoring weights to ONNX.

Produces models/aesthetic_crop.onnx -- the optional tie-breaker model
core/aesthetic_crop.py's OnnxAestheticCropScorer loads when present. Auto Crop only runs it
on the handful of already-safe top candidate rectangles (see processing.py), so this never
overrides the deterministic safety scoring -- it only helps pick among ties.

tensorflow/tf2onnx are only needed for this one-time export; the app runs the ONNX with
onnxruntime alone.

    pip install tensorflow tf2onnx     # build-time only
    python scripts/convert_nima_to_onnx.py

Source architecture + weights are titu1994/neural-image-assessment (MIT license), a Keras
re-implementation of NIMA trained on the AVA aesthetics dataset: MobileNet (alpha=1.0, no
top) + Dropout(0.75) + Dense(10, softmax) over 224x224 RGB, scored as a 10-bin quality
distribution (mean rating 1-10) -- exactly the NIMA-style output aesthetic_output_to_score()
already normalizes to [0, 1]. Preprocessing matches keras.applications.mobilenet's
preprocess_input (scale to [-1, 1]) -- aesthetic_crop.py's default
PORTRAIT_AESTHETIC_CROP_NORMALIZE is "minus_one" for exactly this reason.
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://github.com/titu1994/neural-image-assessment/releases/download/v0.3/mobilenet_weights.h5"
OUT = Path(__file__).resolve().parents[1] / "models" / "aesthetic_crop.onnx"
INPUT_SIZE = 224


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading:\n  {url}\n-> {dest}")
    urllib.request.urlretrieve(url, dest)


def _build_model(tf):
    base_model = tf.keras.applications.mobilenet.MobileNet(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=1.0,
        include_top=False,
        pooling="avg",
        weights=None,
    )
    x = tf.keras.layers.Dropout(0.75)(base_model.output)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(base_model.input, x, name="nima_mobilenet")


def _validate(out: Path) -> bool:
    """Sanity-check the exported ONNX with the app's own scorer class, the same one
    Auto Crop will actually load -- not just a raw onnxruntime smoke test."""
    import os

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from portrait_enhancer.core.aesthetic_crop import OnnxAestheticCropScorer

    old_model = os.environ.get("PORTRAIT_AESTHETIC_CROP_MODEL")
    old_norm = os.environ.get("PORTRAIT_AESTHETIC_CROP_NORMALIZE")
    os.environ["PORTRAIT_AESTHETIC_CROP_MODEL"] = str(out)
    os.environ["PORTRAIT_AESTHETIC_CROP_NORMALIZE"] = "minus_one"
    try:
        scorer = OnnxAestheticCropScorer()
        if not scorer.available:
            print(f"  ! Model did not load: {scorer.reason_unavailable}")
            return False

        rng = np.random.default_rng(0)
        smooth = np.full((256, 256, 3), 0.5, dtype=np.float32)
        structured = np.clip(rng.normal(0.5, 0.2, (256, 256, 3)), 0.0, 1.0).astype(np.float32)
        score_a = scorer(smooth)
        score_b = scorer(structured)
        if score_a is None or score_b is None:
            print("  ! Scoring returned None on a valid crop.")
            return False
        if not (0.0 <= score_a <= 1.0 and 0.0 <= score_b <= 1.0):
            print(f"  ! Scores out of [0, 1] range: {score_a}, {score_b}")
            return False
        if abs(score_a - score_b) < 1e-6:
            print(f"  ! Model gives identical scores ({score_a}) for very different images.")
            return False
        print(f"  OK: backend={scorer.backend} flat={score_a:.3f} structured={score_b:.3f}")
        return True
    finally:
        for key, value in (
            ("PORTRAIT_AESTHETIC_CROP_MODEL", old_model),
            ("PORTRAIT_AESTHETIC_CROP_NORMALIZE", old_norm),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"Need tensorflow for export: {exc}")
        print("  pip install tensorflow tf2onnx")
        return 1
    try:
        import tf2onnx
    except Exception as exc:
        print(f"Need tf2onnx for export: {exc}")
        print("  pip install tf2onnx")
        return 1

    work = Path(tempfile.mkdtemp(prefix="nima_"))
    weights_path = work / "mobilenet_weights.h5"
    _fetch(WEIGHTS_URL, weights_path)

    print("Building NIMA (MobileNet) architecture...")
    model = _build_model(tf)
    try:
        model.load_weights(str(weights_path))
    except Exception as exc:
        print(f"Failed to load weights into the rebuilt architecture: {exc}")
        print("If MobileNet's internal layer names/shapes drifted in this TF version, retry with:")
        print("  model.load_weights(path, by_name=True, skip_mismatch=True)")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Exporting ONNX...")
    spec = (tf.TensorSpec((1, INPUT_SIZE, INPUT_SIZE, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=str(OUT))

    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print("Validating with the app's own scorer...")
    if not _validate(OUT):
        print("Validation FAILED. Auto Crop will fall back to the deterministic scorer.")
        return 2

    print("\nDone. Auto Crop will now use this model as a tie-breaker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
