#!/usr/bin/env python3
"""Patch an ONNX model's input/output batch dimension.

This is useful for NAFNet Deep Denoise models exported with a fixed batch of 1 when the
graph itself is batch-agnostic. It does not rewrite operator internals; use it only for
models made of batch-preserving ops, and validate the result with onnxruntime before using
it as the default model.

Examples:

    python scripts/patch_onnx_batch.py models/deep_denoise.onnx models/deep_denoise_b2.onnx --batch-size 2
    python scripts/patch_onnx_batch.py models/deep_denoise.onnx models/deep_denoise_dynamic.onnx --dynamic
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Patch ONNX input/output batch dimension")
    parser.add_argument("input", help="Source ONNX model")
    parser.add_argument("output", help="Output ONNX model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-size", type=int, help="Fixed batch size to write")
    group.add_argument("--dynamic", action="store_true", help="Write symbolic batch dimension 'n'")
    return parser.parse_args()


def _set_batch_dim(value_info, *, batch_size: int | None, dynamic: bool, only_existing_one: bool = False) -> None:
    shape = value_info.type.tensor_type.shape
    if not shape.dim:
        return
    dim = shape.dim[0]
    if only_existing_one and dim.dim_value != 1:
        return
    dim.ClearField("dim_value")
    dim.ClearField("dim_param")
    if dynamic:
        dim.dim_param = "n"
    else:
        dim.dim_value = int(batch_size)


def main() -> int:
    args = _parse_args()
    try:
        import onnx
    except Exception as exc:
        print(f"Need onnx for model patching: {exc}")
        print("  uv pip install onnx")
        return 1

    src = Path(args.input).expanduser().resolve()
    dst = Path(args.output).expanduser().resolve()
    if not src.exists():
        print(f"Input model not found: {src}")
        return 1
    if args.batch_size is not None and args.batch_size < 1:
        print("--batch-size must be >= 1")
        return 1

    model = onnx.load(str(src))
    for value_info in list(model.graph.input) + list(model.graph.output):
        _set_batch_dim(value_info, batch_size=args.batch_size, dynamic=bool(args.dynamic))
    for value_info in model.graph.value_info:
        _set_batch_dim(value_info, batch_size=args.batch_size, dynamic=bool(args.dynamic), only_existing_one=True)
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst), save_as_external_data=False)
    sidecar = dst.with_suffix(dst.suffix + ".data")
    if sidecar.exists():
        sidecar.unlink()

    label = "dynamic" if args.dynamic else f"fixed batch {args.batch_size}"
    print(f"Wrote {dst} ({label}, {dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
