#!/usr/bin/env python3
"""Install/validate the optional Mask DINO person-instance model assets.

This fetches the official Mask DINO repository config and the released Swin-L COCO instance
checkpoint used by Portrait Enhancer's accuracy-first Person backend. It intentionally does not
pip-install PyTorch/Detectron2/MaskDINO dependencies: those are platform-specific and should be
installed in the user's active environment following the upstream instructions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REPO_URL = "https://github.com/IDEA-Research/MaskDINO.git"
WEIGHTS_URL = (
    "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/"
    "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
)
WEIGHTS_NAME = "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
CONFIG_REL = "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/validate Mask DINO person-instance assets.")
    parser.add_argument("--models-dir", default="models", help="Folder that stores model assets.")
    parser.add_argument("--repo-url", default=REPO_URL, help="Mask DINO git repository URL.")
    parser.add_argument("--repo-dir", default=None, help="Destination repo folder; defaults to models/MaskDINO.")
    parser.add_argument("--weights-url", default=WEIGHTS_URL, help="Checkpoint URL.")
    parser.add_argument("--weights-path", default=None, help="Checkpoint path; defaults to models/<checkpoint name>.")
    parser.add_argument("--skip-download", action="store_true", help="Only validate existing repo/config/checkpoint.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    models_dir = Path(args.models_dir)
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_dir) if args.repo_dir else models_dir / "MaskDINO"
    if not repo_dir.is_absolute():
        repo_dir = root / repo_dir
    weights_path = Path(args.weights_path) if args.weights_path else models_dir / WEIGHTS_NAME
    if not weights_path.is_absolute():
        weights_path = root / weights_path

    if not args.skip_download:
        ensure_repo(repo_dir, args.repo_url)
        download_file(args.weights_url, weights_path)

    patch_cpu_fallback(repo_dir)
    ok = validate_assets(repo_dir, weights_path)
    if ok:
        print("Mask DINO assets ready.")
        print(f"Repo:    {repo_dir}")
        print(f"Config:  {repo_dir / CONFIG_REL}")
        print(f"Weights: {weights_path}")
        print("")
        print("Runtime requirements still need to be installed in your Python environment:")
        print("  - torch")
        print("  - detectron2")
        print("  - MaskDINO dependencies from the cloned repo")
        print("")
        print("Optional overrides:")
        print(f"  PORTRAIT_MASKDINO_REPO={repo_dir}")
        print(f"  PORTRAIT_MASKDINO_WEIGHTS={weights_path}")
        print("  PORTRAIT_MASKDINO_DEVICE=cpu|cuda")
        return 0
    return 1


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if (repo_dir / ".git").exists() or (repo_dir / CONFIG_REL).exists():
        print(f"Mask DINO repo already present: {repo_dir}")
        return
    if shutil.which("git") is None:
        raise SystemExit("git is required to clone Mask DINO; install git or pass --repo-dir to an existing clone.")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning Mask DINO into {repo_dir} ...")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Checkpoint already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading {url}")
    print(f"       to {dest}")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            copied += len(chunk)
            if total:
                pct = copied / total * 100.0
                print(f"\r{pct:5.1f}% ({copied // (1024 * 1024)} MB)", end="", flush=True)
    if total:
        print()
    tmp.replace(dest)


def validate_assets(repo_dir: Path, weights_path: Path) -> bool:
    ok = True
    config_path = repo_dir / CONFIG_REL
    if not repo_dir.exists():
        print(f"Missing Mask DINO repo: {repo_dir}", file=sys.stderr)
        ok = False
    if not config_path.exists():
        print(f"Missing Mask DINO config: {config_path}", file=sys.stderr)
        ok = False
    if not weights_path.exists() or weights_path.stat().st_size <= 0:
        print(f"Missing Mask DINO checkpoint: {weights_path}", file=sys.stderr)
        ok = False
    return ok


def patch_cpu_fallback(repo_dir: Path) -> None:
    """Allow the cloned Mask DINO repo to import without the CUDA-only deformable-attn op."""
    target = repo_dir / "maskdino/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py"
    if not target.exists():
        return
    text = target.read_text()
    if "MSDA = None" in text and "MultiScaleDeformableAttention CUDA op is unavailable" in text:
        return
    old = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    info_string = (
        "\\n\\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\\n"
        "\\t`cd maskdino/modeling/pixel_decoder/ops`\\n"
        "\\t`sh make.sh`\\n"
    )
    raise ModuleNotFoundError(info_string)
'''
    new = '''try:
    import MultiScaleDeformableAttention as MSDA
except ModuleNotFoundError as e:
    # Portrait Enhancer supports CPU-only macOS installs. The official repo aborts import when
    # the CUDA extension is missing, but this module already provides ms_deform_attn_core_pytorch
    # for CPU fallback.
    MSDA = None
'''
    if old not in text:
        print(f"Warning: could not patch CPU fallback in {target}; file layout changed.", file=sys.stderr)
        return
    text = text.replace(old, new)
    old_forward = "    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):\n        ctx.im2col_step = im2col_step\n"
    new_forward = "    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):\n        if MSDA is None:\n            raise RuntimeError(\"MultiScaleDeformableAttention CUDA op is unavailable\")\n        ctx.im2col_step = im2col_step\n"
    if old_forward in text:
        text = text.replace(old_forward, new_forward)
    target.write_text(text)
    print(f"Patched CPU fallback for Mask DINO deformable attention: {target}")


if __name__ == "__main__":
    raise SystemExit(main())
