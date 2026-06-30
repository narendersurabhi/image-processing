import os

# Keep the test suite fast and deterministic regardless of which optional models happen to be
# installed locally: force the watershed Person split instead of loading the heavy ~359MB SAM
# ViT-B encoder on every FaceSegmenter construction. The real SAM path is verified manually with
# the model present (same convention the suite already uses for MODNet/NAFNet -- their real paths
# aren't exercised by unit tests either). Set before any portrait_enhancer import so it's in
# effect when InstanceSegmenter.__init__ reads it.
os.environ.setdefault("PORTRAIT_DISABLE_SAM", "1")
os.environ.setdefault("PORTRAIT_DISABLE_MASKDINO", "1")
os.environ.setdefault("PORTRAIT_DISABLE_RMBG", "1")

# Force CPU-only ONNX during tests: the suite exercises segmentation *logic*, not the accelerator
# path, and loading/compiling CoreML models is slow -- worse, an interrupted CoreML compile can
# wedge the macOS ANE compiler daemon and block every subsequent model load. Hiding the CoreML
# provider makes all model backends fall back to CPU, keeping the suite fast and robust regardless
# of the host's CoreML state.
try:
    import onnxruntime as _ort

    _orig_providers = _ort.get_available_providers
    _ort.get_available_providers = lambda: [p for p in _orig_providers() if p != "CoreMLExecutionProvider"]
except Exception:
    pass
