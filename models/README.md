# Face Parsing Model

Recommended model:

- `resnet18.onnx` from `yakhyo/face-parsing`
- Release URL: `https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx`

Expected local filename:

- `models/face_parsing.onnx`

Download command:

```bash
cd /Users/narendersurabhi/Downloads/files
curl -L https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx -o models/face_parsing.onnx
```

Install optional runtime dependencies:

```bash
pip install -r requirements/optional-model.txt
```

Then run the app:

```bash
python portrait_enhancer_v2.py
```

Notes:

- The app will automatically use `models/face_parsing.onnx` when the optional dependencies are installed.
- Alternative model path: set `PORTRAIT_FACE_PARSING_ONNX` to any absolute ONNX model path.
- Larger alternative model: `https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.onnx`

## Optional Face Detector Model

For stronger face target detection than Haar fallback, add a YuNet detector model:

- Recommended local filename: `models/face_detection_yunet.onnx`
- Alternative env var: `PORTRAIT_FACE_DETECTOR_ONNX=/abs/path/to/face_detection_yunet.onnx`

One commonly used YuNet release filename is:

- `face_detection_yunet_2023mar.onnx`

If present, the app will use OpenCV `FaceDetectorYN` for face boxes and fall back to Haar when the model is missing.

## Optional Face Landmark Model

For more precise face/eye/lip masks and better expression warps, add a MediaPipe face landmarker task model:

- Recommended local filename: `models/face_landmarker.task`
- Alternative env var: `PORTRAIT_FACE_LANDMARKER_TASK=/abs/path/to/face_landmarker.task`

If present, the app will use `mediapipe.tasks.face_landmarker` when `mediapipe.solutions.face_mesh` is unavailable.

## Optional Subject Segmentation Model

For true people-vs-background masking in group photos, add a MediaPipe image segmenter model:

- Recommended local filename: `models/subject_segmenter.tflite`
- Alternative env var: `PORTRAIT_SUBJECT_SEGMENTER_MODEL=/abs/path/to/subject_segmenter.tflite`

The app also checks these fallback filenames:

- `models/subject_segmenter.task`
- `models/selfie_multiclass_256x256.tflite`

If present, the app will use `mediapipe.tasks.image_segmenter` for `Subjects` and `Background` masks and fall back to the face-derived heuristic mask when the model is missing.

## Optional Face Refinement Model

For higher-quality local generative face cleanup, add a CodeFormer-compatible ONNX model:

- Recommended local filename: `models/codeformer.onnx`
- Alternative env var: `PORTRAIT_CODEFORMER_ONNX=/abs/path/to/codeformer.onnx`

The app uses this model only when the `Face -> AI Refine` slider is above `0`.

Notes:

- It runs on the active face region only, not the whole image.
- It is optional and safely falls back to the normal non-generative pipeline when the model is missing.
- ONNX Runtime CoreML will be preferred on Apple Silicon when available.

## Optional Facial Hair Segmentation Model

For dedicated moustache/beard exclusion from skin retouch, add a facial-hair ONNX model:

- Recommended local filename: `models/facial_hair.onnx`
- Alternative env var: `PORTRAIT_FACIAL_HAIR_ONNX=/abs/path/to/facial_hair.onnx`

Fallback filename also checked:

- `models/beard_mustache.onnx`

Notes:

- This model is optional.
- It runs on the active face crop only.
- If missing or unusable, the app falls back to the built-in facial-hair protection mask.
