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

## Optional Aesthetic Crop Scoring Model

Auto Crop always starts with deterministic, safety-first candidates from the subject mask, face
boxes, and face/eye guides. To let a learned model choose the most pleasing candidate among those
safe options, add a NIMA/MUSIQ-style aesthetic scorer:

- Recommended local filename: `models/aesthetic_crop.onnx`
- Alternative env var: `PORTRAIT_AESTHETIC_CROP_MODEL=/abs/path/to/aesthetic_crop.onnx`
- Fallback filenames also checked: `models/nima.onnx`, `models/musiq.onnx`

The scorer is optional. If the model or `onnxruntime` is missing, Auto Crop uses the deterministic
crop scorer unchanged. When present, the app runs the model only on the strongest safe crop
candidates, so it can influence composition without choosing a crop that cuts off subjects, faces,
or required headroom.

Supported output contracts:

- scalar aesthetic score in `[0, 1]`
- scalar NIMA-style mean score in `[1, 10]`
- NIMA-style ordinal distribution/logits, commonly 10 bins

Input defaults to RGB `224x224`, `[-1, 1]`-normalized (Keras `mode="tf"`, the preprocessing every
documented NIMA backbone uses). The loader auto-detects NCHW vs NHWC from the ONNX input shape.
Tuning:

- `PORTRAIT_AESTHETIC_CROP_SIZE=224` overrides the fallback input size for dynamic-shape models.
- `PORTRAIT_AESTHETIC_CROP_NORMALIZE=minus_one|imagenet|none` controls preprocessing.
- `PORTRAIT_AESTHETIC_CROP_USE_COREML=1` opts into CoreML; CPU is the default because the model
  runs on only a handful of small crop previews.

The shipped default is **NIMA** (Talebi & Milanfar, *Neural Image Assessment*, TIP 2018),
MobileNet weights from titu1994/neural-image-assessment (MIT license, AVA-trained) converted to
ONNX. It scores a 10-bin aesthetic-quality distribution. To (re)generate it:

```bash
pip install tensorflow tf2onnx              # build-time only; the app runs on onnxruntime
python scripts/convert_nima_to_onnx.py      # downloads the MIT-licensed weights, writes models/aesthetic_crop.onnx
```

## Optional Accuracy-First Subject/Background Matting (RMBG-2.0)

For the strongest `Subjects` / `Background` separation, add BRIA RMBG-2.0. It is a
BiRefNet-based foreground/background model and is preferred ahead of MODNet when installed.

Important license note: RMBG-2.0 weights are gated on Hugging Face and released for
non-commercial use unless separately licensed by BRIA. Review the model card/license before
downloading.

Required local folder:

- `models/RMBG-2.0/`
- Alternative env var: `PORTRAIT_RMBG_MODEL=/abs/path/to/RMBG-2.0`

Install runtime deps in the active environment:

```bash
pip install ".[rmbg]"
```

Download + validate:

```bash
python scripts/download_rmbg_model.py --i-understand-license
python scripts/download_rmbg_model.py --skip-download
```

Switches:

- `PORTRAIT_DISABLE_RMBG=1` disables this backend.
- `PORTRAIT_RMBG_DEVICE=cpu|cuda|mps|auto` controls the PyTorch device. Default is `cpu`.
- `PORTRAIT_RMBG_SIZE=1024` controls the square inference size.

When Mask DINO is also available, RMBG's generic foreground alpha is softly gated by the union of
detected person instances. This keeps RMBG's cleaner subject edges while suppressing foreground
clutter such as plates, bottles, or chairs in group/table scenes.

## Optional High-Quality Matting Model (MODNet)

For the sharpest people-vs-background masks — recovering hair detail and full limbs that the
256px subject segmenter blurs away — add a MODNet alpha-matting ONNX model:

- Recommended local filename: `models/modnet.onnx`
- Alternative env var: `PORTRAIT_MATTING_MODEL=/abs/path/to/modnet.onnx`

The app also checks these fallback filenames:

- `models/modnet_photographic_portrait_matting.onnx`
- `models/matting.onnx`

Download + validate (downloads to `models/modnet.onnx` and verifies it loads/infers):

```bash
python scripts/download_matting_model.py            # uses default mirror
python scripts/download_matting_model.py --url <URL> # or your own trusted MODNet ONNX URL
```

The canonical source is the official MODNet repo (`ZHKKKe/MODNet`), which documents the ONNX
export of `modnet_photographic_portrait_matting`. Any MODNet ONNX with a single image input
(NCHW, RGB normalized to [-1, 1]) and a single alpha output works.

Backend preference for `Subjects`/`Background`: **RMBG-2.0/BiRefNet foreground matting** →
**MODNet matting** → MediaPipe selfie segmenter → face-derived heuristic. All paths then snap the
boundary to real image edges with a guided filter, so the model is a quality upgrade and safely
optional.

Tuning:

- `PORTRAIT_MATTING_REF_SIZE` (default `768`) sets the longer-edge resolution the net runs at.
  Higher = more detail but slower/more memory; MODNet was trained near 512, so 512–1024 is the
  useful range.

## Optional SAM Instance Masks (ViT-B)

The `Person` layer normally derives each individual's mask by geometrically splitting the
`Subjects` blob with a face-seeded watershed — which can't cleanly separate people who touch or
overlap. Add the Segment Anything (SAM ViT-B) ONNX models to instead produce **learned per-person
instance masks**: the model is prompted with positive points on one person and negative points on
the others, giving a true per-individual boundary.

The same SAM models also refine the active `Face` layer boundary. Face parsing and landmarks still
provide the semantic parts (`Skin`, `Eyes`, `Lips`, `Brows`, `Hair`, `Facial hair`); SAM only acts
as a plausibility-gated boundary refinement, so a bad/partial SAM face candidate falls back to the
existing parser or heuristic mask.

Two ONNX files are needed (the standard SAM split):

- `models/sam_vit_b_encoder.onnx` — heavy image encoder, run once per image (cached/memoized).
- `models/sam_vit_b_decoder.onnx` — lightweight mask decoder, run per person/face prompt.
- Alternative env vars: `PORTRAIT_SAM_ENCODER` / `PORTRAIT_SAM_DECODER` (absolute paths).
- Fallback filenames also checked: `models/sam_encoder.onnx`, `models/sam_decoder.onnx`.

Download + validate (fetches both files to `models/` and verifies each loads/infers):

```
python scripts/download_sam_model.py                          # uses default mirror
python scripts/download_sam_model.py --url-encoder <URL> --url-decoder <URL>   # trusted sources
python scripts/download_sam_model.py --skip-download          # validate existing files only
```

The decoder must follow the standard SAM ONNX signature (`image_embeddings`, `point_coords`,
`point_labels`, `mask_input`, `has_mask_input`, `orig_im_size` → `masks`, `iou_predictions`), the
form produced by the official SAM exporter and common community exports.

When Mask DINO is not installed, the `Person` layer uses **SAM instance masks** → face-seeded
watershed split → (with <2 faces) Person aliases Subjects. SAM is a quality upgrade and safely
optional — without it, Person keeps using the watershed split exactly as before. There's also a
**per-person plausibility net**: if SAM returns an implausibly small mask for one individual (it
occasionally returns just the head), that person falls back to the watershed mask, so a truncated
mask never reaches you while the other people keep SAM's precise instance masks.

## Optional Accuracy-First Person Instance Model (Mask DINO)

For the cleanest multi-person `Person` layer, install the Mask DINO Swin-L COCO instance
checkpoint. This backend detects actual COCO `person` instances first, then assigns each detected
face to one person mask. It runs before SAM/watershed because it solves the identity problem
directly; SAM remains a boundary/refinement fallback.

Required assets:

- `models/MaskDINO/` — clone of the official Mask DINO repository, used for the config/model code.
- `models/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth`
  — official Swin-L instance checkpoint.
- Alternative env vars:
  - `PORTRAIT_MASKDINO_REPO=/abs/path/to/MaskDINO`
  - `PORTRAIT_MASKDINO_CONFIG=/abs/path/to/config.yaml`
  - `PORTRAIT_MASKDINO_WEIGHTS=/abs/path/to/checkpoint.pth`
  - `PORTRAIT_MASKDINO_DEVICE=cpu|cuda`

Download/validate assets:

```bash
python scripts/download_maskdino_model.py
python scripts/download_maskdino_model.py --skip-download
```

Runtime Python packages are platform-specific and are not installed by the script:

- `torch`
- `detectron2`
- Mask DINO dependencies from the cloned repo

Switches:

- `PORTRAIT_DISABLE_MASKDINO=1` disables this backend and uses SAM/watershed.
- `PORTRAIT_MASKDINO_SCORE_THRESH=0.25` controls the person instance confidence threshold.

Backend preference for the `Person` layer is now: **Mask DINO person instances** → **SAM prompted
person masks** → face-seeded watershed split → (with <2 faces) Person aliases Subjects.

Backend preference for the `Face` layer boundary: **SAM refinement** → face parser/landmarks →
heuristic face mask. The face-specific path can be disabled independently with
`PORTRAIT_DISABLE_SAM_FACE=1` while leaving SAM `Person` masks enabled.

Runtime notes:

- The SAM encoder/decoder run on **CPU by default**. Measured: the ViT-B encoder loads in ~0.2s and
  infers in ~1.6s on CPU, versus ~60s first-run on the CoreML provider (model compile) plus
  intermittent CoreML failures on ViT-B — so CPU is both faster and more reliable here. Set
  `PORTRAIT_SAM_USE_COREML=1` to opt back into the accelerated provider if a future model/runtime
  makes it worthwhile.
- `PORTRAIT_DISABLE_SAM=1` disables all SAM usage even when the SAM models are installed (useful to
  skip the encoder entirely, or to compare backends).
- `PORTRAIT_DISABLE_SAM_FACE=1` disables only Face boundary refinement; `Person` can still use SAM.

## Optional ML Denoise Model

For learned noise reduction that preserves edges and fine detail far better than the
bilateral-filter fallback (which tends to smear texture when pushed), add an image-to-image
denoiser ONNX model:

- Recommended local filename: `models/denoise.onnx`
- Alternative env var: `PORTRAIT_DENOISE_MODEL=/abs/path/to/denoise.onnx`
- Fallback filenames also checked: `models/denoiser.onnx`, `models/dncnn.onnx`

Validate (or download from a source you trust) with:

```bash
python scripts/download_denoise_model.py --url <URL>      # download + validate
python scripts/download_denoise_model.py --skip-download  # validate an existing file
```

The model must take one RGB image (NCHW/NHWC, [0,1]) and output a denoised image of the same
shape. The noise-reduction slider routes through it when present and blends original->denoised
by the slider amount; during fast interactive preview it uses the bilateral path to keep
slider dragging snappy, then the model on the settled preview and on export. Inference is tiled
(env `PORTRAIT_DENOISE_TILE`, default `1024`; halo `PORTRAIT_DENOISE_HALO`, default `32`) so
full-resolution RAWs don't exhaust memory.

The shipped default is **DnCNN** (Zhang et al., *Beyond a Gaussian Denoiser*, TIP 2017),
`dncnn_color_blind` converted to ONNX from the official PyTorch weights. It is a conservative,
detail-preserving denoiser. To (re)generate it:

```bash
pip install torch onnx onnxscript          # build-time only; the app runs on onnxruntime
python scripts/convert_dncnn_to_onnx.py     # writes models/denoise.onnx (~2.7 MB)
python scripts/download_denoise_model.py --skip-download   # validate
```

If you want stronger or noise-level-tunable denoising, an FFDNet-style model (which takes a
noise-level map as a second input) drops in by replacing the file and adjusting `MLDenoiser`
in `portrait_enhancer/core/denoise.py`; no other code changes are needed.

## Optional Deep Denoise Model (heavy, real-noise)

For the highest-quality denoise on a *single hero image*, add a NAFNet real-noise model that
backs the **Deep Denoise** button. Unlike the DnCNN slider model (trained on synthetic Gaussian
noise), NAFNet is trained on real camera-noise pairs (SIDD), so it cleans real sensor grain far
better -- but it is heavy (~116M params, ~5 min per 24MP image on CoreML), which is why it is a
one-shot opt-in action baked into the source pixels, not a live slider or batch step.

- Recommended local filename: `models/deep_denoise_b2.onnx`
- Alternative env var: `PORTRAIT_DEEP_DENOISE_MODEL=/abs/path/to/deep_denoise.onnx`
- Fallback filenames also checked: `models/deep_denoise.onnx`, `models/nafnet.onnx`

The model has a fixed 256x256 input; `DeepDenoiser` tiles the image at 256 with a halo
(`PORTRAIT_DEEP_DENOISE_HALO`, default `16`) and reflect-pads partial edge tiles. The button
runs full-resolution on a background thread (progress shown in the status bar) and bakes the
result into the source, so a later export of that image includes it. Both the original and
denoised pixels are cached in memory, so after the one-time pass the button toggles
Revert / Re-apply instantly without recomputing.

Build the default from the official NAFNet-SIDD-width64 weights:

```bash
pip install torch onnx onnxscript gdown      # build-time only; the app runs on onnxruntime
python scripts/convert_nafnet_to_onnx.py --batch-size 2 --out models/deep_denoise_b2.onnx
```

Tile batching can improve throughput by running multiple 256x256 tiles in one ONNX/CoreML
call, but it raises memory pressure. The app defaults to batch 2 when
`models/deep_denoise_b2.onnx` is present. Build a fixed-batch export from weights:

```bash
python scripts/convert_nafnet_to_onnx.py --batch-size 2 --out models/deep_denoise_b2.onnx
```

Or patch the existing batch-1 ONNX when the graph is batch-agnostic:

```bash
python scripts/patch_onnx_batch.py models/deep_denoise.onnx models/deep_denoise_b2.onnx --batch-size 2
uv run --extra qt portrait-enhancer-qt
```

On 16 GB systems, batch 2 is the default upper bound. Batch 4 can increase memory pressure
enough to lose throughput, so only use it behind `PORTRAIT_DEEP_DENOISE_MODEL` after
measured tiles/minute improves without swap or compression spikes.

SCUNet (`cszn/SCUNet`, blind real-noise via realistic synthesis) is a viable alternative with
comparable quality and a smaller file (~110 MB) but is ~2.7x slower per tile; it drops in by
replacing the file (the fixed-256 tiling in `DeepDenoiser` works unchanged).

## Optional Auto WB (AI) Model

For learned auto white balance that isn't fooled by a dominant color (a red wall, a green
lawn) the way Gray World is, add an illuminant-estimation ONNX model (FC4 family):

- Recommended local filename: `models/white_balance.onnx`
- Alternative env var: `PORTRAIT_WB_MODEL=/abs/path/to/white_balance.onnx`
- Fallback filenames also checked: `models/awb.onnx`, `models/fc4.onnx`

Validate (or download from a source you trust) with:

```bash
python scripts/download_wb_model.py --url <URL>      # download + validate
python scripts/download_wb_model.py --skip-download  # validate an existing file
```

The "Auto AI" button in the White Balance panel runs the model, converts the result into
Temperature/Tint, and falls back to Gray World if the model is missing or fails.

Two model families are supported automatically:

- **Image-to-image AWB** (e.g. Deep White-Balance): output is a corrected image; the app
  derives the WB gains from the input->output color shift in linear light.
- **Illuminant estimation** (FC4 family): output reduces to a 3-vector RGB illuminant.

The shipped default is **Deep White-Balance** (Afifi & Brown, CVPR 2020), converted to ONNX
from the official PyTorch weights. To (re)generate it:

```bash
pip install torch onnx onnxscript          # build-time only; the app runs on onnxruntime
python scripts/convert_deepwb_to_onnx.py    # writes models/white_balance.onnx (~17.5 MB)
python scripts/download_wb_model.py --skip-download   # validate
```

Tuning:

- `PORTRAIT_WB_REF_SIZE` (default `512`) — resolution the model runs at.
- `PORTRAIT_WB_LINEARIZE_INPUT=1` — feed a linearized image if your model expects linear RGB.

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
