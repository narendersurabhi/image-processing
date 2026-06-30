"""Face/feature segmentation utilities with model and heuristic backends."""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .utils import refine_mask_edges, smooth_mask, to_uint8

MASK_KEYS = ("face", "skin", "eyes", "lips", "hair", "brows", "facial_hair", "subjects", "person", "background")
FACE_OVAL_LANDMARKS = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_EYE_LANDMARKS = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE_LANDMARKS = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LIPS_LANDMARKS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78)
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291
MOUTH_UPPER_IDX = 13
MOUTH_LOWER_IDX = 14
LEFT_EYE_UPPER_IDX = 159
LEFT_EYE_LOWER_IDX = 145
RIGHT_EYE_UPPER_IDX = 386
RIGHT_EYE_LOWER_IDX = 374
LEFT_BROW_IDX = 105
RIGHT_BROW_IDX = 334

_YUNET_MAX_DIM = 1600


class FaceBoxDetector:
    """Face box detector with optional YuNet model and Haar fallback."""

    def __init__(self, model_path: str | None = None):
        data = cv2.data.haarcascades
        self._haar = cv2.CascadeClassifier(data + "haarcascade_frontalface_default.xml")
        self.backend = "haar"
        self.reason_unavailable = ""
        self._yunet = None

        model = self._resolve_model_path(model_path)
        if model is None or not hasattr(cv2, "FaceDetectorYN_create"):
            return

        try:
            self._yunet = cv2.FaceDetectorYN_create(
                str(model),
                "",
                (320, 320),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.backend = "yunet"
        except Exception as exc:
            self.reason_unavailable = f"yunet init failed: {exc}"
            self._yunet = None

    def detect(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._yunet is not None:
            faces = self._detect_yunet(img_float)
            if faces:
                return faces
        return self._detect_haar(img_float)

    def _detect_yunet(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        image = to_uint8(img_float)
        h, w = image.shape[:2]
        # Running YuNet at full native (e.g. RAW) resolution lets high-frequency
        # texture in busy backgrounds (patterns, fabric, foliage) register as
        # spurious tiny "faces" -- downscale first so detection happens at the
        # scale the model expects, then map boxes back to original coordinates.
        scale = min(1.0, _YUNET_MAX_DIM / max(h, w))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        sh, sw = image.shape[:2]
        try:
            self._yunet.setInputSize((sw, sh))
            _retval, detections = self._yunet.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            faces = _normalize_detector_faces(detections, (sh, sw))
        except Exception:
            return []
        if scale < 1.0:
            inv = 1.0 / scale
            faces = [tuple(int(round(v * inv)) for v in f) for f in faces]
        return faces

    def _detect_haar(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray_eq = cv2.equalizeHist(cv2.cvtColor(cv2.cvtColor(to_uint8(img_float), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY))
        h, w = gray_eq.shape[:2]
        faces = self._haar.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(max(30, w // 20), max(30, h // 20)),
        )
        return sorted([tuple(int(v) for v in f) for f in faces], key=lambda r: r[2] * r[3], reverse=True)

    def _resolve_model_path(self, explicit_path: str | None) -> Path | None:
        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))

        env_path = os.getenv("PORTRAIT_FACE_DETECTOR_ONNX")
        if env_path:
            candidates.append(Path(env_path))

        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "face_detection_yunet.onnx")
        candidates.append(models_dir / "face_detection_yunet_2023mar.onnx")

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None


class HeuristicFaceSegmenter:
    """
    Heuristic masks based on OpenCV Haar cascades + HSV/YCrCb analysis.
    """

    def __init__(self):
        self._face_detector = FaceBoxDetector()
        data = cv2.data.haarcascades
        self.eye_cas = cv2.CascadeClassifier(data + "haarcascade_eye.xml")

    def list_faces(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        return self._face_detector.detect(img_float)

    @property
    def detector_backend(self) -> str:
        return getattr(self._face_detector, "backend", "haar")

    def segment(self, img_float: np.ndarray, face_index: int | None = 0) -> dict:
        h, w = img_float.shape[:2]
        gray_eq = self._gray_eq(img_float)
        faces = self.list_faces(img_float)

        masks = _empty_masks(h, w)

        if len(faces) == 0:
            masks["face"] = self._center_oval(h, w, 0.5, 0.55, 0.30, 0.38)
            masks["skin"] = self._skin_mask(img_float)
            masks["hair"] = self._hair_fallback(img_float, masks["face"])
            masks["facial_hair"] = np.zeros((h, w), dtype=np.float32)
            masks["subjects"], masks["background"] = _subjects_background_masks((h, w), [], masks)
            return masks

        idx = _safe_face_index(face_index, len(faces))
        fx, fy, fw, fh = faces[idx]

        face_m = np.zeros((h, w), dtype=np.float32)
        cx, cy = fx + fw // 2, fy + fh // 2
        cv2.ellipse(face_m, (cx, cy), (fw // 2, int(fh * 0.58)), 0, 0, 360, 1.0, -1)
        masks["face"] = smooth_mask(face_m, sigma=max(2.0, fw * 0.03))

        skin_all = self._skin_mask(img_float)
        skin_roi = np.zeros_like(skin_all)
        y1 = max(0, fy - int(fh * 0.1))
        y2 = min(h, fy + int(fh * 1.5))
        x1 = max(0, fx - int(fw * 0.3))
        x2 = min(w, fx + fw + int(fw * 0.3))
        skin_roi[y1:y2, x1:x2] = skin_all[y1:y2, x1:x2]
        masks["skin"] = smooth_mask(skin_roi, sigma=3.0)

        eye_m = np.zeros((h, w), dtype=np.float32)
        face_roi_gray = gray_eq[fy : fy + fh, fx : fx + fw]
        eyes = self.eye_cas.detectMultiScale(
            face_roi_gray[: fh // 2],
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(max(8, fw // 8), max(6, fw // 10)),
        )
        if len(eyes) > 0:
            for ex, ey, ew, eh in eyes[:2]:
                abs_cx = fx + ex + ew // 2
                abs_cy = fy + ey + eh // 2
                cv2.ellipse(eye_m, (abs_cx, abs_cy), (int(ew * 0.7), int(eh * 0.5)), 0, 0, 360, 1.0, -1)
        else:
            ey_top = fy + int(fh * 0.20)
            ey_bot = fy + int(fh * 0.42)
            third = fw // 3
            for ex_off in [fx + int(fw * 0.15), fx + fw - int(fw * 0.15) - third]:
                ecx = ex_off + third // 2
                ecy = (ey_top + ey_bot) // 2
                cv2.ellipse(eye_m, (ecx, ecy), (third // 2, (ey_bot - ey_top) // 2), 0, 0, 360, 1.0, -1)
        masks["eyes"] = smooth_mask(eye_m, sigma=max(1.0, fw * 0.012))

        lip_m = np.zeros((h, w), dtype=np.float32)
        lip_y1 = fy + int(fh * 0.68)
        lip_y2 = fy + int(fh * 0.92)
        lip_x1 = fx + int(fw * 0.22)
        lip_x2 = fx + fw - int(fw * 0.22)

        lip_region = img_float[lip_y1:lip_y2, lip_x1:lip_x2]
        lip_m[lip_y1:lip_y2, lip_x1:lip_x2] = self._lip_color_mask(lip_region)

        if lip_m.sum() < 100:
            lcx = (lip_x1 + lip_x2) // 2
            lcy = (lip_y1 + lip_y2) // 2
            cv2.ellipse(
                lip_m,
                (lcx, lcy),
                ((lip_x2 - lip_x1) // 2, (lip_y2 - lip_y1) // 2),
                0,
                0,
                360,
                1.0,
                -1,
            )

        masks["lips"] = smooth_mask(lip_m, sigma=max(2.0, fw * 0.018))
        masks["hair"] = self._hair_mask(img_float, fx, fy, fw, fh, h, w)
        masks["facial_hair"] = _facial_hair_mask(
            img_float,
            face_box=(fx, fy, fw, fh),
            masks=masks,
            guides=None,
        )
        masks["subjects"], masks["background"] = _subjects_background_masks((h, w), faces, masks)
        return masks

    def _gray_eq(self, img_float):
        bgr = cv2.cvtColor(to_uint8(img_float), cv2.COLOR_RGB2BGR)
        return cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))

    def _skin_mask(self, img_float):
        hsv = cv2.cvtColor(to_uint8(img_float), cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(to_uint8(img_float), cv2.COLOR_RGB2YCrCb)

        m1 = cv2.inRange(hsv, np.array([0, 20, 60], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
        m2 = cv2.inRange(hsv, np.array([0, 10, 40], dtype=np.uint8), np.array([20, 180, 200], dtype=np.uint8))
        m3 = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))

        combined = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned.astype(np.float32) / 255.0

    def _lip_color_mask(self, region):
        if region.size == 0:
            return np.zeros(region.shape[:2], dtype=np.float32)
        hsv = cv2.cvtColor(to_uint8(region), cv2.COLOR_RGB2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 30, 50]), np.array([180, 255, 255]))
        m3 = cv2.inRange(hsv, np.array([140, 20, 80]), np.array([180, 150, 255]))
        combined = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
        return combined.astype(np.float32) / 255.0

    def _hair_mask(self, img_float, fx, fy, fw, fh, h, w):
        mask = np.zeros((h, w), dtype=np.float32)

        hair_y2 = fy + int(fh * 0.15)
        hair_x1 = max(0, fx - int(fw * 0.5))
        hair_x2 = min(w, fx + fw + int(fw * 0.5))
        mask[:hair_y2, hair_x1:hair_x2] = 1.0

        skin = self._skin_mask(img_float)
        dark = 1.0 - skin

        ext_y1 = max(0, fy - int(fh * 0.6))
        ext_y2 = min(h, fy + int(fh * 0.3))
        ext_x1 = max(0, fx - int(fw * 0.7))
        ext_x2 = min(w, fx + fw + int(fw * 0.7))
        zone = np.zeros((h, w), dtype=np.float32)
        zone[ext_y1:ext_y2, ext_x1:ext_x2] = 1.0

        hair_candidate = dark * zone
        mask = np.maximum(mask, hair_candidate)
        return smooth_mask(mask, sigma=max(3.0, fw * 0.04))

    def _hair_fallback(self, img_float, face_mask):
        h, w = img_float.shape[:2]
        skin = self._skin_mask(img_float)
        return smooth_mask(
            np.clip((1.0 - skin) * (1.0 - face_mask), 0, 1) * self._center_oval(h, w, 0.5, 0.15, 0.45, 0.20),
            sigma=6.0,
        )

    def _center_oval(self, h, w, cx_frac, cy_frac, rx_frac, ry_frac):
        y_grid, x_grid = np.ogrid[:h, :w]
        cx, cy = w * cx_frac, h * cy_frac
        rx, ry = w * rx_frac, h * ry_frac
        dist = ((x_grid - cx) / rx) ** 2 + ((y_grid - cy) / ry) ** 2
        return np.clip(1.0 - dist, 0, 1).astype(np.float32)


class ModelFaceSegmenter:
    """
    Model-based segmentation with ONNX face parsing and MediaPipe landmarks.

    Expected label mapping (CelebAMask-HQ/BiSeNet compatible):
    1=skin, 4/5=eyes, 11/12/13=mouth/lips, 17=hair.
    """

    def __init__(self, model_path: str | None = None):
        self.available = False
        self.reason_unavailable = ""
        self.landmark_backend = "none"
        self.landmark_note = ""
        self.execution_provider = "cpu"
        self._session = None
        self._face_mesh = None
        self._mp_face_mesh = None
        self._task_landmarker = None
        self._mp = None
        self._input_name = None
        self._input_hw = (512, 512)

        mp = None
        try:
            import mediapipe as mp
        except Exception:
            mp = None
        self._mp = mp
        self._init_landmark_backend(mp)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model = self._resolve_model_path(model_path)
        if model is None:
            self.reason_unavailable = "face parsing model not found"
            return

        try:
            providers = self._preferred_onnx_providers(ort)
            self._prepare_coreml_cache_dir(providers)
            self._session = ort.InferenceSession(str(model), providers=providers)
            active_providers = list(self._session.get_providers())
            if "CoreMLExecutionProvider" in active_providers:
                self.execution_provider = "coreml"
            else:
                self.execution_provider = "cpu"
            self._input_name = self._session.get_inputs()[0].name
            self._input_hw = self._infer_input_hw(self._session.get_inputs()[0].shape)
            self.available = True
        except Exception as exc:
            if "CoreMLExecutionProvider" in providers:
                try:
                    self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
                    self.execution_provider = "cpu"
                    self._input_name = self._session.get_inputs()[0].name
                    self._input_hw = self._infer_input_hw(self._session.get_inputs()[0].shape)
                    self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
                    self.available = True
                    return
                except Exception as cpu_exc:
                    self.reason_unavailable = f"model init failed: {cpu_exc}"
                    return
            self.reason_unavailable = f"model init failed: {exc}"

    def segment(self, img_float: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> dict:
        masks, _guides = self.segment_with_guides(img_float, face_box=face_box)
        return masks

    def segment_with_guides(
        self,
        img_float: np.ndarray,
        face_box: tuple[int, int, int, int] | None = None,
    ) -> tuple[dict, dict | None]:
        h, w = img_float.shape[:2]
        full_labels, full_class_probs = self._run_parsing_model(img_float)
        full_masks = _build_model_masks_from_output(full_labels, full_class_probs)

        masks = dict(full_masks)
        crop_box = None
        if face_box is not None:
            crop_box = _expanded_face_crop(face_box, (h, w), x_scale=1.1, y_scale=1.25, y_up_scale=0.85)
            x1, y1, x2, y2 = crop_box
            if (x2 - x1) >= 32 and (y2 - y1) >= 32:
                crop_img = img_float[y1:y2, x1:x2]
                crop_labels, crop_class_probs = self._run_parsing_model(crop_img)
                crop_masks = _build_model_masks_from_output(crop_labels, crop_class_probs)
                crop_masks = _paste_crop_masks(crop_masks, crop_box, (h, w), crop_img.shape[:2])
                masks = _merge_model_views(full_masks, crop_masks, face_box=face_box, shape_hw=(h, w))
            else:
                crop_box = None

        landmarks, guides = self._landmark_masks(img_float, face_box=face_box)
        if landmarks is None and face_box is not None:
            landmarks, guides = self._landmark_masks_on_crop(img_float, face_box=face_box)
        if landmarks is not None:
            for key in ("face", "eyes", "lips"):
                masks[key] = np.maximum(masks[key], landmarks[key])

        masks = _postprocess_model_masks(masks, img_float, face_box=face_box, shape_hw=(h, w))
        masks = _refine_face_part_masks(masks, img_float, face_box=face_box, guides=guides)
        masks["facial_hair"] = _facial_hair_mask(
            img_float,
            face_box=face_box,
            masks=masks,
            guides=guides,
        )

        width_ref = max(1, face_box[2] if face_box is not None else w)
        masks["face"] = smooth_mask(np.clip(masks["face"], 0, 1), sigma=max(1.0, width_ref * 0.012))
        masks["skin"] = smooth_mask(np.clip(masks["skin"], 0, 1), sigma=max(1.0, width_ref * 0.008))
        masks["brows"] = smooth_mask(np.clip(masks["brows"], 0, 1), sigma=max(0.6, width_ref * 0.003))
        masks["facial_hair"] = smooth_mask(np.clip(masks["facial_hair"], 0, 1), sigma=max(0.8, width_ref * 0.004))
        masks["eyes"] = smooth_mask(np.clip(masks["eyes"], 0, 1), sigma=max(0.6, width_ref * 0.0035))
        masks["lips"] = smooth_mask(np.clip(masks["lips"], 0, 1), sigma=max(0.8, width_ref * 0.0045))
        masks["hair"] = smooth_mask(np.clip(masks["hair"], 0, 1), sigma=max(1.2, width_ref * 0.006))
        faces_for_subjects = [face_box] if face_box is not None else []
        masks["subjects"], masks["background"] = _subjects_background_masks((h, w), faces_for_subjects, masks)
        return masks, guides

    def _run_parsing_model(self, img_float: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        img = to_uint8(img_float)
        in_h, in_w = self._input_hw
        resized = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

        # Common preprocessing for BiSeNet-style models.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

        out = self._session.run(None, {self._input_name: tensor})[0]
        class_probs_small = self._to_class_probabilities(out)
        if class_probs_small is not None:
            labels_small = np.argmax(class_probs_small, axis=-1).astype(np.uint8)
            class_probs = cv2.resize(class_probs_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            class_probs = np.clip(class_probs, 0.0, 1.0).astype(np.float32)
            if class_probs.ndim == 2:
                class_probs = class_probs[:, :, None]
        else:
            labels_small = self._to_label_map(out)
            class_probs = None
        labels = cv2.resize(labels_small.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return labels, class_probs

    def _to_class_probabilities(self, output: np.ndarray) -> np.ndarray | None:
        scores = self._extract_class_scores(output)
        if scores is None or scores.ndim != 3:
            return None
        return _softmax_last_axis(scores)

    def _to_label_map(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 4:
            if output.shape[1] > 1:
                return np.argmax(output[0], axis=0)
            if output.shape[-1] > 1:
                return np.argmax(output[0], axis=-1)
        if output.ndim == 3:
            if output.shape[0] > 1:
                return np.argmax(output, axis=0)
            if output.shape[-1] > 1:
                return np.argmax(output, axis=-1)
            return output[0]
        if output.ndim == 2:
            return output
        raise RuntimeError(f"unsupported model output shape: {output.shape}")

    def _extract_class_scores(self, output: np.ndarray) -> np.ndarray | None:
        if output.ndim == 4:
            if output.shape[1] > 1:
                return np.transpose(output[0], (1, 2, 0)).astype(np.float32)
            if output.shape[-1] > 1:
                return output[0].astype(np.float32)
            return None
        if output.ndim == 3:
            if output.shape[0] > 1:
                return np.transpose(output, (1, 2, 0)).astype(np.float32)
            if output.shape[-1] > 1:
                return output.astype(np.float32)
            return None
        return None

    def _init_landmark_backend(self, mp) -> None:
        self.landmark_backend = "none"
        self.landmark_note = ""
        self._face_mesh = None
        self._mp_face_mesh = None
        self._task_landmarker = None

        if mp is None:
            self.landmark_note = "mediapipe unavailable; using ONNX parsing only"
            return

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                self._mp_face_mesh = mp.solutions.face_mesh
                self.landmark_backend = "mediapipe.solutions.face_mesh"
                return
            except Exception as exc:
                self.landmark_note = f"mediapipe.solutions.face_mesh unavailable: {exc}"

        if hasattr(mp, "tasks"):
            task_model = self._resolve_landmarker_path()
            if task_model is None:
                if not self.landmark_note:
                    self.landmark_note = "face landmarker task model not found; using ONNX parsing only"
                return
            try:
                from mediapipe.tasks.python import vision

                options = vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(task_model)),
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=5,
                )
                self._task_landmarker = vision.FaceLandmarker.create_from_options(options)
                self.landmark_backend = "mediapipe.tasks.face_landmarker"
                self.landmark_note = ""
                return
            except Exception as exc:
                self.landmark_note = f"mediapipe.tasks.face_landmarker unavailable: {exc}"
                return

        if not self.landmark_note:
            self.landmark_note = "mediapipe landmark backend unavailable; using ONNX parsing only"

    def _landmark_masks(self, img_float: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict | None, dict | None]:
        if self._face_mesh is not None and self._mp_face_mesh is not None:
            return self._landmark_masks_solutions(img_float, face_box=face_box)
        if self._task_landmarker is not None and self._mp is not None:
            return self._landmark_masks_tasks(img_float, face_box=face_box)
        return None, None

    def _landmark_masks_on_crop(
        self,
        img_float: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> tuple[dict | None, dict | None]:
        """Retry landmarks on an enlarged per-face crop.

        In group photos the full preview can leave each face only ~70-90px wide. MediaPipe often
        misses those faces even though YuNet detected them. Cropping around the target face and
        upscaling the crop gives the landmarker a normal portrait-scale input, then maps masks and
        expression guides back into full-preview coordinates.
        """
        h, w = img_float.shape[:2]
        fx, fy, fw, fh = face_box
        if fw <= 0 or fh <= 0:
            return None, None

        crop_box = _expanded_face_crop(face_box, (h, w), x_scale=1.65, y_scale=1.85, y_up_scale=1.05)
        x1, y1, x2, y2 = crop_box
        crop = img_float[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch < 16 or cw < 16:
            return None, None

        target_face = 320.0
        scale = max(1.0, target_face / max(float(fw), float(fh), 1.0))
        scale = min(scale, 768.0 / max(float(ch), float(cw), 1.0), 6.0)
        if scale > 1.01:
            resized = cv2.resize(
                crop,
                (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            resized = crop
            scale = 1.0

        local_face = (
            int(round((fx - x1) * scale)),
            int(round((fy - y1) * scale)),
            max(1, int(round(fw * scale))),
            max(1, int(round(fh * scale))),
        )
        landmarks, guides = self._landmark_masks(resized, face_box=local_face)
        if landmarks is None:
            return None, None

        pasted = _paste_crop_masks(landmarks, crop_box, (h, w), resized.shape[:2])
        mapped_guides = _map_crop_guides_to_full(guides, crop_box, scale)
        return pasted, mapped_guides

    def _landmark_masks_solutions(self, img_float: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict | None, dict | None]:
        image = to_uint8(img_float)
        result = self._face_mesh.process(image)
        if not result.multi_face_landmarks:
            return None, None

        h, w = image.shape[:2]
        landmark_sets = [face_landmarks.landmark for face_landmarks in result.multi_face_landmarks]
        landmarks = self._select_landmark_set(landmark_sets, w, h, face_box)
        if landmarks is None:
            return None, None

        face = self._draw_region(landmarks, self._mp_face_mesh.FACEMESH_FACE_OVAL, w, h)
        left_eye = self._draw_region(landmarks, self._mp_face_mesh.FACEMESH_LEFT_EYE, w, h)
        right_eye = self._draw_region(landmarks, self._mp_face_mesh.FACEMESH_RIGHT_EYE, w, h)
        lips = self._draw_region(landmarks, self._mp_face_mesh.FACEMESH_LIPS, w, h)
        eyes = cv2.dilate(np.maximum(left_eye, right_eye), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.float32)

        return {
            "face": face.astype(np.float32),
            "eyes": eyes.astype(np.float32),
            "lips": lips.astype(np.float32),
        }, self._guides_from_landmarks(landmarks, w, h)

    def _landmark_masks_tasks(self, img_float: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> tuple[dict | None, dict | None]:
        image = to_uint8(img_float)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        result = self._task_landmarker.detect(mp_image)
        face_landmarks_list = getattr(result, "face_landmarks", None)
        if not face_landmarks_list:
            return None, None

        h, w = image.shape[:2]
        landmarks = self._select_landmark_set(face_landmarks_list, w, h, face_box)
        if landmarks is None:
            return None, None

        face = self._draw_region_indices(landmarks, FACE_OVAL_LANDMARKS, w, h)
        left_eye = self._draw_region_indices(landmarks, LEFT_EYE_LANDMARKS, w, h)
        right_eye = self._draw_region_indices(landmarks, RIGHT_EYE_LANDMARKS, w, h)
        lips = self._draw_region_indices(landmarks, LIPS_LANDMARKS, w, h)
        eyes = cv2.dilate(np.maximum(left_eye, right_eye), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(np.float32)

        return {
            "face": face.astype(np.float32),
            "eyes": eyes.astype(np.float32),
            "lips": lips.astype(np.float32),
        }, self._guides_from_landmarks(landmarks, w, h)

    def _draw_region(self, landmarks, connections, w, h):
        indices = sorted({idx for edge in connections for idx in edge})
        if len(indices) < 3:
            return np.zeros((h, w), dtype=np.float32)

        points = np.array(
            [[int(np.clip(landmarks[idx].x * w, 0, w - 1)), int(np.clip(landmarks[idx].y * h, 0, h - 1))] for idx in indices],
            dtype=np.int32,
        )

        hull = cv2.convexHull(points)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(np.float32)

    def _draw_region_indices(self, landmarks, indices, w, h):
        if len(indices) < 3:
            return np.zeros((h, w), dtype=np.float32)
        points = np.array(
            [
                [
                    int(np.clip(landmarks[idx].x * w, 0, w - 1)),
                    int(np.clip(landmarks[idx].y * h, 0, h - 1)),
                ]
                for idx in indices
            ],
            dtype=np.int32,
        )
        hull = cv2.convexHull(points)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(np.float32)

    def _select_landmark_set(self, landmark_sets, w: int, h: int, face_box: tuple[int, int, int, int] | None):
        if not landmark_sets:
            return None
        if face_box is None:
            return landmark_sets[0]

        fx, fy, fw, fh = face_box
        target = np.array([fx + fw * 0.5, fy + fh * 0.5], dtype=np.float32)
        # The landmarker can detect fewer faces than the box detector (e.g. a smaller/more
        # rotated second face it's less confident about) -- without a rejection radius, "closest
        # available" still returns some other face's landmarks, pasting their eyes/lips into this
        # face's mask at the wrong location (confirmed: a 2-face photo where the landmarker only
        # found face 1 was reusing face 1's landmarks for face 2 too, identical down to the pixel
        # count). A genuine match should be within roughly a face's width/height of the target.
        max_dist = max(fw, fh) * 1.5
        best = None
        best_score = None
        for landmarks in landmark_sets:
            box = self._landmark_bbox(landmarks, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            dist = float(np.linalg.norm(center - target))
            if dist > max_dist:
                continue
            scale_penalty = abs((x2 - x1) - fw) * 0.25 + abs((y2 - y1) - fh) * 0.20
            score = dist + scale_penalty
            if best_score is None or score < best_score:
                best = landmarks
                best_score = score
        return best

    def _landmark_bbox(self, landmarks, w: int, h: int) -> tuple[int, int, int, int] | None:
        if landmarks is None or len(landmarks) == 0:
            return None
        xs = [int(np.clip(pt.x * w, 0, w - 1)) for pt in landmarks]
        ys = [int(np.clip(pt.y * h, 0, h - 1)) for pt in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    def _guides_from_landmarks(self, landmarks, w: int, h: int) -> dict:
        def point(idx: int) -> tuple[float, float]:
            pt = landmarks[idx]
            return (
                float(np.clip(pt.x * w, 0, w - 1)),
                float(np.clip(pt.y * h, 0, h - 1)),
            )

        mouth_left = point(MOUTH_LEFT_IDX)
        mouth_right = point(MOUTH_RIGHT_IDX)
        mouth_upper = point(MOUTH_UPPER_IDX)
        mouth_lower = point(MOUTH_LOWER_IDX)
        left_eye_upper = point(LEFT_EYE_UPPER_IDX)
        left_eye_lower = point(LEFT_EYE_LOWER_IDX)
        right_eye_upper = point(RIGHT_EYE_UPPER_IDX)
        right_eye_lower = point(RIGHT_EYE_LOWER_IDX)
        left_brow = point(LEFT_BROW_IDX)
        right_brow = point(RIGHT_BROW_IDX)

        return {
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
            "mouth_upper": mouth_upper,
            "mouth_lower": mouth_lower,
            "mouth_center": (
                float((mouth_left[0] + mouth_right[0] + mouth_upper[0] + mouth_lower[0]) * 0.25),
                float((mouth_left[1] + mouth_right[1] + mouth_upper[1] + mouth_lower[1]) * 0.25),
            ),
            "left_eye_upper": left_eye_upper,
            "left_eye_lower": left_eye_lower,
            "right_eye_upper": right_eye_upper,
            "right_eye_lower": right_eye_lower,
            "left_eye_center": (
                float((left_eye_upper[0] + left_eye_lower[0]) * 0.5),
                float((left_eye_upper[1] + left_eye_lower[1]) * 0.5),
            ),
            "right_eye_center": (
                float((right_eye_upper[0] + right_eye_lower[0]) * 0.5),
                float((right_eye_upper[1] + right_eye_lower[1]) * 0.5),
            ),
            "left_brow": left_brow,
            "right_brow": right_brow,
        }

    def _resolve_model_path(self, explicit_path: str | None) -> Path | None:
        candidates = []

        if explicit_path:
            candidates.append(Path(explicit_path))

        env_path = os.getenv("PORTRAIT_FACE_PARSING_ONNX")
        if env_path:
            candidates.append(Path(env_path))

        repo_default = Path(__file__).resolve().parents[2] / "models" / "face_parsing.onnx"
        candidates.append(repo_default)

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _resolve_landmarker_path(self) -> Path | None:
        candidates = []

        env_path = os.getenv("PORTRAIT_FACE_LANDMARKER_TASK")
        if env_path:
            candidates.append(Path(env_path))

        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "face_landmarker.task")
        candidates.append(models_dir / "face_landmarker_v2_with_blendshapes.task")

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _infer_input_hw(self, shape) -> tuple[int, int]:
        if len(shape) >= 4 and isinstance(shape[-2], int) and isinstance(shape[-1], int):
            return int(shape[-2]), int(shape[-1])
        return 512, 512

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ["TMPDIR"] = cache_str
        os.environ["TMP"] = cache_str
        os.environ["TEMP"] = cache_str
        tempfile.tempdir = cache_str


class SubjectSegmenter:
    """Optional full-person segmentation using MediaPipe ImageSegmenter tasks."""

    def __init__(self):
        self.available = False
        self.backend = "heuristic"
        self.reason_unavailable = ""
        self._mp = None
        self._segmenter = None

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
        except Exception as exc:
            self.reason_unavailable = f"mediapipe tasks unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "subject segmenter model not found"
            return

        try:
            options = vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                output_confidence_masks=True,
                output_category_mask=True,
            )
            self._segmenter = vision.ImageSegmenter.create_from_options(options)
            self._mp = mp
            self.available = True
            self.backend = "mediapipe.tasks.image_segmenter"
        except Exception as exc:
            self.reason_unavailable = f"subject segmenter init failed: {exc}"

    def segment(self, img_float: np.ndarray) -> np.ndarray | None:
        if not self.available or self._segmenter is None or self._mp is None:
            return None
        image = to_uint8(img_float)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        try:
            result = self._segmenter.segment(mp_image)
        except Exception:
            return None

        confidence_masks = getattr(result, "confidence_masks", None)
        if confidence_masks:
            conf_arrays = []
            for mask in confidence_masks:
                arr = np.asarray(mask.numpy_view(), dtype=np.float32)
                arr = np.squeeze(arr)
                if arr.ndim != 2 or arr.size == 0:
                    continue
                if arr.shape != image.shape[:2]:
                    arr = cv2.resize(arr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                conf_arrays.append(np.clip(arr, 0.0, 1.0).astype(np.float32))
            if not conf_arrays:
                return None
            if len(conf_arrays) == 1:
                subject = np.clip(conf_arrays[0], 0.0, 1.0)
            else:
                stacked = np.stack(conf_arrays, axis=-1)
                if stacked.shape[-1] <= 1:
                    subject = np.clip(stacked[:, :, 0], 0.0, 1.0)
                else:
                    subject = np.clip(stacked[:, :, 1:].max(axis=-1), 0.0, 1.0)
            return subject.astype(np.float32)

        category_mask = getattr(result, "category_mask", None)
        if category_mask is not None:
            cats = np.asarray(category_mask.numpy_view(), dtype=np.uint8)
            cats = np.squeeze(cats)
            if cats.ndim != 2 or cats.size == 0:
                return None
            if cats.shape != image.shape[:2]:
                cats = cv2.resize(cats.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            subject = (cats != 0).astype(np.float32)
            return subject
        return None

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_SUBJECT_SEGMENTER_MODEL")
        if env_path:
            candidates.append(Path(env_path))

        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "subject_segmenter.tflite")
        candidates.append(models_dir / "subject_segmenter.task")
        candidates.append(models_dir / "selfie_multiclass_256x256.tflite")

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None


class BackgroundRemovalSegmenter:
    """Optional accuracy-first foreground/background matting via BRIA RMBG-2.0.

    This is deliberately local-only at runtime: the model must already be present under
    models/RMBG-2.0 (or PORTRAIT_RMBG_MODEL) and all Python deps must already be installed.
    That keeps ordinary app startup predictable and avoids an implicit gated-license download.
    """

    def __init__(self):
        self.available = False
        self.backend = "heuristic"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._model = None
        self._torch = None
        self._device = "cpu"
        try:
            self.image_size = int(os.getenv("PORTRAIT_RMBG_SIZE", "1024"))
        except (TypeError, ValueError):
            self.image_size = 1024
        self.image_size = max(256, min(1536, self.image_size))

        if _env_truthy("PORTRAIT_DISABLE_RMBG"):
            self.reason_unavailable = "disabled via PORTRAIT_DISABLE_RMBG"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "RMBG-2.0 model folder not found"
            return

        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except Exception as exc:
            self.reason_unavailable = f"RMBG runtime unavailable: {exc}"
            return

        device = self._select_device(torch)
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
            )
            model.eval()
            model.to(device)
        except Exception as exc:
            self.reason_unavailable = f"RMBG init failed: {exc}"
            return

        self._torch = torch
        self._model = model
        self._device = device
        self.execution_provider = device
        self.backend = "rmbg2"
        self.available = True

    def segment(self, img_float: np.ndarray) -> np.ndarray | None:
        if not self.available or self._model is None or self._torch is None:
            return None
        arr = np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        image = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
        small = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = np.asarray(small, dtype=np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)

        torch = self._torch
        try:
            with torch.inference_mode():
                input_tensor = torch.from_numpy(tensor).to(self._device)
                out = self._model(input_tensor)
                pred = out[-1] if isinstance(out, (list, tuple)) else out
                pred = torch.sigmoid(pred).detach().float().cpu().numpy()
        except Exception:
            return None

        alpha = np.squeeze(np.asarray(pred, dtype=np.float32))
        if alpha.ndim != 2 or alpha.size == 0:
            return None
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
        return (np.asarray(alpha_img, dtype=np.float32) / 255.0).astype(np.float32)

    def _select_device(self, torch) -> str:
        requested = os.getenv("PORTRAIT_RMBG_DEVICE", "cpu").strip().lower()
        if requested == "auto":
            if getattr(torch.cuda, "is_available", lambda: False)():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda"
        if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_RMBG_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "RMBG-2.0")
        candidates.append(models_dir / "rmbg-2.0")
        candidates.append(models_dir / "briaai_RMBG-2.0")
        for path in candidates:
            if path.exists() and path.is_dir():
                return path
        return None


class MattingSegmenter:
    """Optional high-quality human alpha matting (MODNet, ONNX).

    Produces a soft subject alpha far sharper than the 256x256 selfie segmenter:
    it recovers hair detail and full limbs that the low-res model blurs away. The net
    runs at a bounded reference size (MODNet was trained ~512px; running modestly above
    that keeps detail without straying from its training distribution), then the alpha is
    returned at native resolution -- the caller's guided filter snaps the boundary to the
    real image edges. Falls back silently (available=False) if the model file or
    onnxruntime is missing, so the selfie+guided-filter path keeps working unchanged.
    """

    def __init__(self, ref_size: int | None = None):
        self.available = False
        self.backend = "heuristic"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._session = None
        self._input_name = None
        try:
            self.ref_size = int(ref_size if ref_size is not None else os.getenv("PORTRAIT_MATTING_REF_SIZE", "768"))
        except (TypeError, ValueError):
            self.ref_size = 768
        self.ref_size = max(256, self.ref_size)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "matting model not found"
            return

        try:
            providers = self._preferred_onnx_providers(ort)
            self._prepare_coreml_cache_dir(providers)
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            if "CoreMLExecutionProvider" in set(self._session.get_providers()):
                self.execution_provider = "coreml"
            self._input_name = self._session.get_inputs()[0].name
            self.available = True
            self.backend = "modnet"
        except Exception as exc:
            try:
                self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
                self.execution_provider = "cpu"
                self.available = True
                self.backend = "modnet"
                self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
            except Exception as cpu_exc:
                self.reason_unavailable = f"matting init failed: {cpu_exc}"

    def segment(self, img_float: np.ndarray) -> np.ndarray | None:
        if not self.available or self._session is None:
            return None
        arr = np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return None
        rh, rw = self._inference_size(h, w)
        small = cv2.resize(arr, (rw, rh), interpolation=cv2.INTER_AREA)
        # MODNet normalization: RGB in [0,1] -> [-1,1], NCHW.
        tensor = np.transpose((small - 0.5) / 0.5, (2, 0, 1))[None, ...].astype(np.float32)
        try:
            out = self._session.run(None, {self._input_name: tensor})[0]
        except Exception:
            return None
        alpha = np.squeeze(np.asarray(out, dtype=np.float32))
        if alpha.ndim != 2 or alpha.size == 0:
            return None
        alpha = np.clip(alpha, 0.0, 1.0)
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    def _inference_size(self, h: int, w: int) -> tuple[int, int]:
        """Scale the longer edge toward ref_size (up or down), then floor both sides to a
        multiple of 32 as MODNet's architecture requires."""
        ref = self.ref_size
        if max(h, w) < ref or min(h, w) > ref:
            if w >= h:
                rw, rh = ref, int(round(h / w * ref))
            else:
                rh, rw = ref, int(round(w / h * ref))
        else:
            rh, rw = h, w
        rh = max(32, rh - rh % 32)
        rw = max(32, rw - rw % 32)
        return rh, rw

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_MATTING_MODEL")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        for name in ("modnet.onnx", "modnet_photographic_portrait_matting.onnx", "matting.onnx"):
            candidates.append(models_dir / name)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ.setdefault("TMPDIR", cache_str)
        tempfile.tempdir = cache_str


# SAM ViT-B preprocessing constants (pixel space, 0-255 RGB) and fixed encoder input size.
_SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
_SAM_INPUT_SIZE = 1024


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip() not in ("", "0", "false", "False", "no", "No")


class PersonInstanceSegmenter:
    """Optional accuracy-first person instance masks via Mask DINO Swin-L.

    This backend is deliberately separate from SAM. Mask DINO is the person-identity model: it
    predicts one mask per COCO `person` instance. SAM remains a promptable boundary/refinement
    tool and a fallback path. The adapter is lazy/optional so the app keeps starting normally when
    the research stack (PyTorch + Detectron2 + MaskDINO repo + checkpoint) is not installed.
    """

    PERSON_CLASS_ID = 0  # Detectron2's contiguous COCO id for "person".
    DEFAULT_REPO_DIR = "MaskDINO"
    DEFAULT_CONFIG = "configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"
    DEFAULT_WEIGHTS = "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"

    def __init__(self):
        self.available = False
        self.backend = "maskdino"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._predictor = None
        self._config_path = None
        self._weights_path = None
        self._last_image_key = None
        self._last_instances = None
        try:
            self.score_threshold = float(os.getenv("PORTRAIT_MASKDINO_SCORE_THRESH", "0.25"))
        except (TypeError, ValueError):
            self.score_threshold = 0.25

        if _env_truthy("PORTRAIT_DISABLE_MASKDINO"):
            self.reason_unavailable = "disabled via PORTRAIT_DISABLE_MASKDINO"
            return

        repo_path = self._resolve_repo_path()
        if repo_path is not None:
            self._prepare_repo_import(repo_path)

        config_path = self._resolve_config_path(repo_path)
        weights_path = self._resolve_weights_path()
        if config_path is None:
            self.reason_unavailable = "Mask DINO config not found"
            return
        if weights_path is None:
            self.reason_unavailable = "Mask DINO checkpoint not found"
            return
        self._config_path = config_path
        self._weights_path = weights_path

        try:
            self._predictor = self._build_predictor(config_path, weights_path)
            self.available = True
        except Exception as exc:
            self.reason_unavailable = f"Mask DINO init failed: {exc}"
            self._predictor = None

    def person_masks(
        self,
        img_float: np.ndarray,
        faces: list[tuple[int, int, int, int]],
        subjects_mask: np.ndarray | None = None,
    ) -> list[np.ndarray] | None:
        """Return one person-instance mask per detected face, or None on any unusable result."""
        if not self.available or self._predictor is None or len(faces) == 0:
            return None
        image = to_uint8(img_float)
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None

        image_key = (id(img_float), h, w)
        if self._last_image_key == image_key and self._last_instances is not None:
            instances = self._last_instances
        else:
            try:
                predictions = self._predictor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            except Exception:
                return None
            instances = self._extract_person_instances(predictions, (h, w))
            self._last_image_key = image_key
            self._last_instances = instances
        if not instances:
            return None
        assigned = self._assign_faces_to_instances(faces, instances, (h, w))
        if assigned is None:
            return None

        masks = []
        for mask in assigned:
            work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
            if subjects_mask is not None:
                # MODNet/Subjects sometimes captures hair detail DINO misses at the edge, but it
                # also includes every person. Use it only as a gentle confidence floor inside the
                # DINO instance, not as an expansion source.
                work = work * np.maximum(np.clip(np.asarray(subjects_mask, dtype=np.float32), 0.0, 1.0), 0.6)
            work = np.clip(smooth_mask(work, sigma=max(1.0, w * 0.0025)), 0.0, 1.0).astype(np.float32)
            masks.append(work)
        return masks

    def _build_predictor(self, config_path: Path, weights_path: Path):
        try:
            import torch
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            from maskdino import add_maskdino_config
        except Exception as exc:
            raise RuntimeError(f"required package unavailable: {exc}") from exc

        device = os.getenv("PORTRAIT_MASKDINO_DEVICE", "").strip()
        if not device:
            device = "cuda" if getattr(torch.cuda, "is_available", lambda: False)() else "cpu"
        self.execution_provider = device

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.DEVICE = device
        if hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        try:
            cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = min(
                float(cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD),
                self.score_threshold,
            )
        except Exception:
            pass
        cfg.freeze()
        # Mask DINO prints a huge training-loss dictionary while constructing the model. Suppress
        # that import-time noise so app startup and System Check stay readable.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            return DefaultPredictor(cfg)

    def _extract_person_instances(self, predictions, shape_hw: tuple[int, int]) -> list[dict]:
        h, w = shape_hw
        instances = predictions.get("instances") if isinstance(predictions, dict) else None
        if instances is None:
            return []
        try:
            instances = instances.to("cpu")
        except Exception:
            pass

        try:
            classes = np.asarray(instances.pred_classes)
            scores = np.asarray(instances.scores, dtype=np.float32)
            masks = np.asarray(instances.pred_masks, dtype=np.float32)
            boxes = np.asarray(instances.pred_boxes.tensor, dtype=np.float32)
        except Exception:
            return []
        if masks.ndim != 3:
            return []

        out = []
        for cls, score, mask, box in zip(classes, scores, masks, boxes):
            if int(cls) != self.PERSON_CLASS_ID or float(score) < self.score_threshold:
                continue
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            out.append({
                "mask": (mask > 0.5).astype(np.float32),
                "box": tuple(float(v) for v in box[:4]),
                "score": float(score),
            })
        return out

    @classmethod
    def _assign_faces_to_instances(
        cls,
        faces: list[tuple[int, int, int, int]],
        instances: list[dict],
        shape_hw: tuple[int, int],
    ) -> list[np.ndarray] | None:
        """Greedily assign each detected face box to one unique person instance mask."""
        if not faces or len(instances) < len(faces):
            return None
        h, w = shape_hw
        candidates = []
        for face_idx, face in enumerate(faces):
            for inst_idx, inst in enumerate(instances):
                score = cls._face_instance_score(face, inst, (h, w))
                if score > 0.0:
                    candidates.append((score, face_idx, inst_idx))
        if not candidates:
            return None

        assignments = {}
        used_instances = set()
        for _score, face_idx, inst_idx in sorted(candidates, reverse=True):
            if face_idx in assignments or inst_idx in used_instances:
                continue
            assignments[face_idx] = inst_idx
            used_instances.add(inst_idx)
            if len(assignments) == len(faces):
                break
        if len(assignments) != len(faces):
            return None
        return [np.asarray(instances[assignments[idx]]["mask"], dtype=np.float32) for idx in range(len(faces))]

    @staticmethod
    def _face_instance_score(
        face: tuple[int, int, int, int],
        instance: dict,
        shape_hw: tuple[int, int],
    ) -> float:
        h, w = shape_hw
        fx, fy, fw, fh = face
        x1 = int(np.clip(fx, 0, w))
        y1 = int(np.clip(fy, 0, h))
        x2 = int(np.clip(fx + fw, 0, w))
        y2 = int(np.clip(fy + fh, 0, h))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        mask = np.asarray(instance.get("mask"), dtype=np.float32)
        if mask.shape != (h, w):
            return 0.0
        face_region = mask[y1:y2, x1:x2]
        coverage = float(np.mean(face_region > 0.5)) if face_region.size else 0.0
        cx = int(np.clip(round(fx + fw * 0.5), 0, w - 1))
        cy = int(np.clip(round(fy + fh * 0.5), 0, h - 1))
        center_hit = 1.0 if mask[cy, cx] > 0.5 else 0.0

        box = instance.get("box")
        box_overlap = 0.0
        if box is not None:
            bx1, by1, bx2, by2 = [float(v) for v in box]
            ix1 = max(float(x1), bx1)
            iy1 = max(float(y1), by1)
            ix2 = min(float(x2), bx2)
            iy2 = min(float(y2), by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            box_overlap = inter / max(float((x2 - x1) * (y2 - y1)), 1.0)

        if coverage < 0.03 and center_hit <= 0.0 and box_overlap < 0.35:
            return 0.0
        return coverage * 4.0 + center_hit * 2.0 + box_overlap + float(instance.get("score", 0.0)) * 0.1

    def _resolve_repo_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_MASKDINO_REPO")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / self.DEFAULT_REPO_DIR)
        for path in candidates:
            if path.exists() and path.is_dir():
                return path
        return None

    def _resolve_config_path(self, repo_path: Path | None) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_MASKDINO_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "maskdino_swinl_instance.yaml")
        if repo_path is not None:
            candidates.append(repo_path / self.DEFAULT_CONFIG)
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _resolve_weights_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_MASKDINO_WEIGHTS")
        if env_path:
            candidates.append(Path(env_path))
        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / self.DEFAULT_WEIGHTS)
        candidates.append(models_dir / "maskdino_swinl_instance.pth")
        candidates.append(models_dir / "maskdino_person_instance.pth")
        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    @staticmethod
    def _prepare_repo_import(repo_path: Path) -> None:
        path_str = str(repo_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


class InstanceSegmenter:
    """Optional learned instance masks via Segment Anything (SAM ViT-B, ONNX).

    Unlike the watershed Person split (which geometrically divides the already-computed Subjects
    blob using face boxes), this asks a trained promptable segmenter for each individual directly:
    the heavy image encoder runs once per image (memoized -- the embedding depends only on the
    image, not on which face), then the lightweight mask decoder runs per face. Each face is
    prompted with positive points on its own body and NEGATIVE points on the other detected faces,
    which is what yields a clean per-individual instance even where two bodies touch/overlap -- the
    case watershed cannot solve.

    The same memoized embedding also supports Face boundary refinement via a box/point prompt.
    Falls back silently (available=False) when the encoder/decoder ONNX files or onnxruntime are
    missing, so callers keep using their existing fallback masks unchanged.
    """

    def __init__(self):
        self.available = False
        self.backend = "watershed"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._encoder = None
        self._decoder = None
        self._enc_input_name = None
        self._enc_path = None
        self._coreml_failed = False  # latched once CoreML errors at inference -> use CPU thereafter
        # In-process embedding memo: (h, w, downsample-bytes) -> embedding array. The encoder is
        # the expensive part (~0.5-2s for ViT-B); FaceSegmenter calls segment_with_guides once per
        # face, so without this the encoder would re-run per face on the same image.
        self._emb_sig = None
        self._emb = None

        # Escape hatch: force the watershed Person split even when the SAM models are installed
        # (the ViT-B encoder is heavy/slow, and CoreML can be flaky on it). Also used by the test
        # suite to keep runs fast and deterministic regardless of which models happen to be present.
        if _env_truthy("PORTRAIT_DISABLE_SAM"):
            self.reason_unavailable = "disabled via PORTRAIT_DISABLE_SAM"
            return

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        enc_path, dec_path = self._resolve_model_paths()
        if enc_path is None or dec_path is None:
            self.reason_unavailable = "SAM encoder/decoder model not found"
            return
        self._enc_path = enc_path

        # Both encoder and decoder run on CPU. Measured on this machine: the ViT-B encoder loads
        # in ~0.2s and infers in ~1.6s on CPU, versus the CoreML EP taking ~60s on first run
        # (model compile) AND failing intermittently with "unable to compute the prediction" on
        # ViT-B -- so CPU is both faster and far more reliable here. (CoreML can also wedge the
        # macOS ANE compiler daemon when a compile is interrupted.) Opt back into the accelerated
        # provider with PORTRAIT_SAM_USE_COREML=1 if a future model/runtime makes it worthwhile.
        try:
            if _env_truthy("PORTRAIT_SAM_USE_COREML"):
                providers = self._preferred_onnx_providers(ort)
                self._prepare_coreml_cache_dir(providers)
            else:
                providers = ["CPUExecutionProvider"]
            self._encoder = ort.InferenceSession(str(enc_path), providers=providers)
            self._decoder = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
            self.execution_provider = "coreml" if "CoreMLExecutionProvider" in set(self._encoder.get_providers()) else "cpu"
            self._enc_input_name = self._encoder.get_inputs()[0].name
            self.available = True
            self.backend = "sam"
        except Exception as cpu_exc:
            self.reason_unavailable = f"SAM init failed: {cpu_exc}"

    def _resolve_model_paths(self) -> tuple[Path | None, Path | None]:
        models_dir = Path(__file__).resolve().parents[2] / "models"
        enc_env = os.getenv("PORTRAIT_SAM_ENCODER")
        dec_env = os.getenv("PORTRAIT_SAM_DECODER")
        enc_candidates = ([Path(enc_env)] if enc_env else []) + [
            models_dir / "sam_vit_b_encoder.onnx",
            models_dir / "sam_encoder.onnx",
        ]
        dec_candidates = ([Path(dec_env)] if dec_env else []) + [
            models_dir / "sam_vit_b_decoder.onnx",
            models_dir / "sam_decoder.onnx",
        ]
        enc = next((p for p in enc_candidates if p.exists() and p.is_file()), None)
        dec = next((p for p in dec_candidates if p.exists() and p.is_file()), None)
        return enc, dec

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ.setdefault("TMPDIR", cache_str)
        tempfile.tempdir = cache_str

    def embed(self, img_float: np.ndarray):
        """Run (or reuse) the SAM image encoder. Returns (embedding, scale) where scale maps
        original-image pixel coords into the encoder's resized frame, or None on failure."""
        if not self.available or self._encoder is None:
            return None
        arr = np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return None
        scale = float(_SAM_INPUT_SIZE) / float(max(h, w))

        sig = self._embedding_signature(arr)
        if self._emb is not None and sig == self._emb_sig:
            return self._emb, scale

        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(arr * 255.0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        normalized = (resized - _SAM_PIXEL_MEAN) / _SAM_PIXEL_STD
        padded = np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = normalized
        tensor = np.transpose(padded, (2, 0, 1))[None, ...].astype(np.float32)
        emb = self._run_encoder(tensor)
        if emb is None:
            return None
        self._emb, self._emb_sig = emb, sig
        return emb, scale

    def _run_encoder(self, tensor):
        """Run the encoder, falling back to a CPU session if the CoreML provider errors at
        inference (observed: ViT-B intermittently fails on the CoreML EP with a generic
        'unable to compute the prediction'). On the first such error we latch to CPU for the rest
        of this instance's life -- slower but reliable -- rather than dropping all the way back to
        the watershed split for a mere CoreML hiccup."""
        if not self._coreml_failed:
            try:
                return self._encoder.run(None, {self._enc_input_name: tensor})[0]
            except Exception:
                self._coreml_failed = True  # fall through to CPU
        cpu = self._ensure_cpu_encoder()
        if cpu is None:
            return None
        try:
            return cpu.run(None, {self._enc_input_name: tensor})[0]
        except Exception:
            return None

    def _ensure_cpu_encoder(self):
        if self.execution_provider != "cpu" and self._enc_path is not None:
            try:
                import onnxruntime as ort

                self._encoder = ort.InferenceSession(str(self._enc_path), providers=["CPUExecutionProvider"])
                self._enc_input_name = self._encoder.get_inputs()[0].name
                self.execution_provider = "cpu"
            except Exception:
                return None
        return self._encoder

    @staticmethod
    def _embedding_signature(arr: np.ndarray):
        h, w = arr.shape[:2]
        small = cv2.resize(arr, (32, 32), interpolation=cv2.INTER_AREA)
        return (h, w, small.astype(np.float32).tobytes())

    def instance_masks(self, img_float, faces, subjects_mask):
        """One soft whole-body mask per face, or None if unavailable/failed for the whole image.

        Each face is prompted with positive points along its own body column and negative points
        on every other face; the best (highest predicted IoU) decoder mask is taken, softly gated
        by the Subjects mask to suppress background leak, AND cross-person suppressed (see
        _suppress_cross_person_spill) so two people's masks can't both claim the same territory."""
        if not self.available or self._decoder is None or len(faces) == 0:
            return None
        embedded = self.embed(img_float)
        if embedded is None:
            return None
        emb, scale = embedded
        h, w = np.asarray(img_float).shape[:2]
        subj = np.clip(np.asarray(subjects_mask, dtype=np.float32), 0.0, 1.0)

        body_points = [self._body_points((fx, fy, fw, fh), (h, w), subj) for (fx, fy, fw, fh) in faces]
        raw_masks = []
        for idx in range(len(faces)):
            pos = body_points[idx]
            neg = [pt for j, pts in enumerate(body_points) if j != idx for pt in pts]
            mask = self._decode_one(emb, scale, (h, w), pos, neg)
            if mask is None:
                return None  # one hard failure -> abandon SAM for this image, fall back to watershed
            raw_masks.append(mask)

        raw_masks = _suppress_cross_person_spill(raw_masks)
        masks = []
        for idx, mask in enumerate(raw_masks):
            # SAM can still return a disconnected low-confidence island around another person's
            # hair/head that no other mask claims strongly enough for cross-person suppression to
            # remove. Keep the selected person's anchor-connected component and hard-exclude the
            # other detected heads before smoothing, so those islands don't become visible halos.
            mask = _exclude_other_face_regions(mask, faces, idx, (h, w))
            mask = _keep_anchor_connected_mask(mask, body_points[idx])
            # Soft-gate by subjects so a leak into background can't survive, then feather to match
            # the watershed path's edge treatment.
            mask = mask * np.maximum(subj, 0.15)
            mask = np.clip(smooth_mask(mask, sigma=max(2.0, w * 0.006)), 0.0, 1.0).astype(np.float32)
            # The smoothing/heatmap-visible feather can reintroduce a soft edge into another
            # detected head area. Apply the same hard exclusion after feathering as well.
            mask = _exclude_other_face_regions(mask, faces, idx, (h, w))
            masks.append(mask)
        return masks

    def face_mask(
        self,
        img_float: np.ndarray,
        face_box: tuple[int, int, int, int],
        *,
        guides: dict | None = None,
        other_faces: list[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray | None:
        """One prompted face/head candidate mask for `face_box`, or None on failure.

        This intentionally returns only the raw SAM boundary candidate. Semantic decisions
        (whether it is a plausible Face mask, and how it should gate Skin/Eyes/Lips/Hair) are
        made by FaceSegmenter, where the parser/landmark masks are available.
        """
        if not self.available or self._decoder is None or face_box is None:
            return None
        embedded = self.embed(img_float)
        if embedded is None:
            return None
        emb, scale = embedded
        h, w = np.asarray(img_float).shape[:2]
        points, labels = self._face_prompt(face_box, (h, w), guides=guides, other_faces=other_faces or [])
        return self._decode_prompt(emb, scale, (h, w), points, labels)

    def click_mask(self, img_float: np.ndarray, point_xy: tuple[float, float]) -> np.ndarray | None:
        """One SAM mask for a single positive-point click, or None on failure.

        Manual-editing primitive for "click the object, get its mask" -- no negative points, no
        box, no plausibility gate. Unlike the automatic Person/Face paths (which validate against
        a baseline because their output feeds an unattended pipeline), this is a deliberate,
        user-initiated, Undo-able action: the caller looks at the result and keeps or discards it,
        so no automatic acceptance/rejection belongs here.
        """
        if not self.available or self._decoder is None:
            return None
        embedded = self.embed(img_float)
        if embedded is None:
            return None
        emb, scale = embedded
        h, w = np.asarray(img_float).shape[:2]
        return self._decode_prompt(emb, scale, (h, w), [tuple(point_xy)], [1])

    def _body_points(self, face_box, shape_hw, subjects_mask):
        """Foreground prompt points for one person. SAM needs enough anchors spread over the
        whole body, or it returns just the head (the incomplete-person case seen on the first
        ViT-B run). We seed the face center plus a dense column of torso/leg points AND lateral
        shoulder/hip points, keeping any that fall on the Subjects region under a *loose*
        threshold (>0.3, not >0.5 -- the Subjects mask is itself soft/imperfect over bodies, so a
        strict cutoff was dropping real body points and starving the prompt). The face center is
        always kept as a guaranteed anchor."""
        h, w = shape_hw
        fx, fy, fw, fh = face_box
        cx = fx + fw * 0.5
        # Down the centerline (face -> torso -> legs), plus lateral points to capture body width.
        candidates = [
            (cx, fy + fh * 0.5),
            (cx, fy + fh * 1.2),
            (cx, fy + fh * 1.9),
            (cx, fy + fh * 2.7),
            (cx, fy + fh * 3.6),
            (cx - fw * 0.6, fy + fh * 1.8),
            (cx + fw * 0.6, fy + fh * 1.8),
            (cx - fw * 0.5, fy + fh * 2.8),
            (cx + fw * 0.5, fy + fh * 2.8),
        ]
        on_body = subjects_mask > 0.3
        points = []
        for px, py in candidates:
            ix = int(np.clip(round(px), 0, w - 1))
            iy = int(np.clip(round(py), 0, h - 1))
            if on_body[iy, ix]:
                points.append((float(ix), float(iy)))
        # Always include the face center as a guaranteed anchor (dedup against the first candidate).
        fc = (float(np.clip(round(cx), 0, w - 1)), float(np.clip(round(fy + fh * 0.5), 0, h - 1)))
        if fc not in points:
            points.insert(0, fc)
        return points

    def _face_prompt(
        self,
        face_box: tuple[int, int, int, int],
        shape_hw: tuple[int, int],
        *,
        guides: dict | None = None,
        other_faces: list[tuple[int, int, int, int]] | None = None,
    ) -> tuple[list[tuple[float, float]], list[int]]:
        h, w = shape_hw
        fx, fy, fw, fh = face_box
        cx = fx + fw * 0.5
        cy = fy + fh * 0.5

        # A conservative box gives SAM the expected face extent without inviting hair/clothing.
        x1 = np.clip(fx - fw * 0.15, 0, max(0, w - 1))
        y1 = np.clip(fy - fh * 0.18, 0, max(0, h - 1))
        x2 = np.clip(fx + fw * 1.15, 0, max(0, w - 1))
        y2 = np.clip(fy + fh * 1.14, 0, max(0, h - 1))

        points: list[tuple[float, float]] = []
        labels: list[int] = []
        seen = set()

        def add(px: float, py: float, label: int) -> None:
            ix = float(np.clip(round(px), 0, max(0, w - 1)))
            iy = float(np.clip(round(py), 0, max(0, h - 1)))
            key = (int(ix), int(iy), int(label))
            if key in seen:
                return
            seen.add(key)
            points.append((ix, iy))
            labels.append(int(label))

        add(x1, y1, 2)
        add(x2, y2, 3)
        for px, py in (
            (cx, cy),
            (fx + fw * 0.33, fy + fh * 0.58),
            (fx + fw * 0.67, fy + fh * 0.58),
            (cx, fy + fh * 0.83),
        ):
            add(px, py, 1)

        if guides:
            for key in ("left_eye_center", "right_eye_center", "mouth_center"):
                pt = guides.get(key)
                if not pt or len(pt) < 2:
                    continue
                px, py = float(pt[0]), float(pt[1])
                if (fx - fw * 0.25) <= px <= (fx + fw * 1.25) and (fy - fh * 0.25) <= py <= (fy + fh * 1.25):
                    add(px, py, 1)

        for px, py in (
            (cx, fy - fh * 0.22),
            (cx, fy + fh * 1.22),
            (fx - fw * 0.28, cy),
            (fx + fw * 1.28, cy),
        ):
            add(px, py, 0)

        for ofx, ofy, ofw, ofh in other_faces or []:
            # Skip the current face if the caller passed the full face list unchanged.
            if abs(ofx - fx) <= 1 and abs(ofy - fy) <= 1 and abs(ofw - fw) <= 1 and abs(ofh - fh) <= 1:
                continue
            add(ofx + ofw * 0.5, ofy + ofh * 0.5, 0)

        return points, labels

    def _decode_one(self, emb, scale, shape_hw, pos_points, neg_points):
        points = [tuple(p) for p in pos_points] + [tuple(p) for p in neg_points]
        labels = [1] * len(pos_points) + [0] * len(neg_points)
        return self._decode_prompt(emb, scale, shape_hw, points, labels)

    def _decode_prompt(self, emb, scale, shape_hw, points, labels):
        h, w = shape_hw
        coords = [list(p) for p in points]
        labels = list(labels)
        # The official SAM ONNX path expects a trailing not-a-point padding marker.
        coords.append([0.0, 0.0])
        labels.append(-1)
        point_coords = (np.array(coords, dtype=np.float32) * scale)[None, :, :]
        point_labels = np.array(labels, dtype=np.float32)[None, :]
        decoder_inputs = {
            "image_embeddings": emb.astype(np.float32),
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1,), dtype=np.float32),
            "orig_im_size": np.array([h, w], dtype=np.float32),
        }
        # Only feed inputs the model actually declares (export variants differ slightly in which
        # of these they expose).
        wanted = {i.name for i in self._decoder.get_inputs()}
        feed = {k: v for k, v in decoder_inputs.items() if k in wanted}
        try:
            outputs = self._decoder.run(None, feed)
        except Exception:
            return None
        masks = np.asarray(outputs[0], dtype=np.float32)
        iou = np.asarray(outputs[1], dtype=np.float32).reshape(-1) if len(outputs) > 1 else None
        if masks.ndim == 4:
            cand = masks[0]
        elif masks.ndim == 3:
            cand = masks
        else:
            return None
        best = int(np.argmax(iou)) if iou is not None and iou.size == cand.shape[0] else 0
        logits = cand[best]
        if logits.shape != (h, w):
            logits = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
        soft = 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> soft alpha
        return np.clip(soft, 0.0, 1.0).astype(np.float32)


class FacialHairSegmenter:
    """Optional dedicated facial-hair segmentation on a face crop."""

    def __init__(self):
        self.available = False
        self.backend = "fallback"
        self.reason_unavailable = ""
        self.execution_provider = "cpu"
        self._session = None
        self._input_name = None
        self._input_hw = (256, 256)

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self.reason_unavailable = "facial hair model not found"
            return

        providers = self._preferred_onnx_providers(ort)
        self._prepare_coreml_cache_dir(providers)
        try:
            self._session = ort.InferenceSession(str(model_path), providers=providers)
            active = list(self._session.get_providers())
            self.execution_provider = "coreml" if "CoreMLExecutionProvider" in active else "cpu"
            self._input_name = self._session.get_inputs()[0].name
            self._input_hw = self._infer_input_hw(self._session.get_inputs()[0].shape)
            self.available = True
            self.backend = f"onnx:{self.execution_provider}"
        except Exception as exc:
            if "CoreMLExecutionProvider" in providers:
                try:
                    self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    self.execution_provider = "cpu"
                    self._input_name = self._session.get_inputs()[0].name
                    self._input_hw = self._infer_input_hw(self._session.get_inputs()[0].shape)
                    self.available = True
                    self.backend = "onnx:cpu"
                    self.reason_unavailable = f"coreml unavailable; using cpu ({exc})"
                    return
                except Exception as cpu_exc:
                    self.reason_unavailable = f"facial hair model init failed: {cpu_exc}"
                    return
            self.reason_unavailable = f"facial hair model init failed: {exc}"

    def segment(
        self,
        img_float: np.ndarray,
        *,
        face_box: tuple[int, int, int, int] | None,
        guides: dict | None = None,
        masks: dict | None = None,
    ) -> np.ndarray | None:
        if not self.available or self._session is None or face_box is None:
            return None

        h, w = img_float.shape[:2]
        crop_box = _expanded_face_crop(face_box, (h, w), x_scale=1.08, y_scale=1.18, y_up_scale=0.42)
        x1, y1, x2, y2 = crop_box
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            return None

        crop = to_uint8(img_float[y1:y2, x1:x2])
        in_h, in_w = self._input_hw
        resized = cv2.resize(crop, (in_w, in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        tensor = np.transpose(resized, (2, 0, 1))[None, ...]

        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            self.reason_unavailable = f"facial hair inference failed: {exc}"
            self.available = False
            self.backend = "fallback"
            return None

        crop_mask = self._to_mask(outputs)
        if crop_mask is None:
            return None
        crop_mask = cv2.resize(crop_mask.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        full_mask = np.zeros((h, w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = np.clip(crop_mask, 0.0, 1.0)

        fallback = _facial_hair_mask(img_float, face_box=face_box, masks=masks or _empty_masks(h, w), guides=guides)
        merged = np.clip(np.maximum(full_mask, fallback * 0.20), 0.0, 1.0).astype(np.float32)
        return merged

    def _resolve_model_path(self) -> Path | None:
        candidates = []
        env_path = os.getenv("PORTRAIT_FACIAL_HAIR_ONNX")
        if env_path:
            candidates.append(Path(env_path))

        models_dir = Path(__file__).resolve().parents[2] / "models"
        candidates.append(models_dir / "facial_hair.onnx")
        candidates.append(models_dir / "beard_mustache.onnx")

        for path in candidates:
            if path.exists() and path.is_file():
                return path
        return None

    def _preferred_onnx_providers(self, ort) -> list[str]:
        available = set(ort.get_available_providers())
        providers = []
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _prepare_coreml_cache_dir(self, providers: list[str]) -> None:
        if "CoreMLExecutionProvider" not in providers:
            return
        cache_dir = Path(__file__).resolve().parents[2] / ".ort_coreml_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_str = str(cache_dir)
        os.environ["TMPDIR"] = cache_str
        os.environ["TMP"] = cache_str
        os.environ["TEMP"] = cache_str
        tempfile.tempdir = cache_str

    def _infer_input_hw(self, shape) -> tuple[int, int]:
        dims = list(shape or [])
        if len(dims) >= 4:
            h = dims[-2]
            w = dims[-1]
        elif len(dims) == 3:
            h = dims[-2]
            w = dims[-1]
        else:
            return 256, 256
        if not isinstance(h, int) or h <= 0:
            h = 256
        if not isinstance(w, int) or w <= 0:
            w = 256
        return int(h), int(w)

    def _to_mask(self, output) -> np.ndarray | None:
        if isinstance(output, (list, tuple)):
            yolo_mask = self._decode_yolov8_seg_output(list(output))
            if yolo_mask is not None:
                if not self.backend.endswith(":yolov8-seg"):
                    self.backend = f"onnx:{self.execution_provider}:yolov8-seg"
                return yolo_mask
            if not output:
                return None
            arr = np.asarray(output[0])
        else:
            arr = np.asarray(output)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            mask = arr.astype(np.float32)
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 2) and arr.shape[-1] not in (1, 2):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                mask = arr[:, :, 0].astype(np.float32)
            elif arr.shape[-1] == 2:
                probs = _softmax_last_axis(arr.astype(np.float32))
                mask = probs[:, :, 1]
            else:
                labels = np.argmax(arr, axis=-1)
                mask = (labels > 0).astype(np.float32)
        else:
            return None

        if mask.min() < 0.0 or mask.max() > 1.0:
            if mask.min() >= 0.0 and mask.max() > 1.5:
                unique_count = int(np.unique(mask.astype(np.int32)).size)
                if unique_count <= 8:
                    mask = (mask > 0.5).astype(np.float32)
                else:
                    mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float32)))
            else:
                mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float32)))
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _decode_yolov8_seg_output(self, outputs: list[np.ndarray]) -> np.ndarray | None:
        if len(outputs) < 2:
            return None

        pred = None
        proto = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 4:
                proto = arr
            elif arr.ndim in (2, 3):
                pred = arr
        if pred is None or proto is None:
            return None

        proto_arr = np.asarray(proto)
        while proto_arr.ndim > 3 and proto_arr.shape[0] == 1:
            proto_arr = proto_arr[0]
        if proto_arr.ndim != 3:
            return None
        proto_c, mask_h, mask_w = proto_arr.shape

        pred_arr = np.asarray(pred)
        while pred_arr.ndim > 2 and pred_arr.shape[0] == 1:
            pred_arr = pred_arr[0]
        if pred_arr.ndim != 2:
            return None
        if pred_arr.shape[0] <= proto_c + 16:
            pred_arr = pred_arr.T
        if pred_arr.ndim != 2:
            return None

        attrs = pred_arr.shape[1]
        class_count = attrs - 4 - proto_c
        if class_count <= 0:
            return None

        cls_scores = pred_arr[:, 4 : 4 + class_count]
        scores = cls_scores.max(axis=1)
        if scores.size == 0:
            return None
        keep = np.where(scores > 0.20)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(scores))], dtype=np.int32)

        coeffs = pred_arr[keep, 4 + class_count : 4 + class_count + proto_c]
        boxes = pred_arr[keep, :4]
        selected_scores = scores[keep]
        proto_flat = proto_arr.reshape(proto_c, -1)
        mask_logits = coeffs @ proto_flat
        mask_stack = 1.0 / (1.0 + np.exp(-mask_logits.astype(np.float32)))
        mask_stack = mask_stack.reshape(len(keep), mask_h, mask_w)

        input_h, input_w = self._input_hw
        scale_x = mask_w / max(float(input_w), 1.0)
        scale_y = mask_h / max(float(input_h), 1.0)
        yy, xx = np.mgrid[0:mask_h, 0:mask_w]
        aggregate = np.zeros((mask_h, mask_w), dtype=np.float32)
        for idx, mask in enumerate(mask_stack):
            cx, cy, bw, bh = boxes[idx]
            x1 = np.clip((cx - bw * 0.5) * scale_x, 0.0, mask_w - 1)
            y1 = np.clip((cy - bh * 0.5) * scale_y, 0.0, mask_h - 1)
            x2 = np.clip((cx + bw * 0.5) * scale_x, x1 + 1.0, mask_w)
            y2 = np.clip((cy + bh * 0.5) * scale_y, y1 + 1.0, mask_h)
            box_mask = ((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).astype(np.float32)
            weighted = np.clip(mask * box_mask * float(np.clip(selected_scores[idx], 0.0, 1.0)), 0.0, 1.0)
            aggregate = np.maximum(aggregate, weighted)
        return np.clip(aggregate, 0.0, 1.0).astype(np.float32)


class FaceSegmenter:
    """
    Runtime wrapper selecting model-based segmentation when available.

    Backends:
    - auto: use model backend if ready, otherwise heuristic fallback
    - model: force model backend if available, otherwise fallback with reason
    - heuristic: always use heuristic backend
    """

    def __init__(self, backend: str = "auto", model_path: str | None = None):
        self._heuristic = HeuristicFaceSegmenter()
        self._model = ModelFaceSegmenter(model_path=model_path) if backend in ("auto", "model") else None
        self._background_removal = BackgroundRemovalSegmenter()
        self._matting = MattingSegmenter()
        self._subjects = SubjectSegmenter()
        self._facial_hair = FacialHairSegmenter()
        self._person_instances = PersonInstanceSegmenter()
        self._instance = InstanceSegmenter()

        self.backend = "heuristic"
        self.backend_label = "heuristic"
        self.detector_backend = getattr(self._heuristic, "detector_backend", "haar")
        self.detector_backend_label = f"detect={self.detector_backend}"
        if self._background_removal is not None and getattr(self._background_removal, "available", False):
            self.subject_backend = self._background_removal.backend
        elif self._matting is not None and getattr(self._matting, "available", False):
            self.subject_backend = self._matting.backend
        else:
            self.subject_backend = getattr(self._subjects, "backend", "heuristic")
        self.subject_backend_label = f"subjects={self.subject_backend}"
        self.facial_hair_backend = getattr(self._facial_hair, "backend", "fallback")
        self.facial_hair_backend_label = f"f_hair={self.facial_hair_backend}"
        self.face_part_backend = "semantic"
        self.face_part_backend_label = "parts=semantic"
        # Which backend produced the most recent Person mask: "sam" (learned instance masks) or
        # "watershed" (the geometric split of the Subjects blob). Set by _person_mask; surfaced so
        # the UI/status can say which path ran rather than presenting both as equivalent.
        self.person_backend = "watershed"
        self.person_backend_label = f"person={self.person_backend}"
        # Which backend produced/refined the active Face mask. SAM is a boundary refinement on
        # top of the model/heuristic semantic mask, so rejected/unavailable SAM leaves this at the
        # base backend.
        self.face_backend = "heuristic"
        self.face_backend_label = "face=heuristic"
        self.reason_unavailable = ""
        # "" = no split attempted (0-1 faces) or N/A; "normal"/"low" set by the most recent
        # _person_mask call -- "low" means the detected faces are close enough together
        # (e.g. a parent holding a child) that the watershed boundary is a rougher
        # approximation than usual. Surfaced so the UI can say so, not silently presented as
        # ground truth.
        self.person_split_confidence = ""

        if backend == "heuristic":
            return

        if self._model is not None and self._model.available:
            self.backend = "model"
            self.backend_label = self._model_backend_label()
            self.face_backend = "model"
            self.face_backend_label = "face=model"
            return

        if self._model is not None:
            self.reason_unavailable = self._model.reason_unavailable
            self.backend = "heuristic"
            self.backend_label = "heuristic (fallback)"

    def face_guides(self, img_float: np.ndarray, face_box: tuple[int, int, int, int] | None = None) -> dict | None:
        """Landmark-only guides (eye/mouth points) -- skips the parsing + matting pipeline that
        segment_with_guides runs, so it's far cheaper when a caller (e.g. blink detection) only
        needs landmark coordinates. Returns None if no model landmarker is available."""
        if self._model is not None and getattr(self._model, "landmark_backend", "none") != "none":
            try:
                _landmarks, guides = self._model._landmark_masks(img_float, face_box=face_box)
                return guides
            except Exception:
                return None
        return None

    def list_faces(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        self.detector_backend = getattr(self._heuristic, "detector_backend", "haar")
        self.detector_backend_label = f"detect={self.detector_backend}"
        if self._background_removal is not None and getattr(self._background_removal, "available", False):
            self.subject_backend = self._background_removal.backend
        elif self._matting is not None and getattr(self._matting, "available", False):
            self.subject_backend = self._matting.backend
        else:
            self.subject_backend = getattr(self._subjects, "backend", "heuristic")
        self.subject_backend_label = f"subjects={self.subject_backend}"
        self.facial_hair_backend = getattr(self._facial_hair, "backend", "fallback")
        self.facial_hair_backend_label = f"f_hair={self.facial_hair_backend}"
        self.face_part_backend = "semantic"
        self.face_part_backend_label = "parts=semantic"
        base_face_backend = "model" if self.backend == "model" else "heuristic"
        self.face_backend = base_face_backend
        self.face_backend_label = f"face={base_face_backend}"
        return self._heuristic.list_faces(img_float)

    def segment(self, img_float: np.ndarray, face_index: int | None = 0) -> dict:
        masks, _guides = self.segment_with_guides(img_float, face_index=face_index)
        return masks

    def segment_with_guides(self, img_float: np.ndarray, face_index: int | None = 0) -> tuple[dict, dict | None]:
        faces = self.list_faces(img_float)
        focus_mask = _focus_mask_from_faces(img_float.shape[:2], faces, face_index)

        if self.backend == "model" and self._model is not None:
            try:
                target_face = faces[_safe_face_index(face_index, len(faces))] if faces else None
                masks, guides = self._model.segment_with_guides(img_float, face_box=target_face)
                masks = self._refine_face_masks_with_sam(img_float, faces, face_index, masks, guides)
                masks["facial_hair"] = self._facial_hair_mask(img_float, target_face, masks, guides)
                if focus_mask is not None:
                    masks = _apply_focus_mask(masks, focus_mask)
                masks["subjects"], masks["background"] = self._scene_subject_masks(img_float, faces, masks)
                person_mask = self._person_mask(img_float, faces, face_index, masks["subjects"])
                masks["person"] = person_mask
                masks = self._apply_person_identity_gate_to_parts(masks, person_mask)
                return masks, guides
            except Exception as exc:
                self.reason_unavailable = f"model runtime error: {exc}"
                self.backend = "heuristic"
                self.backend_label = "heuristic (runtime fallback)"

        masks = self._heuristic.segment(img_float, face_index=face_index)
        target_face = faces[_safe_face_index(face_index, len(faces))] if faces else None
        masks = self._refine_face_masks_with_sam(img_float, faces, face_index, masks, None)
        masks["facial_hair"] = self._facial_hair_mask(img_float, target_face, masks, None)
        masks["subjects"], masks["background"] = self._scene_subject_masks(img_float, faces, masks)
        person_mask = self._person_mask(img_float, faces, face_index, masks["subjects"])
        masks["person"] = person_mask
        masks = self._apply_person_identity_gate_to_parts(masks, person_mask)
        return masks, None

    def _model_backend_label(self) -> str:
        if self._model is None:
            return "model"
        provider = getattr(self._model, "execution_provider", "cpu")
        landmark_backend = getattr(self._model, "landmark_backend", "none")
        if landmark_backend == "mediapipe.tasks.face_landmarker":
            return f"model (onnx:{provider} + tasks-landmarks)"
        if landmark_backend != "none":
            return f"model (onnx:{provider} + mesh-landmarks)"
        return f"model (onnx:{provider} only)"

    def _scene_subject_masks(self, img_float: np.ndarray, faces: list[tuple[int, int, int, int]], masks: dict) -> tuple[np.ndarray, np.ndarray]:
        heuristic_subjects, heuristic_background = _subjects_background_masks(img_float.shape[:2], faces, masks)

        # Backend preference: RMBG-2.0/BiRefNet background removal (best general foreground
        # separation) -> MODNet matting (portrait alpha) -> MediaPipe selfie segmenter (256px)
        # -> elliptical heuristic.
        model_subjects = None
        backend = "heuristic"
        if self._background_removal is not None and self._background_removal.available:
            try:
                model_subjects = self._background_removal.segment(img_float)
            except Exception:
                model_subjects = None
            if model_subjects is not None:
                backend = self._background_removal.backend
        if model_subjects is None and self._matting is not None and self._matting.available:
            try:
                model_subjects = self._matting.segment(img_float)
            except Exception:
                model_subjects = None
            if model_subjects is not None:
                backend = self._matting.backend
        if model_subjects is None and self._subjects is not None and self._subjects.available:
            try:
                model_subjects = self._subjects.segment(img_float)
            except Exception:
                model_subjects = None
            if model_subjects is not None:
                backend = self._subjects.backend

        if model_subjects is None:
            self.subject_backend = "heuristic"
            self.subject_backend_label = "subjects=heuristic"
            return heuristic_subjects, heuristic_background

        model_subjects = np.clip(model_subjects.astype(np.float32), 0.0, 1.0)
        # Fill interior holes (e.g. dark hair/clothing the model under-confidently scores)
        # using only the reliable per-pixel feature masks -- face/skin/hair -- NOT the
        # blobby elliptical body guesses, which previously leaked subject onto background.
        feature_union = np.zeros_like(model_subjects)
        for key in ("face", "skin", "hair"):
            value = masks.get(key)
            if value is not None:
                feature_union = np.maximum(feature_union, np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0))

        if backend == "rmbg2":
            person_union = self._person_instance_union(img_float, faces, np.maximum(model_subjects, heuristic_subjects))
            if person_union is not None:
                # RMBG is deliberately foreground-generic: in family/table scenes it can include
                # plates, bottles, or chairs. Mask DINO is person-specific, so use it as a soft
                # keep-region while still preserving the face/hair semantic union.
                keep = np.maximum(person_union, feature_union)
                keep = np.clip(smooth_mask(keep, sigma=max(2.0, img_float.shape[1] * 0.004)), 0.0, 1.0)
                model_subjects = model_subjects * np.clip(keep * 1.25, 0.0, 1.0)

        merged_subjects = np.maximum(model_subjects, feature_union)
        # Snap the boundary to real image edges instead of blurring it into a soft blob.
        merged_subjects = _guided_refine_mask(merged_subjects, img_float)
        merged_background = np.clip(1.0 - merged_subjects, 0.0, 1.0).astype(np.float32)
        self.subject_backend = backend
        self.subject_backend_label = f"subjects={backend}"
        return merged_subjects, merged_background

    def _person_instance_union(
        self,
        img_float: np.ndarray,
        faces: list[tuple[int, int, int, int]],
        subjects_hint: np.ndarray,
    ) -> np.ndarray | None:
        if not faces or self._person_instances is None or not getattr(self._person_instances, "available", False):
            return None
        try:
            person_masks = self._person_instances.person_masks(img_float, faces, subjects_hint)
        except Exception:
            return None
        if not person_masks:
            return None
        union = np.zeros(img_float.shape[:2], dtype=np.float32)
        for mask in person_masks:
            if mask is None:
                continue
            arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
            if arr.shape != union.shape:
                arr = cv2.resize(arr, (union.shape[1], union.shape[0]), interpolation=cv2.INTER_LINEAR)
            union = np.maximum(union, arr)
        if float(np.mean(union > 0.10)) < 0.002:
            return None
        return union.astype(np.float32)

    def _apply_person_identity_gate_to_parts(self, masks: dict, person_mask: np.ndarray) -> dict:
        gated = _apply_person_identity_gate_to_face_parts(
            masks,
            person_mask,
            person_backend=self.person_backend,
        )
        if gated is masks:
            self.face_part_backend = "semantic"
            self.face_part_backend_label = "parts=semantic"
        else:
            self.face_part_backend = "person-gated"
            self.face_part_backend_label = "parts=person-gated"
        return gated

    def _refine_face_masks_with_sam(
        self,
        img_float: np.ndarray,
        faces: list[tuple[int, int, int, int]],
        face_index: int | None,
        masks: dict[str, np.ndarray],
        guides: dict | None,
    ) -> dict[str, np.ndarray]:
        base_backend = "model" if self.backend == "model" else "heuristic"
        self.face_backend = base_backend
        self.face_backend_label = f"face={base_backend}"

        if not faces:
            return masks
        idx = _safe_face_index(face_index, len(faces))
        target_face = faces[idx]

        if _env_truthy("PORTRAIT_DISABLE_SAM_FACE"):
            return masks
        if self._instance is None or not getattr(self._instance, "available", False):
            return masks
        if not hasattr(self._instance, "face_mask"):
            return masks

        try:
            sam_mask = self._instance.face_mask(
                img_float,
                target_face,
                guides=guides,
                other_faces=faces,
            )
        except Exception:
            sam_mask = None

        zones = _region_zones(np.asarray(img_float).shape[:2], target_face)
        candidate, reason = _validate_sam_face_mask(
            sam_mask,
            base_face=masks.get("face", np.zeros(np.asarray(img_float).shape[:2], dtype=np.float32)),
            hair_mask=masks.get("hair"),
            zones=zones,
            face_box=target_face,
            guides=guides,
        )
        if candidate is None:
            if sam_mask is not None:
                self.face_backend = f"{base_backend} ({reason})"
                self.face_backend_label = f"face={base_backend}*"
            return masks

        self.face_backend = "sam"
        self.face_backend_label = "face=sam"
        return _merge_sam_face_masks(masks, candidate, img_float, target_face)

    def _person_mask(
        self,
        img_float: np.ndarray,
        faces: list[tuple[int, int, int, int]],
        face_index: int | None,
        subjects_mask: np.ndarray,
    ) -> np.ndarray:
        """Per-person full-body mask for `face_index`. Prefers learned SAM instance masks
        (InstanceSegmenter), but with a per-person plausibility net: SAM occasionally returns a
        truncated mask for one individual (just the head), so the cheap watershed split is always
        computed as a baseline and used for that person when SAM's mask is implausibly small
        relative to it -- "SAM where it's confident, watershed otherwise". Aliases `subjects_mask`
        unchanged -- "Person 1 IS Subjects", not a missing mask -- when there are fewer than 2
        faces, or the watershed split fails `_validate_person_split`."""
        shape_hw = np.asarray(img_float).shape[:2]
        h, w = shape_hw
        subjects_mask = np.clip(np.asarray(subjects_mask, dtype=np.float32), 0.0, 1.0)
        if len(faces) < 2:
            self.person_split_confidence = ""
            self.person_backend = "subjects"
            self.person_backend_label = "person=subjects"
            return subjects_mask

        idx = _safe_face_index(face_index, len(faces))

        # Always compute the watershed split -- it's cheap, and serves as both the fallback and
        # the plausibility yardstick for the SAM mask. Keep the raw labels too: even when the
        # full split is rejected as a fallback, the selected basin is still a useful guardrail for
        # clipping SAM away from another detected person's head/hair.
        raw_label_map = _watershed_person_labels(shape_hw, faces, subjects_mask)
        total_area = float(np.sum(subjects_mask > 0.5))
        label_map, confidence = _validate_person_split(raw_label_map, faces, total_area)
        if label_map is not None:
            ws_mask = np.clip(
                smooth_mask((label_map == idx + 1).astype(np.float32), sigma=max(2.0, w * 0.006)),
                0.0, 1.0,
            ).astype(np.float32)
        else:
            ws_mask = subjects_mask  # split rejected -> Person aliases Subjects

        # Accuracy-first path: a real person instance segmenter identifies all COCO `person`
        # instances, then maps each detected face to one instance mask. This is the right model
        # class for person identity; SAM is promptable boundary segmentation and remains the next
        # fallback/refinement path.
        person_instances = getattr(self, "_person_instances", None)
        if person_instances is not None and getattr(person_instances, "available", False):
            try:
                masks = person_instances.person_masks(img_float, faces, subjects_mask)
            except Exception:
                masks = None
            if masks is not None and len(masks) == len(faces):
                person_mask = np.clip(masks[idx], 0.0, 1.0).astype(np.float32)
                person_area = float(np.sum(person_mask > 0.5))
                face_area = float(faces[idx][2] * faces[idx][3])
                if person_area > max(25.0, face_area * 0.35):
                    self.person_backend = "maskdino"
                    self.person_backend_label = "person=maskdino"
                    self.person_split_confidence = ""
                    return person_mask

        # Preferred fallback path: learned promptable SAM masks. Any failure (model absent, decode
        # error) returns None and we keep the watershed result.
        if self._instance is not None and getattr(self._instance, "available", False):
            try:
                instances = self._instance.instance_masks(img_float, faces, subjects_mask)
            except Exception:
                instances = None
            if instances is not None and len(instances) == len(faces):
                sam_mask = np.clip(instances[idx], 0.0, 1.0).astype(np.float32)
                gate_label_map = label_map if label_map is not None else raw_label_map
                gated_by_watershed = False
                if gate_label_map is not None:
                    gated = _gate_sam_person_mask_to_watershed(sam_mask, gate_label_map, idx, shape_hw)
                    if _is_plausible_gated_sam_person_mask(sam_mask, gated):
                        sam_mask = gated
                        gated_by_watershed = True
                sam_area = float(np.sum(sam_mask > 0.5))
                ws_area = float(np.sum(ws_mask > 0.5))
                # SAM truncated to (roughly) the head shows up as a mask far smaller than the
                # watershed body region -- below half its area, prefer the watershed mask for this
                # person so a partial mask never reaches the user. Only use a validated watershed
                # split for this area test; a rejected split can still be a gate, but not a
                # trustworthy size baseline.
                if label_map is not None and ws_area > 0 and sam_area < 0.5 * ws_area:
                    self.person_backend = "watershed (sam-incomplete)"
                    self.person_backend_label = "person=watershed*"
                    self.person_split_confidence = confidence
                    return ws_mask
                self.person_backend = "sam (watershed-gated)" if gated_by_watershed else "sam"
                self.person_backend_label = "person=sam+ws" if gated_by_watershed else "person=sam"
                self.person_split_confidence = confidence if label_map is not None else ""
                return sam_mask

        self.person_backend = "watershed"
        self.person_backend_label = "person=watershed"
        self.person_split_confidence = confidence
        return ws_mask

    def _facial_hair_mask(
        self,
        img_float: np.ndarray,
        face_box: tuple[int, int, int, int] | None,
        masks: dict,
        guides: dict | None,
    ) -> np.ndarray:
        fallback = _facial_hair_mask(img_float, face_box=face_box, masks=masks, guides=guides)
        if self._facial_hair is None or not self._facial_hair.available:
            self.facial_hair_backend = "fallback"
            self.facial_hair_backend_label = "f_hair=fallback"
            return fallback

        try:
            model_mask = self._facial_hair.segment(img_float, face_box=face_box, guides=guides, masks=masks)
        except Exception:
            model_mask = None
        if model_mask is None:
            self.facial_hair_backend = "fallback"
            self.facial_hair_backend_label = "f_hair=fallback"
            return fallback

        self.facial_hair_backend = self._facial_hair.backend
        self.facial_hair_backend_label = f"f_hair={self.facial_hair_backend}"
        return np.clip(np.maximum(model_mask.astype(np.float32), fallback * 0.15), 0.0, 1.0).astype(np.float32)


def _safe_face_index(face_index: int | None, count: int) -> int:
    if count <= 0:
        return 0
    if face_index is None:
        return 0
    return int(np.clip(int(face_index), 0, count - 1))


def _focus_mask_from_faces(shape_hw, faces: list[tuple[int, int, int, int]], face_index: int | None):
    h, w = shape_hw
    if len(faces) == 0:
        return None

    idx = _safe_face_index(face_index, len(faces))
    fx, fy, fw, fh = faces[idx]

    focus = np.zeros((h, w), dtype=np.float32)
    cx = fx + fw // 2
    cy = fy + fh // 2
    cv2.ellipse(
        focus,
        (cx, cy),
        (max(1, int(fw * 0.95)), max(1, int(fh * 1.15))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    return np.clip(smooth_mask(focus, sigma=max(2.0, fw * 0.04)), 0.0, 1.0)


def _expanded_face_crop(
    face_box: tuple[int, int, int, int],
    shape_hw: tuple[int, int],
    *,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    y_up_scale: float | None = None,
) -> tuple[int, int, int, int]:
    h, w = shape_hw
    fx, fy, fw, fh = face_box
    pad_x = fw * float(x_scale)
    pad_y_down = fh * float(y_scale)
    pad_y_up = fh * float(y_scale if y_up_scale is None else y_up_scale)
    x1 = max(0, int(np.floor(fx - pad_x)))
    y1 = max(0, int(np.floor(fy - pad_y_up)))
    x2 = min(w, int(np.ceil(fx + fw + pad_x)))
    y2 = min(h, int(np.ceil(fy + fh + pad_y_down)))
    return x1, y1, x2, y2


def _paste_crop_masks(
    crop_masks: dict[str, np.ndarray],
    crop_box: tuple[int, int, int, int] | None,
    full_shape_hw: tuple[int, int],
    crop_shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    full_h, full_w = full_shape_hw
    if crop_box is None:
        return {key: value.astype(np.float32) for key, value in crop_masks.items()}

    x1, y1, x2, y2 = crop_box
    crop_h, crop_w = crop_shape_hw
    if (y2 - y1) != crop_h or (x2 - x1) != crop_w:
        # Defensive resize if a caller provides masks with mismatched geometry.
        resized_masks = {}
        for key, value in crop_masks.items():
            resized_masks[key] = cv2.resize(value.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        crop_masks = resized_masks

    out = _empty_masks(full_h, full_w)
    for key, value in crop_masks.items():
        out[key][y1:y2, x1:x2] = np.clip(value, 0.0, 1.0).astype(np.float32)
    return out


def _map_crop_guides_to_full(guides: dict | None, crop_box: tuple[int, int, int, int], scale: float) -> dict | None:
    if not guides:
        return None
    x1, y1, _x2, _y2 = crop_box
    denom = max(float(scale), 1e-6)
    mapped = {}
    for key, value in guides.items():
        if value is None or len(value) < 2:
            continue
        mapped[key] = (
            float(x1 + float(value[0]) / denom),
            float(y1 + float(value[1]) / denom),
        )
    return mapped


def _normalize_detector_faces(detections, shape_hw: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = shape_hw
    if detections is None:
        return []
    detections = np.asarray(detections)
    if detections.size == 0:
        return []
    if detections.ndim == 1:
        detections = detections[None, :]

    faces = []
    for row in detections:
        if len(row) < 4:
            continue
        x, y, fw, fh = row[:4]
        x = int(np.clip(np.floor(x), 0, max(0, w - 1)))
        y = int(np.clip(np.floor(y), 0, max(0, h - 1)))
        fw = int(max(1, min(w - x, np.ceil(fw))))
        fh = int(max(1, min(h - y, np.ceil(fh))))
        faces.append((x, y, fw, fh))
    return sorted(faces, key=lambda r: r[2] * r[3], reverse=True)


def _apply_focus_mask(masks: dict, focus_mask: np.ndarray) -> dict:
    if focus_mask is None:
        return masks

    out = {}
    for key, mask in masks.items():
        if key in {"subjects", "background"}:
            out[key] = np.clip(mask, 0.0, 1.0).astype(np.float32)
        else:
            out[key] = np.clip(mask * focus_mask, 0.0, 1.0).astype(np.float32)
    return out


def _apply_person_identity_gate_to_face_parts(
    masks: dict,
    person_mask: np.ndarray | None,
    *,
    person_backend: str = "",
) -> dict:
    """Clamp face-part masks to the selected person instance.

    Skin/eyes/lips/hair are semantic face-part masks, so the parser decides "what" each pixel is.
    The selected Person mask decides "whose" pixel it is. Applying this as a final ownership gate
    prevents a feature layer for Face 2 from retaining Face 1 hair/skin/lip islands when the
    semantic parser or geometric fallback was too broad.
    """
    if person_mask is None:
        return masks
    backend = str(person_backend or "").lower()
    if backend.startswith("subjects"):
        return masks

    gate_src = np.clip(np.asarray(person_mask, dtype=np.float32), 0.0, 1.0)
    if gate_src.ndim != 2 or gate_src.size == 0:
        return masks
    if float(np.sum(gate_src > 0.15)) < 25.0:
        return masks

    h, w = gate_src.shape[:2]
    pad = min(32, max(3, int(round(min(h, w) * 0.004))))
    gate_binary = _fill_mask_holes((gate_src > 0.15).astype(np.uint8)).astype(np.float32)
    gate = _dilate_soft_mask(gate_binary, pixels=pad)
    if float(np.sum(gate > 0.5)) < 25.0:
        return masks

    out = dict(masks)
    changed = False
    for key in ("face", "skin", "eyes", "lips", "hair", "brows", "facial_hair"):
        value = masks.get(key)
        if value is None:
            continue
        mask = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
        if mask.shape != gate.shape:
            continue
        out[key] = np.clip(mask * gate, 0.0, 1.0).astype(np.float32)
        changed = True
    return out if changed else masks


def _empty_masks(h, w):
    return {k: np.zeros((h, w), dtype=np.float32) for k in MASK_KEYS}


def _guided_refine_mask(
    mask: np.ndarray,
    guide_img: np.ndarray,
    *,
    radius_frac: float = 0.012,
    eps: float = 1e-4,
) -> np.ndarray:
    """Snap a soft/coarse mask to real image edges with a guided filter, using the
    full-resolution photo as the guide. The subject segmentation model runs at 256x256
    and is upscaled ~20x, so its raw boundary is a smooth blob that cuts into hair and
    misses the silhouette; guided filtering re-aligns the boundary to actual luminance/
    color edges in the image without the edge-destroying Gaussian blur used before."""
    src = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    h, w = src.shape[:2]
    if h == 0 or w == 0:
        return src
    guide = (np.clip(np.asarray(guide_img, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    if guide.shape[:2] != (h, w):
        guide = cv2.resize(guide, (w, h), interpolation=cv2.INTER_LINEAR)
    radius = max(2, int(round(min(h, w) * float(radius_frac))))
    try:
        refined = cv2.ximgproc.guidedFilter(guide, src, radius, float(eps))
    except Exception:
        # ximgproc not built in this OpenCV -- fall back to a mild edge-preserving smooth.
        return np.clip(smooth_mask(src, sigma=max(2.0, w * 0.004)), 0.0, 1.0).astype(np.float32)
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def _subjects_background_masks(shape_hw, faces: list[tuple[int, int, int, int]], masks: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape_hw
    subjects = np.zeros((h, w), dtype=np.float32)

    for fx, fy, fw, fh in faces:
        cx = fx + fw * 0.5
        body_cy = fy + fh * 2.05
        body = _ellipse_mask(shape_hw, (cx, body_cy), (fw * 1.35, fh * 2.75))
        shoulders = _ellipse_mask(shape_hw, (cx, fy + fh * 1.28), (fw * 1.55, fh * 0.88))
        torso = _ellipse_mask(shape_hw, (cx, fy + fh * 1.85), (fw * 1.18, fh * 1.75))
        subjects = np.maximum(subjects, np.clip(body * 0.55 + shoulders * 0.75 + torso * 0.85, 0.0, 1.0))

    feature_union = np.zeros((h, w), dtype=np.float32)
    for key, weight in (
        ("face", 1.0),
        ("skin", 0.9),
        ("hair", 0.95),
        ("eyes", 0.55),
        ("lips", 0.55),
    ):
        if key in masks and masks[key] is not None:
            feature_union = np.maximum(feature_union, np.clip(np.asarray(masks[key], dtype=np.float32) * weight, 0.0, 1.0))

    if faces:
        subjects = np.maximum(subjects, feature_union)
    else:
        subjects = np.maximum(subjects, feature_union * 0.85)

    subjects = np.clip(smooth_mask(subjects, sigma=max(2.0, w * 0.010)), 0.0, 1.0).astype(np.float32)
    background = np.clip(1.0 - subjects, 0.0, 1.0).astype(np.float32)
    background = np.clip(smooth_mask(background, sigma=max(2.0, w * 0.010)), 0.0, 1.0).astype(np.float32)
    subjects = np.clip(1.0 - background, 0.0, 1.0).astype(np.float32)
    return subjects, background


def _watershed_person_labels(
    shape_hw: tuple[int, int],
    faces: list[tuple[int, int, int, int]],
    subjects_mask: np.ndarray,
) -> np.ndarray | None:
    """Split the binary subjects mask into one region per face, by running watershed over
    the mask's own (negated) distance transform, seeded with one marker per face center plus
    a dilated background marker. This is the standard technique for splitting touching/
    overlapping blobs at their narrowest connection (used for touching-cell separation in
    microscopy): the boundary follows the silhouette's own shape, not a straight line between
    face centers.

    NOTE on the inherent limitation: where two people's bodies truly overlap in 2D (one partly
    in front of another, or a child fully inside a parent's silhouette with no "neck" to cut
    at), watershed still returns a complete partition -- it does not degrade gracefully -- so
    the boundary can run straight through a person's body in that case. There is no way to do
    better than this nearest-seed-along-the-silhouette approximation without true occlusion-
    aware instance segmentation. Callers must run `_validate_person_split` on the result before
    trusting it; this function alone makes no quality judgement.
    """
    h, w = shape_hw
    if h <= 0 or w <= 0 or len(faces) < 2:
        return None

    binary = (np.asarray(subjects_mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    if not np.any(binary):
        return None

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Watershed floods "downhill" from markers; feeding it the *inverted* distance transform
    # means basins are deepest at each seed and ridges form along the mask's narrowest
    # connections (its true geometric "necks") -- not at the raw Euclidean midpoint between
    # two face centers, which would ignore the silhouette shape entirely.
    landscape = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)

    markers = np.zeros((h, w), dtype=np.int32)
    background_label = len(faces) + 1
    # Dilate the "outside the mask" region inward before marking it sure-background, so the
    # background marker never directly touches a foreground marker -- OpenCV's watershed
    # otherwise draws a spurious thin boundary wherever two different markers are adjacent.
    outside = cv2.dilate((binary == 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    markers[outside > 0] = background_label

    # Seed each person with a BODY region -- a head ellipse plus a torso ellipse extending down
    # from the face -- clipped to the subjects blob, NOT a tiny dot at the face center. A face
    # center sits near the top edge of the body silhouette, where the distance transform (hence
    # the watershed landscape) is a ridge, not a basin: a dot there floods only the head, and the
    # background marker claims the rest of the body, leaving most of each person unassigned
    # (measured 66% of the subjects area unclaimed on a real 2-person photo; ~10% with this
    # body seed). The downward-body assumption holds for upright portrait subjects.
    fg = binary > 0
    seed_regions = []
    for (fx, fy, fw, fh) in faces:
        cx = fx + fw * 0.5
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(region, (int(cx), int(fy + fh * 0.5)), (max(1, int(fw * 0.45)), max(1, int(fh * 0.5))), 0, 0, 360, 1, -1)
        cv2.ellipse(region, (int(cx), int(fy + fh * 2.2)), (max(1, int(fw * 0.6)), max(1, int(fh * 2.2))), 0, 0, 360, 1, -1)
        seed_regions.append((region.astype(bool) & fg).astype(np.uint8))

    # Where two people's seed capsules overlap, the assignment is genuinely ambiguous -- keep
    # only the pixels each capsule owns alone (claimed by exactly one), eroded so no seed abuts
    # the background marker or another seed (which would make watershed draw a spurious boundary
    # there). If ANY face's solo region comes out empty -- the capsules overlap so heavily that
    # one person is swallowed, the signature of faces very close together (a parent holding a
    # child) -- the body-capsule heuristic has broken down; fall back to uniform face-center dots
    # for ALL faces, the safe symmetric baseline that gives neither person an unfair seed (and
    # which _validate_person_split then flags low-confidence rather than trusting outright).
    claimed_count = np.stack(seed_regions, axis=0).sum(axis=0) if seed_regions else np.zeros((h, w))
    solo_regions = [
        cv2.erode(((region > 0) & (claimed_count == 1)).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        for region in seed_regions
    ]
    if all(np.any(solo) for solo in solo_regions):
        for idx, solo in enumerate(solo_regions):
            markers[solo > 0] = idx + 1
    else:
        seed_radius = max(2, int(round(min(h, w) * 0.01)))
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            cx = int(np.clip(fx + fw * 0.5, 0, w - 1))
            cy = int(np.clip(fy + fh * 0.5, 0, h - 1))
            cv2.circle(markers, (cx, cy), seed_radius, idx + 1, -1)

    cv2.watershed(landscape, markers)

    label_map = np.where(binary > 0, markers, 0)
    # Boundary pixels (-1) and anything that leaked the background label are not a person.
    label_map = np.where((label_map < 1) | (label_map > len(faces)), 0, label_map)
    return label_map.astype(np.int32)


def _validate_person_split(
    label_map: np.ndarray | None,
    faces: list[tuple[int, int, int, int]],
    total_area: float,
) -> tuple[np.ndarray | None, str]:
    """Sanity-check a watershed person split before trusting it. Returns (None, "") to mean
    "don't split -- alias Person to Subjects for this image" when the split looks unreliable,
    or (label_map, confidence) where confidence is "normal" or "low" (faces close enough
    together -- e.g. a parent holding a child -- that the boundary is a rougher approximation
    than usual, surfaced so the UI can say so rather than presenting it as ground truth).
    """
    if label_map is None or not faces or total_area <= 0:
        return None, ""

    n = len(faces)
    fair_share = total_area / n
    areas = [float(np.sum(label_map == idx + 1)) for idx in range(n)]
    # A person ending up with far less than their "fair share" of the total subjects area is
    # the signature of a failed split (one seed's basin swallowed almost everything) -- abandon
    # the whole split rather than show a degenerate sliver mask for that person.
    if any(area < fair_share * 0.15 for area in areas):
        return None, ""

    widths = [float(fw) for _fx, _fy, fw, _fh in faces]
    avg_width = sum(widths) / n if widths else 0.0
    centers = [(fx + fw * 0.5, fy + fh * 0.5) for fx, fy, fw, fh in faces]
    min_pair_dist = None
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = float(np.hypot(dx, dy))
            if min_pair_dist is None or dist < min_pair_dist:
                min_pair_dist = dist
    confidence = "normal"
    if avg_width > 0 and min_pair_dist is not None and min_pair_dist < avg_width * 1.5:
        confidence = "low"
    return label_map, confidence


def _suppress_cross_person_spill(raw_masks: list[np.ndarray]) -> list[np.ndarray]:
    """Enforce mutual exclusivity across independently-decoded per-person SAM masks.

    Unlike the watershed split (a hard partition by construction -- a pixel can only carry one
    label, so it structurally cannot spill), each person's SAM mask here is decoded in its own
    separate prompt call. The negative points on every *other* person's body discourage bleed but
    don't guarantee it: through contact/occlusion or a weak prompt, one person's mask can still
    claim territory that another person's own (independently prompted) mask claims more
    confidently -- the "person 2 spilling onto person 1" symptom. Where that happens, the
    less-confident claimant loses that pixel entirely; each person keeps the territory only they
    confidently claim. A single face's list (nothing to compare against) passes through unchanged.
    """
    if len(raw_masks) < 2:
        return list(raw_masks)
    stack = np.stack([np.clip(np.asarray(m, dtype=np.float32), 0.0, 1.0) for m in raw_masks], axis=0)
    winner = np.argmax(stack, axis=0)
    out = []
    for idx, mask in enumerate(stack):
        # Only suppress where this face has a real (non-trivial) claim AND someone else's claim
        # is stronger -- a pixel neither face claims confidently is left alone here (the existing
        # Subjects gate and smoothing handle ambiguous/background territory separately).
        contested = (mask > 0.3) & (winner != idx)
        out.append(np.where(contested, 0.0, mask).astype(np.float32))
    return out


def _exclude_other_face_regions(
    mask: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    current_idx: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Remove other detected heads from a per-person SAM mask.

    The Person layer should include the selected person's hair, but never another detected face's
    hair/head. This catches low-confidence SAM halos that are disconnected or too weak for the
    mutual-overlap suppressor to notice.
    """
    if len(faces) < 2:
        return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0).astype(np.float32)
    h, w = shape_hw
    exclusion = np.zeros((h, w), dtype=np.uint8)
    for idx, (fx, fy, fw, fh) in enumerate(faces):
        if idx == current_idx:
            continue
        cx = fx + fw * 0.5
        # Face boxes usually sit on the face, while the visible spill is most obvious around
        # hair above/around that box. Use overlapping head+hair ovals rather than the face box
        # alone so the other person's hair outline is protected too.
        head = _ellipse_mask((h, w), (cx, fy + fh * 0.48), (fw * 1.05, fh * 0.92))
        hair_cap = _ellipse_mask((h, w), (cx, fy + fh * 0.08), (fw * 1.08, fh * 0.48))
        side_hair = _ellipse_mask((h, w), (cx, fy + fh * 0.33), (fw * 1.18, fh * 0.72))
        exclusion = np.maximum(exclusion, (np.maximum.reduce([head, hair_cap, side_hair]) > 0.03).astype(np.uint8))
    if not np.any(exclusion):
        return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0).astype(np.float32)
    pad = max(3, int(round(min(h, w) * 0.006)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    exclusion = cv2.dilate(exclusion, kernel, iterations=1).astype(bool)
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return np.where(exclusion, 0.0, work).astype(np.float32)


def _gate_sam_person_mask_to_watershed(
    sam_mask: np.ndarray,
    label_map: np.ndarray | None,
    current_idx: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Constrain a SAM person candidate to the selected watershed basin.

    SAM gives better boundaries, but in crowded photos it can still carry a selected person mask
    across into another detected head/hair region. The watershed split is a rough partition, yet
    it is useful as a guardrail: the selected SAM mask may extend a little beyond its basin for
    edge tolerance, but it cannot keep a distant halo assigned to another face.
    """
    work = np.clip(np.asarray(sam_mask, dtype=np.float32), 0.0, 1.0)
    if label_map is None or work.ndim != 2 or work.size == 0:
        return work.astype(np.float32)
    labels = np.asarray(label_map)
    if labels.shape != work.shape:
        return work.astype(np.float32)

    h, w = shape_hw
    if (h, w) != work.shape:
        h, w = work.shape
    basin = (labels == int(current_idx) + 1).astype(np.uint8)
    if not np.any(basin):
        return work.astype(np.float32)

    pad = max(6, int(round(min(h, w) * 0.012)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    gate = cv2.dilate(basin, kernel, iterations=1).astype(bool)
    return np.where(gate, work, 0.0).astype(np.float32)


def _is_plausible_gated_sam_person_mask(original: np.ndarray, gated: np.ndarray) -> bool:
    """Return whether a watershed-gated SAM mask retained enough of the original claim.

    This protects against a rejected/degenerate watershed split cutting SAM down to a tiny sliver.
    A large reduction is acceptable -- that is exactly what removes another person's head/hair --
    but the selected person's main region must still survive.
    """
    src = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    out = np.clip(np.asarray(gated, dtype=np.float32), 0.0, 1.0)
    if src.shape != out.shape or src.ndim != 2:
        return False
    src_area = float(np.sum(src > 0.5))
    out_area = float(np.sum(out > 0.5))
    if src_area <= 0 or out_area <= 0:
        return False
    return out_area >= max(25.0, src_area * 0.08)


def _keep_anchor_connected_mask(
    mask: np.ndarray,
    anchors: list[tuple[float, float]],
    *,
    threshold: float = 0.25,
) -> np.ndarray:
    """Keep only the SAM component connected to the selected person's prompt anchors."""
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if work.ndim != 2 or work.size == 0 or not anchors:
        return work.astype(np.float32)
    binary = (work > float(threshold)).astype(np.uint8)
    if not np.any(binary):
        return work.astype(np.float32)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return work.astype(np.float32)

    h, w = work.shape[:2]
    keep_labels = set()
    for px, py in anchors:
        ix = int(np.clip(round(float(px)), 0, w - 1))
        iy = int(np.clip(round(float(py)), 0, h - 1))
        label = int(labels[iy, ix])
        if label > 0:
            keep_labels.add(label)

    if not keep_labels:
        # If every anchor lands just outside the thresholded soft mask, preserve the largest
        # component rather than dropping the person entirely. The plausibility check in _person_mask
        # still catches truly truncated SAM masks against the watershed baseline.
        areas = [(int(stats[label, cv2.CC_STAT_AREA]), label) for label in range(1, count)]
        keep_labels.add(max(areas)[1])

    keep = np.isin(labels, list(keep_labels)).astype(np.uint8)
    pad = max(1, int(round(min(h, w) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    keep = cv2.dilate(keep, kernel, iterations=1).astype(bool)
    return np.where(keep, work, 0.0).astype(np.float32)


def _validate_sam_face_mask(
    sam_mask: np.ndarray | None,
    *,
    base_face: np.ndarray,
    hair_mask: np.ndarray | None,
    zones: dict[str, np.ndarray],
    face_box: tuple[int, int, int, int],
    guides: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Validate a prompted SAM face candidate before it can affect user-visible masks."""
    if sam_mask is None:
        return None, "sam-unavailable"

    candidate = np.clip(np.asarray(sam_mask, dtype=np.float32), 0.0, 1.0)
    base = np.clip(np.asarray(base_face, dtype=np.float32), 0.0, 1.0)
    if candidate.ndim != 2 or candidate.shape != base.shape:
        return None, "sam-unavailable"

    binary = candidate > 0.5
    sam_area = float(np.sum(binary))
    if sam_area <= 0:
        return None, "sam-empty"

    _fx, _fy, fw, fh = face_box
    face_box_area = max(1.0, float(fw * fh))
    base_area = float(np.sum(base > 0.35))
    if base_area > 0 and sam_area < base_area * 0.35:
        return None, "sam-too-small"
    if base_area <= 0 and sam_area < face_box_area * 0.15:
        return None, "sam-too-small"
    if sam_area > face_box_area * 2.25:
        return None, "sam-too-large"

    if not _sam_face_has_target_anchor(candidate, face_box, guides):
        return None, "sam-off-target"

    zone_face = np.clip(np.asarray(zones.get("face", np.zeros_like(candidate)), dtype=np.float32), 0.0, 1.0)
    zone_iou = _binary_iou(candidate > 0.35, zone_face > 0.20)
    if zone_iou < 0.30:
        return None, "sam-low-overlap"
    if base_area > 0:
        base_iou = _binary_iou(candidate > 0.35, base > 0.20)
        if base_iou < 0.25:
            return None, "sam-low-overlap"

    if hair_mask is not None:
        hair = np.clip(np.asarray(hair_mask, dtype=np.float32), 0.0, 1.0)
        if hair.shape == candidate.shape:
            hair_overlap = float(np.sum(binary & (hair > 0.55)))
            if hair_overlap / max(sam_area, 1.0) > 0.45:
                return None, "sam-hair-heavy"

    return candidate.astype(np.float32), "normal"


def _sam_face_has_target_anchor(
    mask: np.ndarray,
    face_box: tuple[int, int, int, int],
    guides: dict | None,
) -> bool:
    h, w = mask.shape[:2]

    def sample(px: float, py: float) -> float:
        ix = int(np.clip(round(px), 0, w - 1))
        iy = int(np.clip(round(py), 0, h - 1))
        return float(mask[iy, ix])

    fx, fy, fw, fh = face_box
    if sample(fx + fw * 0.5, fy + fh * 0.5) > 0.35:
        return True

    if not guides:
        return False
    hits = 0
    for key in ("left_eye_center", "right_eye_center", "mouth_center"):
        pt = guides.get(key)
        if pt and len(pt) >= 2 and sample(float(pt[0]), float(pt[1])) > 0.35:
            hits += 1
    return hits >= 2


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    union = float(np.sum(a_bool | b_bool))
    if union <= 0:
        return 0.0
    return float(np.sum(a_bool & b_bool)) / union


def _merge_sam_face_masks(
    masks: dict[str, np.ndarray],
    sam_mask: np.ndarray,
    guide_img: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Use an accepted SAM mask as a face boundary gate while keeping semantic parts."""
    out = {key: np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0).copy() for key, value in masks.items()}
    h, w = guide_img.shape[:2]
    zones = _region_zones((h, w), face_box)

    base_face = out.get("face", np.zeros((h, w), dtype=np.float32))
    skin = out.get("skin", np.zeros((h, w), dtype=np.float32))
    eyes = out.get("eyes", np.zeros((h, w), dtype=np.float32))
    lips = out.get("lips", np.zeros((h, w), dtype=np.float32))
    brows = out.get("brows", np.zeros((h, w), dtype=np.float32))
    hair = out.get("hair", np.zeros((h, w), dtype=np.float32))

    sam_gate = _edge_aware_region_refine(np.clip(sam_mask, 0.0, 1.0), guide_img, "face")
    sam_gate = np.clip(smooth_mask(sam_gate, sigma=max(0.8, w * 0.003)), 0.0, 1.0).astype(np.float32)
    dilated_gate = _dilate_soft_mask(sam_gate, pixels=max(2, int(round(face_box[2] * 0.04))))

    core_semantic = np.maximum.reduce([skin, eyes * 0.85, lips * 0.85, brows * 0.70])
    face_refined = np.maximum(core_semantic, np.minimum(base_face, np.maximum(sam_gate, dilated_gate)))
    growth = np.clip(sam_gate * zones["face"] * (1.0 - hair * 0.75), 0.0, 1.0)
    face_refined = np.maximum(face_refined, growth * 0.85)
    face_refined = _edge_aware_region_refine(np.clip(face_refined, 0.0, 1.0), guide_img, "face")
    face_refined = np.clip(face_refined, 0.0, 1.0).astype(np.float32)
    out["face"] = face_refined

    clamp = _dilate_soft_mask(face_refined, pixels=max(2, int(round(face_box[2] * 0.05))))
    if "skin" in out:
        out["skin"] = np.clip(out["skin"] * clamp, 0.0, 1.0).astype(np.float32)
    for key in ("eyes", "lips", "brows"):
        if key in out:
            out[key] = np.clip(np.minimum(out[key], clamp), 0.0, 1.0).astype(np.float32)
    if "hair" in out:
        out["hair"] = np.clip(out["hair"] * (1.0 - 0.25 * face_refined), 0.0, 1.0).astype(np.float32)
    return out


def _dilate_soft_mask(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    src = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if src.ndim != 2 or src.size == 0:
        return src
    k = max(1, int(pixels) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return np.clip(cv2.dilate(src, kernel, iterations=1), 0.0, 1.0).astype(np.float32)


def _mask_from_labels(labels: np.ndarray, target_labels: set[int]) -> np.ndarray:
    return np.isin(labels, list(target_labels)).astype(np.float32)


def _mask_from_class_probs(class_probs: np.ndarray, target_labels: set[int]) -> np.ndarray:
    if class_probs.ndim != 3 or not target_labels:
        return np.zeros(class_probs.shape[:2], dtype=np.float32)
    valid_indices = [idx for idx in sorted(target_labels) if 0 <= idx < class_probs.shape[-1]]
    if not valid_indices:
        return np.zeros(class_probs.shape[:2], dtype=np.float32)
    return np.clip(np.sum(class_probs[:, :, valid_indices], axis=-1), 0.0, 1.0).astype(np.float32)


def _build_model_masks_from_output(labels: np.ndarray, class_probs: np.ndarray | None) -> dict[str, np.ndarray]:
    if class_probs is not None:
        return {
            "skin": _mask_from_class_probs(class_probs, {1}),
            "brows": _mask_from_class_probs(class_probs, {2, 3}),
            "facial_hair": np.zeros(class_probs.shape[:2], dtype=np.float32),
            "eyes": _mask_from_class_probs(class_probs, {4, 5}),
            "lips": _mask_from_class_probs(class_probs, {11, 12, 13}),
            "hair": _mask_from_class_probs(class_probs, {17}),
            "face": _mask_from_class_probs(class_probs, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
        }
    return {
        "skin": _mask_from_labels(labels, {1}),
        "brows": _mask_from_labels(labels, {2, 3}),
        "facial_hair": np.zeros(labels.shape[:2], dtype=np.float32),
        "eyes": _mask_from_labels(labels, {4, 5}),
        "lips": _mask_from_labels(labels, {11, 12, 13}),
        "hair": _mask_from_labels(labels, {17}),
        "face": _mask_from_labels(labels, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
    }


def _merge_model_views(
    full_masks: dict[str, np.ndarray],
    crop_masks: dict[str, np.ndarray],
    *,
    face_box: tuple[int, int, int, int] | None,
    shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    if face_box is None:
        return {key: np.maximum(full_masks[key], crop_masks[key]).astype(np.float32) for key in full_masks}

    zones = _region_zones(shape_hw, face_box)
    detail_zone = np.clip(zones["face"] * 0.65 + zones["eyes"] * 0.35 + zones["lips"] * 0.35, 0.0, 1.0)
    hair_zone = np.clip(zones["hair"], 0.0, 1.0)

    merged = {}
    for key in full_masks:
        full = np.clip(full_masks[key], 0.0, 1.0)
        crop = np.clip(crop_masks.get(key, full), 0.0, 1.0)
        if key == "hair":
            crop_weight = hair_zone * 0.45
        else:
            crop_weight = detail_zone * 0.75 + zones.get(key, detail_zone) * 0.15
        crop_weight = np.clip(crop_weight, 0.0, 0.9)
        merged[key] = np.clip(full * (1.0 - crop_weight) + crop * crop_weight, 0.0, 1.0).astype(np.float32)
    return merged


def _postprocess_model_masks(
    masks: dict[str, np.ndarray],
    guide_img: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    shape_hw: tuple[int, int],
) -> dict[str, np.ndarray]:
    h, w = shape_hw
    zones = _region_zones(shape_hw, face_box)
    width_ref = max(1, face_box[2] if face_box is not None else w)
    min_area_face = max(32, int(width_ref * width_ref * 0.01))
    min_area_skin = max(32, int(width_ref * width_ref * 0.008))
    min_area_brows = max(8, int(width_ref * width_ref * 0.0005))
    min_area_eye = max(8, int(width_ref * width_ref * 0.0005))
    min_area_lips = max(12, int(width_ref * width_ref * 0.001))
    min_area_hair = max(64, int(width_ref * width_ref * 0.01))

    face = _zone_fallback(_cleanup_region_mask(masks["face"], zone=zones["face"], min_area=min_area_face, close_size=7, open_size=3, keep_largest=1), zones["face"], face_box)
    brows = _zone_fallback(_cleanup_region_mask(masks.get("brows", np.zeros((h, w), dtype=np.float32)), zone=zones["eyes"], min_area=min_area_brows, close_size=3, open_size=1, keep_largest=2), zones["eyes"], face_box)
    eyes = _zone_fallback(_cleanup_region_mask(masks["eyes"], zone=zones["eyes"], min_area=min_area_eye, close_size=3, open_size=1, keep_largest=2), zones["eyes"], face_box)
    lips = _zone_fallback(_cleanup_region_mask(masks["lips"], zone=zones["lips"], min_area=min_area_lips, close_size=5, open_size=1, keep_largest=1), zones["lips"], face_box)
    hair = _zone_fallback(_cleanup_region_mask(masks["hair"], zone=zones["hair"], min_area=min_area_hair, close_size=9, open_size=3, keep_largest=2), zones["hair"], face_box)
    skin = _zone_fallback(_cleanup_region_mask(masks["skin"], zone=zones["skin"], min_area=min_area_skin, close_size=7, open_size=3, keep_largest=1), zones["skin"], face_box)

    # Make masks less mutually contaminating before the final smoothing pass.
    skin = np.clip(skin * (1.0 - 0.85 * eyes) * (1.0 - 0.75 * lips) * (1.0 - 0.45 * hair), 0.0, 1.0)
    face = np.maximum(face, skin)
    face = np.clip(face * (1.0 - 0.15 * hair), 0.0, 1.0)
    hair = np.clip(np.maximum(hair, zones["hair"] * hair * 0.25), 0.0, 1.0)

    face = _edge_aware_region_refine(face, guide_img, "face")
    skin = _edge_aware_region_refine(skin, guide_img, "skin")
    brows = _edge_aware_region_refine(brows, guide_img, "eyes")
    hair = _edge_aware_region_refine(hair, guide_img, "hair")
    eyes = _edge_aware_region_refine(eyes, guide_img, "eyes")
    lips = _edge_aware_region_refine(lips, guide_img, "lips")

    return {
        "face": face.astype(np.float32),
        "skin": skin.astype(np.float32),
        "brows": brows.astype(np.float32),
        "eyes": eyes.astype(np.float32),
        "lips": lips.astype(np.float32),
        "hair": hair.astype(np.float32),
    }


def _refine_face_part_masks(
    masks: dict[str, np.ndarray],
    guide_img: np.ndarray,
    *,
    face_box: tuple[int, int, int, int] | None,
    guides: dict | None = None,
) -> dict[str, np.ndarray]:
    if face_box is None:
        return masks

    h, w = guide_img.shape[:2]
    zones = _region_zones((h, w), face_box)
    out = {key: np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0).copy() for key, value in masks.items()}
    guide_support = _guide_face_part_support((h, w), face_box, guides)

    for key in ("eyes", "lips", "brows"):
        support = guide_support.get(key)
        if support is None or key not in out:
            continue
        zone = np.clip(zones.get("eyes" if key == "brows" else key, support), 0.0, 1.0)
        out[key] = _merge_feature_with_anatomy_support(out[key], support, zone, guide_img, key)

    face = np.clip(out.get("face", zones["face"]), 0.0, 1.0)
    eyes = np.clip(out.get("eyes", np.zeros((h, w), dtype=np.float32)), 0.0, 1.0)
    lips = np.clip(out.get("lips", np.zeros((h, w), dtype=np.float32)), 0.0, 1.0)
    brows = np.clip(out.get("brows", np.zeros((h, w), dtype=np.float32)), 0.0, 1.0)
    hair = np.clip(out.get("hair", np.zeros((h, w), dtype=np.float32)), 0.0, 1.0)

    if "skin" in out:
        skin = np.clip(out["skin"], 0.0, 1.0)
        face_area = float(np.sum(face > 0.25))
        skin_area = float(np.sum(skin > 0.25))
        skin_support = np.clip(face * zones["skin"], 0.0, 1.0)
        skin_support *= np.clip(1.0 - 0.90 * eyes, 0.0, 1.0)
        skin_support *= np.clip(1.0 - 0.85 * lips, 0.0, 1.0)
        skin_support *= np.clip(1.0 - 0.55 * brows, 0.0, 1.0)
        skin_support *= np.clip(1.0 - 0.40 * hair, 0.0, 1.0)
        if face_area > 0 and skin_area < face_area * 0.35:
            skin = np.maximum(skin, skin_support * 0.85)
        else:
            skin = np.maximum(skin, skin_support * 0.20)
        out["skin"] = _edge_aware_region_refine(np.clip(skin, 0.0, 1.0), guide_img, "skin")

    if "hair" in out:
        skin_for_hair = np.clip(out.get("skin", np.zeros((h, w), dtype=np.float32)), 0.0, 1.0)
        hair_likelihood = _head_hair_likelihood_mask(guide_img)
        hair_zone = np.clip(zones["hair"], 0.0, 1.0)
        face_guard = _dilate_soft_mask(
            np.maximum(face * 0.70, skin_for_hair),
            pixels=max(1, int(round(face_box[2] * 0.018))),
        )
        non_face_weight = np.clip(1.0 - 0.82 * skin_for_hair - 0.58 * face_guard, 0.0, 1.0)
        _fx, fy, _fw, fh = face_box
        yy = np.arange(h, dtype=np.float32)[:, None]
        upper_hair_relief = np.clip((fy + fh * 0.56 - yy) / max(fh * 0.36, 1.0), 0.0, 1.0)
        non_face_weight = np.maximum(non_face_weight, upper_hair_relief * np.clip(1.0 - 0.35 * skin_for_hair, 0.0, 1.0))
        hair_candidate = np.clip(hair_likelihood * hair_zone * non_face_weight, 0.0, 1.0)
        hair = np.clip(hair * non_face_weight, 0.0, 1.0)
        hair_area = float(np.sum(hair > 0.25))
        zone_area = float(np.sum(hair_zone > 0.25))
        if zone_area > 0 and hair_area < zone_area * 0.18:
            hair = np.maximum(hair, hair_candidate * 0.95)
        else:
            hair = np.maximum(hair, hair_candidate * 0.35)
        hair = np.clip(hair * (1.0 - 0.65 * eyes) * (1.0 - 0.55 * lips), 0.0, 1.0)
        out["hair"] = _edge_aware_region_refine(hair, guide_img, "hair")

    return {key: np.clip(value, 0.0, 1.0).astype(np.float32) for key, value in out.items()}


def _merge_feature_with_anatomy_support(
    mask: np.ndarray,
    support: np.ndarray,
    zone: np.ndarray,
    guide_img: np.ndarray,
    key: str,
) -> np.ndarray:
    work = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    support = np.clip(np.asarray(support, dtype=np.float32), 0.0, 1.0)
    zone = np.clip(np.asarray(zone, dtype=np.float32), 0.0, 1.0)
    if work.shape != support.shape or work.shape != zone.shape:
        return work.astype(np.float32)

    pad = max(1, int(round(min(work.shape[:2]) * 0.004)))
    clamp = np.clip(_dilate_soft_mask(support, pixels=pad), 0.0, 1.0)
    refined = np.maximum(np.minimum(work, clamp), support * 0.85)
    refined = np.clip(refined * np.maximum(zone, support), 0.0, 1.0)
    refine_region = "eyes" if key in {"eyes", "brows"} else key
    refined = _edge_aware_region_refine(refined, guide_img, refine_region)
    return np.clip(refined * clamp, 0.0, 1.0).astype(np.float32)


def _guide_face_part_support(
    shape_hw: tuple[int, int],
    face_box: tuple[int, int, int, int],
    guides: dict | None,
) -> dict[str, np.ndarray]:
    if not guides:
        return {}
    h, w = shape_hw
    fx, fy, fw, fh = face_box

    def pt(name: str) -> tuple[float, float] | None:
        value = guides.get(name)
        if not value or len(value) < 2:
            return None
        return float(value[0]), float(value[1])

    out: dict[str, np.ndarray] = {}
    left_eye = _eye_support_from_guides(shape_hw, pt("left_eye_center"), pt("left_eye_upper"), pt("left_eye_lower"), fw, fh)
    right_eye = _eye_support_from_guides(shape_hw, pt("right_eye_center"), pt("right_eye_upper"), pt("right_eye_lower"), fw, fh)
    if left_eye is not None or right_eye is not None:
        out["eyes"] = np.maximum(
            left_eye if left_eye is not None else np.zeros((h, w), dtype=np.float32),
            right_eye if right_eye is not None else np.zeros((h, w), dtype=np.float32),
        ).astype(np.float32)

    mouth_left = pt("mouth_left")
    mouth_right = pt("mouth_right")
    mouth_upper = pt("mouth_upper")
    mouth_lower = pt("mouth_lower")
    mouth_center = pt("mouth_center")
    if mouth_left and mouth_right and mouth_upper and mouth_lower:
        cx = (mouth_left[0] + mouth_right[0] + mouth_upper[0] + mouth_lower[0]) * 0.25
        cy = (mouth_left[1] + mouth_right[1] + mouth_upper[1] + mouth_lower[1]) * 0.25
        lip_w = max(abs(mouth_right[0] - mouth_left[0]), fw * 0.14)
        lip_h = max(abs(mouth_lower[1] - mouth_upper[1]), fh * 0.035)
        out["lips"] = _ellipse_mask(shape_hw, (cx, cy), (lip_w * 0.56, lip_h * 0.88)).astype(np.float32)
    elif mouth_center:
        out["lips"] = _ellipse_mask(shape_hw, mouth_center, (fw * 0.14, fh * 0.055)).astype(np.float32)

    brow_masks = []
    for name in ("left_brow", "right_brow"):
        brow = pt(name)
        if brow:
            brow_masks.append(_ellipse_mask(shape_hw, brow, (fw * 0.13, fh * 0.035)))
    if brow_masks:
        out["brows"] = np.maximum.reduce(brow_masks).astype(np.float32)

    return out


def _eye_support_from_guides(
    shape_hw: tuple[int, int],
    center: tuple[float, float] | None,
    upper: tuple[float, float] | None,
    lower: tuple[float, float] | None,
    face_w: int,
    face_h: int,
) -> np.ndarray | None:
    if center is None:
        if upper is None or lower is None:
            return None
        center = ((upper[0] + lower[0]) * 0.5, (upper[1] + lower[1]) * 0.5)
    vertical = abs(lower[1] - upper[1]) if upper and lower else face_h * 0.035
    rx = max(face_w * 0.105, vertical * 2.4)
    ry = max(face_h * 0.028, vertical * 1.15)
    return _ellipse_mask(shape_hw, center, (rx, ry)).astype(np.float32)


def _head_hair_likelihood_mask(img_float: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(img_float, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros(arr.shape[:2], dtype=np.float32)
    lum = np.clip(arr[:, :, 0] * 0.2126 + arr[:, :, 1] * 0.7152 + arr[:, :, 2] * 0.0722, 0.0, 1.0)
    dark = np.clip((0.62 - lum) / 0.50, 0.0, 1.0)
    texture_fine = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), sigmaX=1.2))
    texture_coarse = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), sigmaX=4.0))
    texture = np.clip(texture_fine * 5.5 + texture_coarse * 2.8, 0.0, 1.0)
    try:
        hsv = cv2.cvtColor(to_uint8(arr), cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = np.clip(hsv[:, :, 1] / 255.0, 0.0, 1.0)
    except Exception:
        sat = np.zeros_like(lum, dtype=np.float32)
    hairlike = np.clip(dark * 0.72 + texture * 0.38 + sat * dark * 0.22, 0.0, 1.0)
    return np.clip(smooth_mask(hairlike, sigma=1.0), 0.0, 1.0).astype(np.float32)


def _zone_fallback(mask: np.ndarray, zone: np.ndarray, face_box: tuple[int, int, int, int] | None) -> np.ndarray:
    """If the model (and any landmark assist) produced no usable signal for this feature at
    all, fall back to the face-geometry zone itself rather than leaving the layer completely
    blank with nothing to view or paint over. Confirmed on a real 2-face photo: the parsing
    model's own confidence for the smaller/farther face's eyes never cleared the cleanup
    threshold, and that face had no matching landmark set either (see _select_landmark_set),
    so "eyes" came out all-zero even though a face was clearly detected there."""
    if face_box is None or mask.sum() > 0:
        return mask
    return np.clip(zone, 0.0, 1.0).astype(np.float32)


def _cleanup_region_mask(
    mask: np.ndarray,
    *,
    zone: np.ndarray | None = None,
    min_area: int = 16,
    close_size: int = 3,
    open_size: int = 0,
    keep_largest: int = 0,
) -> np.ndarray:
    work = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if zone is not None:
        work = work * np.clip(zone.astype(np.float32), 0.0, 1.0)
    binary = (work > 0.35).astype(np.uint8)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = _fill_mask_holes(binary)
    binary = _filter_connected_components(binary, min_area=min_area, keep_largest=keep_largest)
    return binary.astype(np.float32)


def _filter_connected_components(binary_mask: np.ndarray, min_area: int = 16, keep_largest: int = 0) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.ndim != 2 or binary_mask.size == 0 or int(binary_mask.max()) == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if count <= 1:
        return binary_mask.astype(np.uint8)

    keep_ids = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            keep_ids.append((area, label))
    if not keep_ids:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    keep_ids.sort(reverse=True)
    if keep_largest and keep_largest > 0:
        keep_ids = keep_ids[:keep_largest]
    keep_label_ids = {label for _area, label in keep_ids}
    return np.isin(labels, list(keep_label_ids)).astype(np.uint8)


def _fill_mask_holes(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.ndim != 2 or binary_mask.size == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    flood = (binary_mask * 255).copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary_mask * 255, holes)
    return (filled > 0).astype(np.uint8)


def _region_zones(shape_hw: tuple[int, int], face_box: tuple[int, int, int, int] | None) -> dict[str, np.ndarray]:
    h, w = shape_hw
    if face_box is None:
        return {
            "face": np.ones((h, w), dtype=np.float32),
            "skin": np.ones((h, w), dtype=np.float32),
            "eyes": _oval_zone(h, w, 0.5, 0.38, 0.20, 0.08),
            "lips": _oval_zone(h, w, 0.5, 0.62, 0.16, 0.09),
            "hair": _oval_zone(h, w, 0.5, 0.28, 0.36, 0.20),
        }

    fx, fy, fw, fh = face_box
    cx = fx + fw / 2.0
    cy = fy + fh / 2.0

    face = _ellipse_mask(shape_hw, (cx, cy), (fw * 0.64, fh * 0.74))
    skin = _ellipse_mask(shape_hw, (cx, fy + fh * 0.55), (fw * 0.55, fh * 0.62))
    eye_band = _ellipse_mask(shape_hw, (cx, fy + fh * 0.36), (fw * 0.52, fh * 0.17))
    left_eye = _ellipse_mask(shape_hw, (fx + fw * 0.33, fy + fh * 0.36), (fw * 0.16, fh * 0.10))
    right_eye = _ellipse_mask(shape_hw, (fx + fw * 0.67, fy + fh * 0.36), (fw * 0.16, fh * 0.10))
    eyes = np.clip(np.maximum(eye_band * 0.65, np.maximum(left_eye, right_eye)), 0.0, 1.0)
    lips = _ellipse_mask(shape_hw, (cx, fy + fh * 0.76), (fw * 0.24, fh * 0.12))
    hair = _ellipse_mask(shape_hw, (cx, fy + fh * 0.15), (fw * 0.95, fh * 0.65))
    hair = np.clip(hair * (1.0 - 0.35 * face), 0.0, 1.0)

    return {
        "face": face.astype(np.float32),
        "skin": skin.astype(np.float32),
        "eyes": eyes.astype(np.float32),
        "lips": lips.astype(np.float32),
        "hair": hair.astype(np.float32),
    }


def _ellipse_mask(shape_hw: tuple[int, int], center_xy: tuple[float, float], radius_xy: tuple[float, float]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = center_xy
    rx, ry = radius_xy
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(rx))), max(1, int(round(ry)))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    return mask


def _oval_zone(h: int, w: int, cx_frac: float, cy_frac: float, rx_frac: float, ry_frac: float) -> np.ndarray:
    return _ellipse_mask((h, w), (w * cx_frac, h * cy_frac), (w * rx_frac, h * ry_frac))


def _softmax_last_axis(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    denom = np.sum(exp_scores, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (exp_scores / denom).astype(np.float32)


def _edge_aware_region_refine(mask: np.ndarray, guide_img: np.ndarray, region: str) -> np.ndarray:
    if mask.ndim != 2 or mask.size == 0:
        return mask.astype(np.float32)

    src = np.clip(mask.astype(np.float32), 0.0, 1.0)

    def safe(refined: np.ndarray) -> np.ndarray:
        refined = np.clip(np.asarray(refined, dtype=np.float32), 0.0, 1.0)
        src_area = float(np.sum(src > 0.05))
        if src_area > 0 and float(np.sum(refined > 0.05)) < src_area * 0.10:
            return src
        return refined

    if region == "hair":
        refined = refine_mask_edges(mask, guide_img, radius=10, eps=5e-4)
        return np.clip(smooth_mask(safe(refined), sigma=1.2), 0.0, 1.0).astype(np.float32)
    if region in {"face", "skin"}:
        refined = refine_mask_edges(mask, guide_img, radius=8, eps=8e-4)
        return np.clip(smooth_mask(safe(refined), sigma=0.9), 0.0, 1.0).astype(np.float32)
    refined = refine_mask_edges(mask, guide_img, radius=4, eps=1e-3)
    return np.clip(smooth_mask(safe(refined), sigma=0.6), 0.0, 1.0).astype(np.float32)


def _facial_hair_mask(
    img_float: np.ndarray,
    *,
    face_box: tuple[int, int, int, int] | None,
    masks: dict,
    guides: dict | None = None,
) -> np.ndarray:
    h, w = img_float.shape[:2]
    if face_box is None:
        return np.zeros((h, w), dtype=np.float32)

    fx, fy, fw, fh = face_box
    face_mask = np.clip(np.asarray(masks.get("face", 0.0), dtype=np.float32), 0.0, 1.0)
    skin_mask = np.clip(np.asarray(masks.get("skin", 0.0), dtype=np.float32), 0.0, 1.0)
    hair_mask = np.clip(np.asarray(masks.get("hair", 0.0), dtype=np.float32), 0.0, 1.0)
    brow_mask = np.clip(np.asarray(masks.get("brows", 0.0), dtype=np.float32), 0.0, 1.0)

    lum = np.clip(
        img_float[:, :, 0] * 0.2126 + img_float[:, :, 1] * 0.7152 + img_float[:, :, 2] * 0.0722,
        0.0,
        1.0,
    ).astype(np.float32)
    dark = np.clip((0.54 - lum) / 0.38, 0.0, 1.0)
    texture = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), sigmaX=1.4)).astype(np.float32)
    texture = np.clip(texture * 5.0, 0.0, 1.0)
    coarse = np.abs(lum - cv2.GaussianBlur(lum, (0, 0), sigmaX=3.0)).astype(np.float32)
    coarse = np.clip(coarse * 3.0, 0.0, 1.0)
    hairlike = np.clip(dark * 0.75 + texture * 0.55 + coarse * 0.35, 0.0, 1.0)

    center_x = fx + fw * 0.5
    mouth_center = guides.get("mouth_center") if guides else None
    mouth_upper = guides.get("mouth_upper") if guides else None
    mouth_lower = guides.get("mouth_lower") if guides else None
    mouth_left = guides.get("mouth_left") if guides else None
    mouth_right = guides.get("mouth_right") if guides else None

    if mouth_center and mouth_upper and mouth_lower:
        lip_w = max(2.0, abs((mouth_right or mouth_center)[0] - (mouth_left or mouth_center)[0]))
        lip_h = max(2.0, abs(mouth_lower[1] - mouth_upper[1]))
        moustache = _ellipse_mask((h, w), (mouth_center[0], mouth_upper[1] - lip_h * 0.65), (lip_w * 0.60, lip_h * 0.95))
        beard = _ellipse_mask((h, w), (mouth_center[0], mouth_lower[1] + fh * 0.16), (lip_w * 0.85, fh * 0.19))
    else:
        moustache = _ellipse_mask((h, w), (center_x, fy + fh * 0.63), (fw * 0.18, fh * 0.06))
        beard = _ellipse_mask((h, w), (center_x, fy + fh * 0.84), (fw * 0.26, fh * 0.17))

    cheek_left = _ellipse_mask((h, w), (fx + fw * 0.23, fy + fh * 0.78), (fw * 0.10, fh * 0.14))
    cheek_right = _ellipse_mask((h, w), (fx + fw * 0.77, fy + fh * 0.78), (fw * 0.10, fh * 0.14))
    support = np.clip(moustache * 0.9 + beard + cheek_left * 0.35 + cheek_right * 0.35, 0.0, 1.0)
    support *= np.clip(face_mask * (1.0 - brow_mask) * (1.0 - hair_mask * 0.2), 0.0, 1.0)

    candidate = np.clip(support * hairlike * (0.55 + (1.0 - skin_mask) * 0.45), 0.0, 1.0)
    candidate = smooth_mask(candidate, sigma=max(1.0, fw * 0.010))
    return np.clip(candidate, 0.0, 1.0).astype(np.float32)
