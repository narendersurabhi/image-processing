"""Face/feature segmentation utilities with model and heuristic backends."""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .utils import refine_mask_edges, smooth_mask, to_uint8

MASK_KEYS = ("face", "skin", "eyes", "lips", "hair", "brows", "facial_hair", "subjects", "background")
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
        try:
            self._yunet.setInputSize((w, h))
            _retval, detections = self._yunet.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return _normalize_detector_faces(detections, (h, w))
        except Exception:
            return []

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

        try:
            import onnxruntime as ort
        except Exception as exc:
            self.reason_unavailable = f"onnxruntime unavailable: {exc}"
            return

        mp = None
        try:
            import mediapipe as mp
        except Exception:
            mp = None
        self._mp = mp

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
            self._init_landmark_backend(mp)
            self.available = True
        except Exception as exc:
            if "CoreMLExecutionProvider" in providers:
                try:
                    self._session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
                    self.execution_provider = "cpu"
                    self._input_name = self._session.get_inputs()[0].name
                    self._input_hw = self._infer_input_hw(self._session.get_inputs()[0].shape)
                    self._init_landmark_backend(mp)
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
        if landmarks is not None:
            for key in ("face", "eyes", "lips"):
                masks[key] = np.maximum(masks[key], landmarks[key])

        masks = _postprocess_model_masks(masks, img_float, face_box=face_box, shape_hw=(h, w))
        masks["facial_hair"] = _facial_hair_mask(
            img_float,
            face_box=face_box,
            masks=masks,
            guides=guides,
        )

        width = max(1, w)
        masks["face"] = smooth_mask(np.clip(masks["face"], 0, 1), sigma=max(2.0, width * 0.015))
        masks["skin"] = smooth_mask(np.clip(masks["skin"], 0, 1), sigma=max(2.0, width * 0.010))
        masks["brows"] = smooth_mask(np.clip(masks["brows"], 0, 1), sigma=max(1.0, width * 0.005))
        masks["facial_hair"] = smooth_mask(np.clip(masks["facial_hair"], 0, 1), sigma=max(1.2, width * 0.006))
        masks["eyes"] = smooth_mask(np.clip(masks["eyes"], 0, 1), sigma=max(1.0, width * 0.006))
        masks["lips"] = smooth_mask(np.clip(masks["lips"], 0, 1), sigma=max(1.5, width * 0.007))
        masks["hair"] = smooth_mask(np.clip(masks["hair"], 0, 1), sigma=max(2.5, width * 0.012))
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
        best = None
        best_score = None
        for landmarks in landmark_sets:
            box = self._landmark_bbox(landmarks, w, h)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            dist = float(np.linalg.norm(center - target))
            scale_penalty = abs((x2 - x1) - fw) * 0.25 + abs((y2 - y1) - fh) * 0.20
            score = dist + scale_penalty
            if best_score is None or score < best_score:
                best = landmarks
                best_score = score
        return best if best is not None else landmark_sets[0]

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
        self._subjects = SubjectSegmenter()
        self._facial_hair = FacialHairSegmenter()

        self.backend = "heuristic"
        self.backend_label = "heuristic"
        self.detector_backend = getattr(self._heuristic, "detector_backend", "haar")
        self.detector_backend_label = f"detect={self.detector_backend}"
        self.subject_backend = getattr(self._subjects, "backend", "heuristic")
        self.subject_backend_label = f"subjects={self.subject_backend}"
        self.facial_hair_backend = getattr(self._facial_hair, "backend", "fallback")
        self.facial_hair_backend_label = f"f_hair={self.facial_hair_backend}"
        self.reason_unavailable = ""

        if backend == "heuristic":
            return

        if self._model is not None and self._model.available:
            self.backend = "model"
            self.backend_label = self._model_backend_label()
            return

        if self._model is not None:
            self.reason_unavailable = self._model.reason_unavailable
            self.backend = "heuristic"
            self.backend_label = "heuristic (fallback)"

    def list_faces(self, img_float: np.ndarray) -> list[tuple[int, int, int, int]]:
        self.detector_backend = getattr(self._heuristic, "detector_backend", "haar")
        self.detector_backend_label = f"detect={self.detector_backend}"
        self.subject_backend = getattr(self._subjects, "backend", "heuristic")
        self.subject_backend_label = f"subjects={self.subject_backend}"
        self.facial_hair_backend = getattr(self._facial_hair, "backend", "fallback")
        self.facial_hair_backend_label = f"f_hair={self.facial_hair_backend}"
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
                masks["facial_hair"] = self._facial_hair_mask(img_float, target_face, masks, guides)
                if focus_mask is not None:
                    masks = _apply_focus_mask(masks, focus_mask)
                masks["subjects"], masks["background"] = self._scene_subject_masks(img_float, faces, masks)
                return masks, guides
            except Exception as exc:
                self.reason_unavailable = f"model runtime error: {exc}"
                self.backend = "heuristic"
                self.backend_label = "heuristic (runtime fallback)"

        masks = self._heuristic.segment(img_float, face_index=face_index)
        target_face = faces[_safe_face_index(face_index, len(faces))] if faces else None
        masks["facial_hair"] = self._facial_hair_mask(img_float, target_face, masks, None)
        masks["subjects"], masks["background"] = self._scene_subject_masks(img_float, faces, masks)
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
        if self._subjects is None or not self._subjects.available:
            self.subject_backend = "heuristic"
            self.subject_backend_label = "subjects=heuristic"
            return heuristic_subjects, heuristic_background

        try:
            model_subjects = self._subjects.segment(img_float)
        except Exception:
            model_subjects = None
        if model_subjects is None:
            self.subject_backend = "heuristic"
            self.subject_backend_label = "subjects=heuristic"
            return heuristic_subjects, heuristic_background

        merged_subjects = np.maximum(
            np.clip(model_subjects.astype(np.float32), 0.0, 1.0),
            np.clip(heuristic_subjects * 0.35, 0.0, 1.0),
        )
        merged_subjects = np.clip(smooth_mask(merged_subjects, sigma=max(2.0, img_float.shape[1] * 0.008)), 0.0, 1.0).astype(np.float32)
        merged_background = np.clip(1.0 - merged_subjects, 0.0, 1.0).astype(np.float32)
        self.subject_backend = self._subjects.backend
        self.subject_backend_label = f"subjects={self.subject_backend}"
        return merged_subjects, merged_background

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


def _empty_masks(h, w):
    return {k: np.zeros((h, w), dtype=np.float32) for k in MASK_KEYS}


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

    face = _cleanup_region_mask(masks["face"], zone=zones["face"], min_area=min_area_face, close_size=7, open_size=3, keep_largest=1)
    brows = _cleanup_region_mask(masks.get("brows", np.zeros((h, w), dtype=np.float32)), zone=zones["eyes"], min_area=min_area_brows, close_size=3, open_size=1, keep_largest=2)
    eyes = _cleanup_region_mask(masks["eyes"], zone=zones["eyes"], min_area=min_area_eye, close_size=3, open_size=1, keep_largest=2)
    lips = _cleanup_region_mask(masks["lips"], zone=zones["lips"], min_area=min_area_lips, close_size=5, open_size=1, keep_largest=1)
    hair = _cleanup_region_mask(masks["hair"], zone=zones["hair"], min_area=min_area_hair, close_size=9, open_size=3, keep_largest=2)
    skin = _cleanup_region_mask(masks["skin"], zone=zones["skin"], min_area=min_area_skin, close_size=7, open_size=3, keep_largest=1)

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

    if region == "hair":
        refined = refine_mask_edges(mask, guide_img, radius=10, eps=5e-4)
        return np.clip(smooth_mask(refined, sigma=1.2), 0.0, 1.0).astype(np.float32)
    if region in {"face", "skin"}:
        refined = refine_mask_edges(mask, guide_img, radius=8, eps=8e-4)
        return np.clip(smooth_mask(refined, sigma=0.9), 0.0, 1.0).astype(np.float32)
    refined = refine_mask_edges(mask, guide_img, radius=4, eps=1e-3)
    return np.clip(smooth_mask(refined, sigma=0.6), 0.0, 1.0).astype(np.float32)


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
