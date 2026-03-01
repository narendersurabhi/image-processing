"""Application constants and layer/slider definitions."""

BG = "#0c0c0e"
PANEL = "#131316"
CARD = "#1a1a1f"
CARD2 = "#202026"
ACCENT = "#d4a853"
ACCENT_BLUE = "#5b9bd5"
TEXT = "#ddd8cf"
TEXT_DIM = "#5a5650"
SEP = "#252528"

LAYER_COLORS = {
    "global": "#d4a853",
    "subjects": "#8bcf7a",
    "background": "#5d7dd4",
    "face": "#c87c4a",
    "skin": "#e8b89a",
    "eyes": "#5b9bd5",
    "lips": "#d45b5b",
    "hair": "#7c5ba0",
}

LAYER_NAMES = {
    "global": "⊙ Global",
    "subjects": "◌ Subjects",
    "background": "▥ Background",
    "face": "◈ Face",
    "skin": "◉ Skin",
    "eyes": "◎ Eyes",
    "lips": "◑ Lips",
    "hair": "◐ Hair",
}

# (key, label, min, max, default)
GLOBAL_SLIDERS = [
    ("exposure", "Exposure", -200, 200, 0),
    ("temperature", "Temperature", -100, 100, 0),
    ("tint", "Tint", -100, 100, 0),
    ("highlights", "Highlights", -100, 100, 0),
    ("shadows", "Shadows", -100, 100, 0),
    ("whites", "Whites", -100, 100, 0),
    ("blacks", "Blacks", -100, 100, 0),
    ("midtones", "Midtones", -100, 100, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("sharpness", "Sharpness", 0, 100, 0),
    ("noise_red", "Noise Reduc.", 0, 100, 0),
    ("vibrance", "Vibrance", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("glow", "Portrait Glow", 0, 100, 0),
    ("vignette", "Vignette", -100, 100, 0),
]

SUBJECTS_SLIDERS = [
    ("exposure", "Exposure", -150, 150, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
]

BACKGROUND_SLIDERS = [
    ("blur", "Blur", 0, 100, 0),
    ("exposure", "Exposure", -150, 150, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
    ("dehaze", "Dehaze", 0, 100, 0),
]

FACE_SLIDERS = [
    ("exposure", "Exposure", -150, 150, 0),
    ("highlights", "Highlights", -100, 100, 0),
    ("shadows", "Shadows", -100, 100, 0),
    ("smile", "Smile", -100, 100, 0),
    ("eye_open", "Eye Open", -100, 100, 0),
    ("brow_lift", "Brow Lift", -100, 100, 0),
    ("mouth_open", "Mouth Open", -100, 100, 0),
    ("jaw_relax", "Jaw Relax", -100, 100, 0),
    ("refine", "AI Refine", 0, 100, 0),
    ("refine_fidelity", "AI Fidelity", 0, 100, 65),
    ("clarity", "Clarity", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("sharpness", "Sharpness", 0, 100, 0),
    ("smooth", "Smooth", 0, 100, 0),
    ("glow", "Glow", 0, 100, 0),
]

SKIN_SLIDERS = [
    ("exposure", "Exposure", -150, 150, 0),
    ("smooth", "Smooth", 0, 100, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
    ("glow", "Glow", 0, 100, 0),
    ("blemish", "Blemish Fix", 0, 100, 0),
]

EYES_SLIDERS = [
    ("brightness", "Brightness", -100, 100, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("whites", "Whiten", 0, 100, 0),
    ("sharpen", "Sharpen", 0, 100, 0),
    ("iris_pop", "Iris Pop", 0, 100, 0),
]

LIPS_SLIDERS = [
    ("brightness", "Brightness", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
    ("smooth", "Smooth", 0, 100, 0),
    ("gloss", "Gloss", 0, 100, 0),
    ("hue_shift", "Hue Shift", -100, 100, 0),
]

HAIR_SLIDERS = [
    ("brightness", "Brightness", -100, 100, 0),
    ("highlights", "Highlights", -100, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("shine", "Shine", 0, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
    ("clarity", "Clarity", -100, 100, 0),
]

ALL_LAYERS = {
    "global": GLOBAL_SLIDERS,
    "subjects": SUBJECTS_SLIDERS,
    "background": BACKGROUND_SLIDERS,
    "face": FACE_SLIDERS,
    "skin": SKIN_SLIDERS,
    "eyes": EYES_SLIDERS,
    "lips": LIPS_SLIDERS,
    "hair": HAIR_SLIDERS,
}

MASK_ORDER = ("background", "subjects", "hair", "skin", "face", "eyes", "lips")
BLEND_MODES = ("normal", "overlay", "soft_light")


def plain_layer_name(layer: str) -> str:
    """Return a plain uppercase name for headers from decorated layer labels."""
    return "".join(ch for ch in LAYER_NAMES[layer] if ch.isalnum() or ch.isspace()).strip().upper()
