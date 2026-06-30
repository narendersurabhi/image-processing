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
    "person": "#a3c768",
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
    "person": "◒ Person",
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
    # Creative warm/cool + green/magenta *grade* (keys kept for back-compat). Renamed from
    # "Temperature"/"Tint" so they don't read as the corrective White Balance section, which is
    # a different operation (Kelvin gains in linear light). See _build_slider_block tooltips.
    ("temperature", "Warmth", -100, 100, 0),
    ("tint", "Tint (G/M)", -100, 100, 0),
    ("highlights", "Highlights", -100, 100, 0),
    ("shadows", "Shadows", -100, 100, 0),
    ("whites", "Whites", -100, 100, 0),
    ("blacks", "Blacks", -100, 100, 0),
    ("midtones", "Midtones", -100, 100, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("sharpness", "Sharpness", 0, 100, 0),
    # Radius (x100, e.g. 140 = 1.4px) and Masking (0-100) deepen the single Sharpness "Amount"
    # slider with the same Amount/Radius/Masking model real raw converters expose -- defaults
    # match the values that were previously hardcoded in _apply_unsharp_mask, so an existing
    # preset with only "sharpness" set renders identically to before these were added.
    ("sharpen_radius", "Sharpen Radius", 50, 300, 140),
    ("sharpen_masking", "Sharpen Masking", 0, 100, 8),
    ("noise_red", "Luminance NR", 0, 100, 0),
    # An *additional* boost on top of Luminance NR's own (unchanged) chroma smoothing -- left
    # at 0, chroma noise reduction behaves exactly as before this was added (noise_red alone
    # drives both channels); raised above 0, color/chroma noise gets extra suppression beyond
    # what Luminance NR alone provides, for stubborn color blotches.
    ("color_noise_red", "Color NR", 0, 100, 0),
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

# Same controls as Subjects -- this is "Subjects, scoped to one person" (see Face Target).
PERSON_SLIDERS = [
    ("exposure", "Exposure", -150, 150, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("noise_red", "Noise Reduc.", 0, 100, 0),
    ("saturation", "Saturation", -100, 100, 0),
    ("warmth", "Warmth", -100, 100, 0),
]

BACKGROUND_SLIDERS = [
    ("blur", "Blur", 0, 100, 0),
    ("exposure", "Exposure", -150, 150, 0),
    ("clarity", "Clarity", -100, 100, 0),
    ("noise_red", "Noise Reduc.", 0, 100, 0),
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
    ("noise_red", "Noise Reduc.", 0, 100, 0),
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
    # Dict order drives both the layer-tab display order (see _tab_layers in main_window.py)
    # and reflects the actual hierarchy of a photo: Subjects (everyone) > Person (one
    # individual's full body) > Face > Skin/Eyes/Lips/Hair (within that face). Background is
    # scene-wide "everything else", not part of the person hierarchy, so it sits last.
    "subjects": SUBJECTS_SLIDERS,
    "person": PERSON_SLIDERS,
    "face": FACE_SLIDERS,
    "skin": SKIN_SLIDERS,
    "eyes": EYES_SLIDERS,
    "lips": LIPS_SLIDERS,
    "hair": HAIR_SLIDERS,
    "background": BACKGROUND_SLIDERS,
}

# Presentational grouping of each layer's sliders into labeled sections so related
# controls are easy to find (Light / Color / Detail / etc.), instead of one flat list.
# Format: layer -> [(group title, [slider keys in display order]), ...]. Any slider key
# not listed in a group is appended under a trailing "More" header by the UI, so this
# never has to be kept perfectly in sync with the *_SLIDERS lists above.
LAYER_SLIDER_GROUPS = {
    "global": [
        # Midtones sits next to Exposure (it's a midtone/contrast control), not stranded after
        # Blacks. Detail follows the app's denoise-first workflow (and the pipeline order): set
        # the noise floor before sharpening, so Luminance/Color NR precede Sharpness.
        ("Light", ["exposure", "midtones", "highlights", "shadows", "whites", "blacks"]),
        ("Color", ["temperature", "tint", "vibrance", "saturation"]),
        ("Detail", ["clarity", "noise_red", "color_noise_red", "sharpness", "sharpen_radius", "sharpen_masking"]),
        ("Effects", ["glow", "vignette"]),
    ],
    "subjects": [
        ("Light", ["exposure"]),
        ("Color", ["saturation", "warmth"]),
        ("Detail", ["clarity"]),
    ],
    "person": [
        ("Light", ["exposure"]),
        ("Color", ["saturation", "warmth"]),
        ("Detail", ["clarity", "noise_red"]),
    ],
    "face": [
        ("Light & Tone", ["exposure", "highlights", "shadows"]),
        ("Expression", ["smile", "eye_open", "brow_lift", "mouth_open", "jaw_relax"]),
        ("Skin & Detail", ["smooth", "clarity", "sharpness", "glow"]),
        ("Color", ["saturation"]),
        ("AI Refine", ["refine", "refine_fidelity"]),
    ],
    "skin": [
        ("Light", ["exposure"]),
        ("Retouch", ["smooth", "blemish", "glow"]),
        ("Detail", ["clarity", "noise_red"]),
        ("Color", ["saturation", "warmth"]),
    ],
    "eyes": [
        ("Light", ["brightness"]),
        ("Whiten", ["whites"]),
        ("Iris & Detail", ["iris_pop", "sharpen", "clarity"]),
        ("Color", ["saturation"]),
    ],
    "lips": [
        ("Light", ["brightness"]),
        ("Color", ["saturation", "warmth", "hue_shift"]),
        ("Finish", ["smooth", "gloss"]),
    ],
    "hair": [
        ("Light & Tone", ["brightness", "highlights"]),
        ("Color", ["saturation", "warmth"]),
        ("Detail & Shine", ["shine", "clarity"]),
    ],
    "background": [
        ("Light", ["exposure"]),
        ("Color", ["saturation", "warmth"]),
        ("Detail", ["clarity", "dehaze", "noise_red"]),
        ("Effects", ["blur"]),
    ],
}

# Render/compositing order for process_all_layers -- independent of the tab display order
# above (which comes from ALL_LAYERS' dict order). "person" is inserted right after
# "subjects" (both are coarse, scene/individual-level adjustments that should land before the
# finer per-feature layers) without disturbing the existing hair/skin/face/eyes/lips sequence.
MASK_ORDER = ("background", "subjects", "person", "hair", "skin", "face", "eyes", "lips")
BLEND_MODES = ("normal", "overlay", "soft_light")


def plain_layer_name(layer: str) -> str:
    """Return a plain uppercase name for headers from decorated layer labels."""
    return "".join(ch for ch in LAYER_NAMES[layer] if ch.isalnum() or ch.isspace()).strip().upper()
