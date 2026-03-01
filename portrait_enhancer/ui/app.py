"""Tkinter UI for Portrait Enhancer v2."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import tkinter as tk
import zlib
from datetime import datetime, timezone
from queue import Empty, Queue
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, ImageTk

from portrait_enhancer.config import (
    ACCENT,
    ACCENT_BLUE,
    ALL_LAYERS,
    BG,
    BLEND_MODES,
    CARD,
    CARD2,
    LAYER_COLORS,
    LAYER_NAMES,
    MASK_ORDER,
    PANEL,
    SEP,
    TEXT,
    TEXT_DIM,
    plain_layer_name,
)
from portrait_enhancer.core.lut import apply_cube_lut, load_cube_lut
from portrait_enhancer.core.processing import process_all_layers
from portrait_enhancer.core.processing import _scale_expression_guides
from portrait_enhancer.core.segmentation import FaceSegmenter
from portrait_enhancer.core.utils import (
    cuda_available,
    resolve_acceleration_mode,
    resize_image,
    smooth_mask,
    to_uint8,
)

try:
    import rawpy

    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

try:
    _SRGB_ICC_BYTES = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
except Exception:
    _SRGB_ICC_BYTES = None


class PortraitEnhancerV2(tk.Tk):
    SUPPORTED_IMAGE_EXTS = (".cr2", ".nef", ".arw", ".dng", ".raw", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    BATCH_OUTPUT_FORMATS = {
        "jpeg": ".jpg",
        "png": ".png",
        "tiff": ".tiff",
    }

    def __init__(self):
        super().__init__()
        self.title("Portrait Enhancer v2 - Selective Layers")
        self.geometry("1520x900")
        self.minsize(1100, 650)
        self.configure(bg=BG)

        # Image state
        self.full_array = None
        self.preview_array = None
        self.full_masks = None
        self.preview_masks = None
        self.full_guides = None
        self.preview_guides = None
        self._auto_full_masks = None
        self._auto_preview_masks = None
        self.preview_scale = 1.0
        self.preview_image = None
        self.tk_image = None
        self.file_path = ""
        self._source_metadata = {}
        self._source_native_array = None
        self._source_is_raw = False
        self._loaded_raw_lut = None
        self._loaded_raw_lut_path = ""

        # Render queue state
        self._image_revision = 0
        self._latest_render_request = 0
        self._render_queue = Queue(maxsize=1)
        self._debounce_id = None

        # Display mapping for canvas -> image painting
        self._display_origin = (0, 0)
        self._display_size = (0, 0)

        # Layer stack state
        self._layer_order = list(MASK_ORDER)
        self._layer_options = {
            layer: {"enabled": True, "opacity": 100.0, "blend_mode": "normal"}
            for layer in MASK_ORDER
        }
        self._color_settings = self._default_color_settings()
        self._detected_faces = []
        self._active_face_index = 0
        self._face_profiles = {}
        self._runtime_settings = self._default_runtime_settings()

        # UI state
        self._show_mask = False
        self._compare_mode = "off"
        self._split_position = 0.5
        self._active_layer = "global"
        self._sync_layer_controls = False
        self._sync_profile_load = False
        self.segmenter = FaceSegmenter()
        self._sliders = {}
        self._slider_label_vars = {}
        self._layer_frames = {}
        self._slider_scroll_canvas = None

        # Layer control vars
        self._opt_enabled_var = tk.BooleanVar(value=True)
        self._opt_opacity_var = tk.DoubleVar(value=100.0)
        self._opt_blend_var = tk.StringVar(value="normal")
        self._opt_opacity_text = tk.StringVar(value="100%")
        self._face_choice_var = tk.StringVar(value="Auto")
        self._face_choices = []
        self._input_profile_var = tk.StringVar(value=self._color_settings["input_profile"])
        self._raw_wb_var = tk.StringVar(value=self._color_settings["raw_white_balance"])
        self._raw_colorspace_var = tk.StringVar(value=self._color_settings["raw_colorspace"])
        self._raw_lut_enabled_var = tk.BooleanVar(value=self._color_settings["raw_lut_enabled"])
        self._working_space_var = tk.StringVar(value=self._color_settings["working_space"])
        self._output_transform_var = tk.StringVar(value=self._color_settings["output_transform"])
        self._icc_policy_var = tk.StringVar(value=self._color_settings["icc_policy"])
        self._source_profile_var = tk.StringVar(value="Source ICC: none")
        self._raw_lut_label_var = tk.StringVar(value="RAW LUT: none")
        self._acceleration_var = tk.StringVar(value=self._runtime_settings["acceleration_mode"])
        self._acceleration_status_var = tk.StringVar(value="")
        self._compare_mode_var = tk.StringVar(value="off")
        self._preset_search_var = tk.StringVar(value="")
        self._preset_category_var = tk.StringVar(value="all")
        self._preset_meta_var = tk.StringVar(value="No preset selected")
        self._sync_color_controls = False
        self._preset_library_entries = []

        # Mask edit vars
        self._mask_edit_enabled = False
        self._mask_paint_mode_var = tk.StringVar(value="paint")
        self._mask_brush_size_var = tk.IntVar(value=24)
        self._mask_history = []
        self._mask_history_index = -1
        self._mask_stroke_active = False
        self._max_mask_history = 40

        self._build_ui()

        self._opt_enabled_var.trace_add("write", self._on_layer_option_controls_changed)
        self._opt_opacity_var.trace_add("write", self._on_layer_option_controls_changed)
        self._opt_blend_var.trace_add("write", self._on_layer_option_controls_changed)
        self._input_profile_var.trace_add("write", self._on_color_settings_changed)
        self._raw_wb_var.trace_add("write", self._on_color_settings_changed)
        self._raw_colorspace_var.trace_add("write", self._on_color_settings_changed)
        self._raw_lut_enabled_var.trace_add("write", self._on_color_settings_changed)
        self._working_space_var.trace_add("write", self._on_color_settings_changed)
        self._output_transform_var.trace_add("write", self._on_color_settings_changed)
        self._icc_policy_var.trace_add("write", self._on_color_settings_changed)
        self._acceleration_var.trace_add("write", self._on_runtime_settings_changed)
        self._compare_mode_var.trace_add("write", self._on_compare_mode_changed)
        self._preset_search_var.trace_add("write", self._on_preset_filter_changed)
        self._preset_category_var.trace_add("write", self._on_preset_filter_changed)

        self._render_thread = threading.Thread(target=self._render_worker, daemon=True)
        self._render_thread.start()
        self._update_acceleration_status()

    def _build_ui(self):
        topbar = tk.Frame(self, bg=PANEL, height=54)
        topbar.pack(fill=tk.X)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="PORTRAIT ENHANCER", font=("Georgia", 12, "bold"), fg=ACCENT, bg=PANEL, padx=20).pack(
            side=tk.LEFT, pady=12
        )
        tk.Label(topbar, text="v2  ·  Selective Layers", font=("Courier", 9), fg=TEXT_DIM, bg=PANEL).pack(
            side=tk.LEFT, pady=14
        )

        btn_row = tk.Frame(topbar, bg=PANEL)
        btn_row.pack(side=tk.RIGHT, padx=14)
        self._btn(btn_row, "Open Image", self._open_file).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Open Project", self._open_project).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Save Project", self._save_project).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Open Preset", self._open_preset).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Save Preset", self._save_preset).pack(side=tk.LEFT, padx=3)
        compare_row = tk.Frame(btn_row, bg=PANEL)
        compare_row.pack(side=tk.LEFT, padx=3)
        tk.Label(compare_row, text="Compare", font=("Helvetica", 8), fg=TEXT_DIM, bg=PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self._compare_mode_combo = ttk.Combobox(
            compare_row,
            values=("off", "split", "before", "side_by_side"),
            textvariable=self._compare_mode_var,
            state="readonly",
            width=14,
        )
        self._compare_mode_combo.pack(side=tk.LEFT)
        self._btn(btn_row, "Batch Export", self._batch_export).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Retry Failed", self._retry_failed_batch).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Re-Analyze", self._reanalyze).pack(side=tk.LEFT, padx=3)
        self._mask_btn = self._btn(btn_row, "Mask View", self._toggle_mask_view)
        self._mask_btn.pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Export", self._export, style="accent").pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Reset Layer", self._reset_layer).pack(side=tk.LEFT, padx=3)
        self._btn(btn_row, "Reset All", self._reset_all).pack(side=tk.LEFT, padx=3)

        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=PANEL, width=312)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tab_frame = tk.Frame(left, bg=BG)
        tab_frame.pack(fill=tk.X, pady=(8, 0))
        self._layer_btns = {}
        for layer in ALL_LAYERS:
            button = tk.Button(
                tab_frame,
                text=LAYER_NAMES[layer],
                font=("Helvetica", 8, "bold"),
                bg=CARD,
                fg=TEXT_DIM,
                relief=tk.FLAT,
                padx=8,
                pady=6,
                cursor="hand2",
                command=lambda l=layer: self._switch_layer(l),
            )
            button.pack(fill=tk.X, padx=6, pady=2)
            self._layer_btns[layer] = button

        face_card = tk.Frame(left, bg=CARD2)
        face_card.pack(fill=tk.X, padx=8, pady=(6, 6))
        tk.Label(face_card, text="Face Target", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )
        self._face_picker = ttk.Combobox(
            face_card,
            textvariable=self._face_choice_var,
            values=("Auto",),
            state="disabled",
            width=28,
        )
        self._face_picker.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._face_picker.bind("<<ComboboxSelected>>", self._on_face_picker_changed)

        color_card = tk.Frame(left, bg=CARD2)
        color_card.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(color_card, text="Color Management", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )

        input_row = tk.Frame(color_card, bg=CARD2)
        input_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(input_row, text="Input ICC", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._input_profile_combo = ttk.Combobox(
            input_row,
            values=("auto", "ignore"),
            textvariable=self._input_profile_var,
            state="readonly",
            width=11,
        )
        self._input_profile_combo.pack(side=tk.RIGHT)

        raw_wb_row = tk.Frame(color_card, bg=CARD2)
        raw_wb_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(raw_wb_row, text="RAW WB", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._raw_wb_combo = ttk.Combobox(
            raw_wb_row,
            values=("camera", "auto"),
            textvariable=self._raw_wb_var,
            state="readonly",
            width=11,
        )
        self._raw_wb_combo.pack(side=tk.RIGHT)

        raw_cs_row = tk.Frame(color_card, bg=CARD2)
        raw_cs_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(raw_cs_row, text="RAW Color", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._raw_colorspace_combo = ttk.Combobox(
            raw_cs_row,
            values=("srgb", "adobe", "prophoto", "xyz", "raw"),
            textvariable=self._raw_colorspace_var,
            state="readonly",
            width=11,
        )
        self._raw_colorspace_combo.pack(side=tk.RIGHT)

        raw_lut_row = tk.Frame(color_card, bg=CARD2)
        raw_lut_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._raw_lut_check = tk.Checkbutton(
            raw_lut_row,
            text="RAW LUT",
            variable=self._raw_lut_enabled_var,
            bg=CARD2,
            fg=TEXT,
            activebackground=CARD2,
            activeforeground=TEXT,
            selectcolor=CARD,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self._raw_lut_check.pack(side=tk.LEFT)
        self._raw_lut_pick_btn = self._btn(raw_lut_row, "Load LUT", self._select_raw_lut)
        self._raw_lut_pick_btn.pack(side=tk.RIGHT)

        raw_lut_info_row = tk.Frame(color_card, bg=CARD2)
        raw_lut_info_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(raw_lut_info_row, textvariable=self._raw_lut_label_var, font=("Courier", 7), fg=TEXT_DIM, bg=CARD2).pack(
            side=tk.LEFT
        )
        self._raw_lut_clear_btn = self._btn(raw_lut_info_row, "Clear", self._clear_raw_lut)
        self._raw_lut_clear_btn.pack(side=tk.RIGHT)

        work_row = tk.Frame(color_card, bg=CARD2)
        work_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(work_row, text="Working", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._working_space_combo = ttk.Combobox(
            work_row,
            values=("srgb", "linear"),
            textvariable=self._working_space_var,
            state="readonly",
            width=11,
        )
        self._working_space_combo.pack(side=tk.RIGHT)

        out_row = tk.Frame(color_card, bg=CARD2)
        out_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(out_row, text="Output Tone", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._output_transform_combo = ttk.Combobox(
            out_row,
            values=("srgb", "gamma22", "gamma18", "linear"),
            textvariable=self._output_transform_var,
            state="readonly",
            width=11,
        )
        self._output_transform_combo.pack(side=tk.RIGHT)

        icc_row = tk.Frame(color_card, bg=CARD2)
        icc_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(icc_row, text="ICC Embed", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._icc_policy_combo = ttk.Combobox(
            icc_row,
            values=("srgb", "preserve_source", "none"),
            textvariable=self._icc_policy_var,
            state="readonly",
            width=16,
        )
        self._icc_policy_combo.pack(side=tk.RIGHT)
        tk.Label(color_card, textvariable=self._source_profile_var, font=("Courier", 7), fg=TEXT_DIM, bg=CARD2).pack(
            anchor="w", padx=8, pady=(0, 7)
        )

        perf_card = tk.Frame(left, bg=CARD2)
        perf_card.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(perf_card, text="Performance", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )
        accel_row = tk.Frame(perf_card, bg=CARD2)
        accel_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(accel_row, text="Acceleration", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._acceleration_combo = ttk.Combobox(
            accel_row,
            values=("auto", "cpu", "cuda"),
            textvariable=self._acceleration_var,
            state="readonly",
            width=11,
        )
        self._acceleration_combo.pack(side=tk.RIGHT)
        tk.Label(perf_card, textvariable=self._acceleration_status_var, font=("Courier", 7), fg=TEXT_DIM, bg=CARD2).pack(
            anchor="w", padx=8, pady=(0, 7)
        )

        preset_card = tk.Frame(left, bg=CARD2)
        preset_card.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Label(preset_card, text="Preset Browser", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )
        preset_filter_row = tk.Frame(preset_card, bg=CARD2)
        preset_filter_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Entry(
            preset_filter_row,
            textvariable=self._preset_search_var,
            bg=CARD,
            fg=TEXT,
            insertbackground=TEXT,
            relief=tk.FLAT,
            highlightthickness=0,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._preset_category_combo = ttk.Combobox(
            preset_filter_row,
            values=("all",),
            textvariable=self._preset_category_var,
            state="readonly",
            width=12,
        )
        self._preset_category_combo.pack(side=tk.RIGHT)
        self._preset_listbox = tk.Listbox(
            preset_card,
            height=6,
            bg=CARD,
            fg=TEXT,
            selectbackground=ACCENT_BLUE,
            selectforeground=BG,
            relief=tk.FLAT,
            highlightthickness=0,
            activestyle="none",
        )
        self._preset_listbox.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._preset_listbox.bind("<Double-Button-1>", lambda _e: self._apply_selected_browser_preset())
        self._preset_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_selected_preset_meta())
        tk.Label(
            preset_card,
            textvariable=self._preset_meta_var,
            font=("Courier", 7),
            fg=TEXT_DIM,
            bg=CARD2,
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, padx=8, pady=(0, 6))
        preset_row = tk.Frame(preset_card, bg=CARD2)
        preset_row.pack(fill=tk.X, padx=8, pady=(0, 7))
        self._btn(preset_row, "Apply", self._apply_selected_browser_preset).pack(side=tk.LEFT, padx=(0, 4))
        self._btn(preset_row, "Save Here", self._save_preset_to_library).pack(side=tk.LEFT, padx=(0, 4))
        self._btn(preset_row, "Refresh", self._refresh_preset_browser).pack(side=tk.LEFT)

        control_card = tk.Frame(left, bg=CARD2)
        control_card.pack(fill=tk.X, padx=8, pady=(6, 6))

        tk.Label(control_card, text="Layer Mix", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )

        self._layer_enabled_check = tk.Checkbutton(
            control_card,
            text="Enabled",
            variable=self._opt_enabled_var,
            bg=CARD2,
            fg=TEXT,
            activebackground=CARD2,
            activeforeground=TEXT,
            selectcolor=CARD,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self._layer_enabled_check.pack(anchor="w", padx=8, pady=(0, 4))

        op_row = tk.Frame(control_card, bg=CARD2)
        op_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(op_row, text="Opacity", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        tk.Label(op_row, textvariable=self._opt_opacity_text, font=("Courier", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.RIGHT)

        self._opacity_scale = tk.Scale(
            control_card,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self._opt_opacity_var,
            showvalue=False,
            bg=CARD2,
            troughcolor=SEP,
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            sliderlength=12,
            width=4,
            length=250,
        )
        self._opacity_scale.pack(fill=tk.X, padx=8)

        blend_row = tk.Frame(control_card, bg=CARD2)
        blend_row.pack(fill=tk.X, padx=8, pady=(4, 6))
        tk.Label(blend_row, text="Blend", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._blend_combo = ttk.Combobox(
            blend_row,
            values=BLEND_MODES,
            textvariable=self._opt_blend_var,
            state="readonly",
            width=11,
        )
        self._blend_combo.pack(side=tk.RIGHT)

        order_row = tk.Frame(control_card, bg=CARD2)
        order_row.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._move_up_btn = self._btn(order_row, "Up", lambda: self._move_active_layer(-1))
        self._move_up_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._move_down_btn = self._btn(order_row, "Down", lambda: self._move_active_layer(1))
        self._move_down_btn.pack(side=tk.LEFT)
        self._order_var = tk.StringVar(value="")
        tk.Label(control_card, textvariable=self._order_var, font=("Courier", 7), fg=TEXT_DIM, bg=CARD2, anchor="w").pack(
            fill=tk.X, padx=8, pady=(0, 7)
        )

        mask_card = tk.Frame(left, bg=CARD2)
        mask_card.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(mask_card, text="Mask Tools", font=("Helvetica", 9, "bold"), fg=TEXT, bg=CARD2).pack(
            anchor="w", padx=8, pady=(7, 2)
        )

        self._mask_edit_btn = self._btn(mask_card, "Edit Mask", self._toggle_mask_edit)
        self._mask_edit_btn.pack(anchor="w", padx=8, pady=(0, 4))

        paint_row = tk.Frame(mask_card, bg=CARD2)
        paint_row.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._paint_radio = tk.Radiobutton(
            paint_row,
            text="Paint",
            value="paint",
            variable=self._mask_paint_mode_var,
            bg=CARD2,
            fg=TEXT,
            activebackground=CARD2,
            activeforeground=TEXT,
            selectcolor=CARD,
        )
        self._paint_radio.pack(side=tk.LEFT)
        self._erase_radio = tk.Radiobutton(
            paint_row,
            text="Erase",
            value="erase",
            variable=self._mask_paint_mode_var,
            bg=CARD2,
            fg=TEXT,
            activebackground=CARD2,
            activeforeground=TEXT,
            selectcolor=CARD,
        )
        self._erase_radio.pack(side=tk.LEFT, padx=(6, 0))

        brush_row = tk.Frame(mask_card, bg=CARD2)
        brush_row.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(brush_row, text="Brush", font=("Helvetica", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.LEFT)
        self._brush_size_text = tk.StringVar(value="24")
        tk.Label(brush_row, textvariable=self._brush_size_text, font=("Courier", 8), fg=TEXT_DIM, bg=CARD2).pack(side=tk.RIGHT)

        self._brush_scale = tk.Scale(
            mask_card,
            from_=2,
            to=80,
            orient=tk.HORIZONTAL,
            variable=self._mask_brush_size_var,
            showvalue=False,
            bg=CARD2,
            troughcolor=SEP,
            activebackground=ACCENT_BLUE,
            highlightthickness=0,
            bd=0,
            sliderrelief=tk.FLAT,
            sliderlength=12,
            width=4,
            length=250,
            command=lambda _v: self._brush_size_text.set(str(self._mask_brush_size_var.get())),
        )
        self._brush_scale.pack(fill=tk.X, padx=8)

        refine_row = tk.Frame(mask_card, bg=CARD2)
        refine_row.pack(fill=tk.X, padx=8, pady=(4, 7))
        self._feather_btn = self._btn(refine_row, "Feather", self._feather_active_mask)
        self._feather_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._reset_mask_btn = self._btn(refine_row, "Reset Mask", self._reset_active_mask)
        self._reset_mask_btn.pack(side=tk.LEFT)

        history_row = tk.Frame(mask_card, bg=CARD2)
        history_row.pack(fill=tk.X, padx=8, pady=(0, 7))
        self._undo_mask_btn = self._btn(history_row, "Undo", self._undo_mask_edit)
        self._undo_mask_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._redo_mask_btn = self._btn(history_row, "Redo", self._redo_mask_edit)
        self._redo_mask_btn.pack(side=tk.LEFT)

        tk.Frame(left, bg=SEP, height=1).pack(fill=tk.X, pady=6, padx=8)

        scroll_canvas = tk.Canvas(left, bg=PANEL, highlightthickness=0)
        scrollbar = tk.Scrollbar(left, orient=tk.VERTICAL, command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self._slider_scroll_canvas = scroll_canvas
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.pack(fill=tk.BOTH, expand=True)

        self._slider_container = tk.Frame(scroll_canvas, bg=PANEL)
        win_id = scroll_canvas.create_window((0, 0), window=self._slider_container, anchor="nw")

        def reconfigure(_event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
            scroll_canvas.itemconfig(win_id, width=scroll_canvas.winfo_width())

        self._slider_container.bind("<Configure>", reconfigure)
        scroll_canvas.bind("<Configure>", lambda event: scroll_canvas.itemconfig(win_id, width=event.width))
        scroll_canvas.bind_all("<MouseWheel>", lambda event: scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        for layer, sliders in ALL_LAYERS.items():
            self._sliders[layer] = {}
            self._slider_label_vars[layer] = {}
            frame = tk.Frame(self._slider_container, bg=PANEL)
            self._layer_frames[layer] = frame
            self._build_slider_panel(frame, layer, sliders)

        right = tk.Frame(body, bg=BG)
        right.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right, bg="#080809", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.hint_id = self.canvas.create_text(
            0,
            0,
            text="Open an image to begin\n\nSupports CR2 · NEF · ARW · DNG · JPEG · PNG",
            font=("Georgia", 14, "italic"),
            fill=TEXT_DIM,
            anchor="center",
            justify=tk.CENTER,
        )
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, bg=BG, fg=TEXT_DIM, font=("Courier", 9), anchor="w", padx=14).pack(
            fill=tk.X, side=tk.BOTTOM, pady=2
        )

        self._refresh_preset_browser()
        self._update_layer_order_label()
        self._switch_layer("global")

    def _build_slider_panel(self, frame, layer, sliders):
        color = LAYER_COLORS.get(layer, ACCENT)

        header = tk.Frame(frame, bg=PANEL)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))
        tk.Label(header, text="●", fg=color, bg=PANEL, font=("Helvetica", 11)).pack(side=tk.LEFT)
        tk.Label(header, text=plain_layer_name(layer), font=("Courier", 9, "bold"), fg=color, bg=PANEL, padx=6).pack(
            side=tk.LEFT
        )

        for key, label, mn, mx, default in sliders:
            var = tk.DoubleVar(value=default)
            sv = tk.StringVar(value="0")
            self._sliders[layer][key] = var
            self._slider_label_vars[layer][key] = sv

            row = tk.Frame(frame, bg=PANEL)
            row.pack(fill=tk.X, padx=12, pady=3)

            top = tk.Frame(row, bg=PANEL)
            top.pack(fill=tk.X)
            tk.Label(top, text=label, font=("Helvetica", 9), fg=TEXT, bg=PANEL, anchor="w").pack(side=tk.LEFT)
            tk.Label(top, textvariable=sv, font=("Courier", 8), fg=TEXT_DIM, bg=PANEL, width=5, anchor="e").pack(side=tk.RIGHT)

            slider = tk.Scale(
                row,
                from_=mn,
                to=mx,
                orient=tk.HORIZONTAL,
                variable=var,
                showvalue=False,
                bg=CARD2,
                troughcolor=SEP,
                activebackground=color,
                highlightthickness=0,
                bd=0,
                sliderrelief=tk.FLAT,
                sliderlength=12,
                width=4,
                length=250,
            )
            slider.pack(fill=tk.X)

            def callback(*_args, v=var, sv_=sv):
                value = v.get()
                sv_.set(f"{int(value):+d}" if value != 0 else "0")
                if not self._sync_profile_load:
                    self._schedule_update()

            var.trace_add("write", callback)

    def _btn(self, parent, text, cmd, style="normal"):
        bg = ACCENT if style == "accent" else CARD
        fg = BG if style == "accent" else TEXT
        return tk.Button(
            parent,
            text=text,
            command=cmd,
            bg=bg,
            fg=fg,
            font=("Helvetica", 8, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2",
            activebackground=ACCENT_BLUE,
            activeforeground=BG,
        )

    def _switch_layer(self, layer):
        self._active_layer = layer
        for frame in self._layer_frames.values():
            frame.pack_forget()
        self._layer_frames[layer].pack(fill=tk.X)
        self.after_idle(lambda l=layer: self._scroll_layer_into_view(l))

        for key, button in self._layer_btns.items():
            color = LAYER_COLORS.get(key, ACCENT)
            if key == layer:
                button.config(bg=CARD, fg=color, font=("Helvetica", 8, "bold"), relief=tk.FLAT)
            else:
                button.config(bg=PANEL, fg=TEXT_DIM, font=("Helvetica", 8), relief=tk.FLAT)

        self._load_active_layer_options()
        self._refresh_mask_tool_state()

        if self._show_mask and self.preview_masks:
            self._render_canvas()

    def _scroll_layer_into_view(self, layer):
        frame = self._layer_frames.get(layer)
        canvas = self._slider_scroll_canvas
        if frame is None or canvas is None:
            return

        self.update_idletasks()
        bbox = canvas.bbox("all")
        if not bbox:
            return

        content_height = max(bbox[3] - bbox[1], 1)
        viewport_height = max(canvas.winfo_height(), 1)
        if content_height <= viewport_height:
            canvas.yview_moveto(0.0)
            return

        top = max(frame.winfo_y() - 8, 0)
        max_top = max(content_height - viewport_height, 1)
        canvas.yview_moveto(min(top / max_top, 1.0))

    def _load_active_layer_options(self):
        self._sync_layer_controls = True
        try:
            if self._active_layer in self._layer_options:
                cfg = self._layer_options[self._active_layer]
                self._opt_enabled_var.set(bool(cfg.get("enabled", True)))
                self._opt_opacity_var.set(float(cfg.get("opacity", 100.0)))
                self._opt_blend_var.set(str(cfg.get("blend_mode", "normal")))
                self._opt_opacity_text.set(f"{int(self._opt_opacity_var.get())}%")
            else:
                self._opt_enabled_var.set(True)
                self._opt_opacity_var.set(100.0)
                self._opt_blend_var.set("normal")
                self._opt_opacity_text.set("100%")
        finally:
            self._sync_layer_controls = False

        selective = self._active_layer in self._layer_options
        state = "normal" if selective else "disabled"
        readonly = "readonly" if selective else "disabled"
        self._layer_enabled_check.configure(state=state)
        self._opacity_scale.configure(state=state)
        self._blend_combo.configure(state=readonly)
        self._move_up_btn.configure(state=state)
        self._move_down_btn.configure(state=state)

    def _on_layer_option_controls_changed(self, *_args):
        if self._sync_layer_controls or self._sync_profile_load:
            return
        if self._active_layer not in self._layer_options:
            return

        cfg = self._layer_options[self._active_layer]
        cfg["enabled"] = bool(self._opt_enabled_var.get())
        cfg["opacity"] = float(np.clip(self._opt_opacity_var.get(), 0.0, 100.0))
        cfg["blend_mode"] = str(self._opt_blend_var.get())
        self._opt_opacity_text.set(f"{int(cfg['opacity'])}%")

        if self.preview_array is not None:
            self._enqueue_render()

    def _move_active_layer(self, delta):
        layer = self._active_layer
        if layer not in self._layer_order:
            return
        idx = self._layer_order.index(layer)
        new_idx = int(np.clip(idx + delta, 0, len(self._layer_order) - 1))
        if new_idx == idx:
            return

        self._layer_order[idx], self._layer_order[new_idx] = self._layer_order[new_idx], self._layer_order[idx]
        self._update_layer_order_label()
        if self.preview_array is not None:
            self._enqueue_render()

    def _update_layer_order_label(self):
        text = " > ".join(name.title() for name in self._layer_order)
        self._order_var.set(f"Order: {text}")

    def _default_layer_options(self):
        return {layer: {"enabled": True, "opacity": 100.0, "blend_mode": "normal"} for layer in MASK_ORDER}

    def _default_runtime_settings(self):
        return {"acceleration_mode": "auto"}

    def _normalize_runtime_settings(self, runtime_settings):
        normalized = self._default_runtime_settings()
        normalized.update(runtime_settings or {})
        if normalized["acceleration_mode"] not in ("auto", "cpu", "cuda"):
            normalized["acceleration_mode"] = "auto"
        return normalized

    def _resolved_acceleration_mode(self):
        return resolve_acceleration_mode(self._runtime_settings.get("acceleration_mode", "auto"))

    def _update_acceleration_status(self):
        requested = self._runtime_settings.get("acceleration_mode", "auto")
        resolved = self._resolved_acceleration_mode()
        available = "cuda_available" if cuda_available() else "cuda_unavailable"
        self._acceleration_status_var.set(f"Accel: req={requested} -> {resolved} ({available})")

    def _on_runtime_settings_changed(self, *_args):
        self._runtime_settings = self._normalize_runtime_settings({"acceleration_mode": self._acceleration_var.get()})
        self._update_acceleration_status()
        if self.preview_array is not None:
            self._enqueue_render()

    def _default_color_settings(self):
        return {
            "input_profile": "auto",
            "raw_white_balance": "camera",
            "raw_colorspace": "srgb",
            "raw_lut_enabled": False,
            "raw_lut_path": "",
            "working_space": "srgb",
            "output_transform": "srgb",
            "icc_policy": "srgb",
        }

    def _normalize_color_settings(self, color_settings):
        normalized = self._default_color_settings()
        normalized.update(color_settings or {})
        if normalized["input_profile"] not in ("auto", "ignore"):
            normalized["input_profile"] = "auto"
        if normalized["raw_white_balance"] not in ("camera", "auto"):
            normalized["raw_white_balance"] = "camera"
        if normalized["raw_colorspace"] not in ("srgb", "adobe", "prophoto", "xyz", "raw"):
            normalized["raw_colorspace"] = "srgb"
        normalized["raw_lut_enabled"] = bool(normalized.get("raw_lut_enabled", False))
        normalized["raw_lut_path"] = str(normalized.get("raw_lut_path", ""))
        if normalized["working_space"] not in ("srgb", "linear"):
            normalized["working_space"] = "srgb"
        if normalized["output_transform"] not in ("srgb", "gamma22", "gamma18", "linear"):
            normalized["output_transform"] = "srgb"
        if normalized["icc_policy"] not in ("srgb", "preserve_source", "none"):
            normalized["icc_policy"] = "srgb"
        return normalized

    def _load_color_controls(self):
        self._sync_color_controls = True
        try:
            self._input_profile_var.set(self._color_settings["input_profile"])
            self._raw_wb_var.set(self._color_settings["raw_white_balance"])
            self._raw_colorspace_var.set(self._color_settings["raw_colorspace"])
            self._raw_lut_enabled_var.set(self._color_settings["raw_lut_enabled"])
            self._working_space_var.set(self._color_settings["working_space"])
            self._output_transform_var.set(self._color_settings["output_transform"])
            self._icc_policy_var.set(self._color_settings["icc_policy"])
        finally:
            self._sync_color_controls = False
        self._update_raw_lut_label()
        self._update_acceleration_status()

    def _on_color_settings_changed(self, *_args):
        if self._sync_color_controls or self._sync_profile_load:
            return
        prev_input_profile = self._color_settings.get("input_profile")
        prev_raw_wb = self._color_settings.get("raw_white_balance")
        prev_raw_cs = self._color_settings.get("raw_colorspace")
        prev_raw_lut_enabled = self._color_settings.get("raw_lut_enabled")
        prev_raw_lut_path = self._color_settings.get("raw_lut_path")
        self._color_settings = self._normalize_color_settings(
            {
                "input_profile": self._input_profile_var.get(),
                "raw_white_balance": self._raw_wb_var.get(),
                "raw_colorspace": self._raw_colorspace_var.get(),
                "raw_lut_enabled": self._raw_lut_enabled_var.get(),
                "raw_lut_path": self._color_settings.get("raw_lut_path", ""),
                "working_space": self._working_space_var.get(),
                "output_transform": self._output_transform_var.get(),
                "icc_policy": self._icc_policy_var.get(),
            }
        )
        self._update_raw_lut_label()
        if self._source_is_raw and (
            prev_raw_wb != self._color_settings.get("raw_white_balance")
            or prev_raw_cs != self._color_settings.get("raw_colorspace")
            or prev_raw_lut_enabled != self._color_settings.get("raw_lut_enabled")
            or prev_raw_lut_path != self._color_settings.get("raw_lut_path")
        ):
            self._reload_current_image_for_color_settings()
            return
        if self._source_native_array is not None and prev_input_profile != self._color_settings.get("input_profile"):
            self._rebuild_loaded_source_for_color_settings()
            return
        if self.preview_array is not None:
            self._enqueue_render()

    def _update_source_profile_label(self):
        if self._source_is_raw:
            wb = self._color_settings.get("raw_white_balance", "camera")
            cs = self._color_settings.get("raw_colorspace", "srgb")
            lut_name = os.path.basename(self._color_settings.get("raw_lut_path", "")) if self._color_settings.get("raw_lut_enabled") else "off"
            self._source_profile_var.set(f"RAW decode: wb={wb} · color={cs} · lut={lut_name}")
            return
        if not self._source_metadata.get("icc_profile"):
            self._source_profile_var.set("Source ICC: none")
            return
        applied = self._source_metadata.get("input_profile_applied", "present")
        self._source_profile_var.set(f"Source ICC: present · {applied}")

    def _update_raw_lut_label(self):
        path = self._color_settings.get("raw_lut_path", "")
        if not path:
            self._raw_lut_label_var.set("RAW LUT: none")
            return
        state = "on" if self._color_settings.get("raw_lut_enabled") else "off"
        self._raw_lut_label_var.set(f"RAW LUT: {os.path.basename(path)} ({state})")

    def _select_raw_lut(self):
        path = filedialog.askopenfilename(
            title="Select RAW LUT",
            filetypes=[("CUBE LUT", "*.cube"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            self._get_raw_lut(path)
        except Exception as ex:
            messagebox.showerror("RAW LUT Error", str(ex))
            return
        self._color_settings["raw_lut_path"] = os.path.abspath(path)
        self._color_settings["raw_lut_enabled"] = True
        self._load_color_controls()
        if self._source_is_raw:
            self._reload_current_image_for_color_settings()
        elif self.preview_array is not None:
            self._enqueue_render()

    def _clear_raw_lut(self):
        self._color_settings["raw_lut_path"] = ""
        self._color_settings["raw_lut_enabled"] = False
        self._loaded_raw_lut = None
        self._loaded_raw_lut_path = ""
        self._load_color_controls()
        if self._source_is_raw:
            self._reload_current_image_for_color_settings()
        elif self.preview_array is not None:
            self._enqueue_render()

    def _get_raw_lut(self, path):
        abs_path = os.path.abspath(path)
        if self._loaded_raw_lut is not None and self._loaded_raw_lut_path == abs_path:
            return self._loaded_raw_lut
        lut = load_cube_lut(abs_path)
        self._loaded_raw_lut = lut
        self._loaded_raw_lut_path = abs_path
        return lut

    def _apply_input_profile_to_array(self, array, metadata, color_settings=None):
        color_cfg = self._normalize_color_settings(color_settings or self._color_settings)
        out = array.astype(np.float32, copy=True)
        meta = dict(metadata or {})
        icc_bytes = meta.get("icc_profile")
        if not icc_bytes:
            meta["input_profile_applied"] = "assume_srgb"
            return out, meta
        if color_cfg["input_profile"] == "ignore":
            meta["input_profile_applied"] = "ignored_embedded"
            return out, meta
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            dst_profile = ImageCms.createProfile("sRGB")
            pil = Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
            converted = ImageCms.profileToProfile(pil, src_profile, dst_profile, outputMode="RGB")
            meta["input_profile_applied"] = "embedded_to_srgb"
            return np.array(converted, dtype=np.float32) / 255.0, meta
        except Exception as ex:
            meta["input_profile_applied"] = f"icc_failed:{type(ex).__name__}"
            return out, meta

    def _rebuild_loaded_source_for_color_settings(self):
        if self._source_native_array is None:
            if self.preview_array is not None:
                self._enqueue_render()
            return
        self._store_active_face_profile()
        full, metadata = self._apply_input_profile_to_array(self._source_native_array, self._source_metadata, self._color_settings)
        self._image_revision += 1
        self.full_array = full
        self.preview_array, self.preview_scale = self._build_preview_proxy(full)
        self._source_metadata = metadata
        self._update_source_profile_label()
        self.preview_image = None
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self._clear_mask_history()
        self._refresh_mask_tool_state()
        self.status_var.set("Rebuilding from input ICC ...")
        self._run_segmentation(self._image_revision)

    def _reload_current_image_for_color_settings(self):
        if not self.file_path:
            return
        project_state = self._serialize_project_state()
        self.status_var.set("Reloading RAW decode ...")
        threading.Thread(target=self._load_thread, args=(self.file_path, project_state), daemon=True).start()

    def _normalize_layer_options(self, layer_options):
        merged = self._default_layer_options()
        for layer, cfg in (layer_options or {}).items():
            if layer not in merged:
                continue
            merged[layer].update(
                {
                    "enabled": bool(cfg.get("enabled", merged[layer]["enabled"])),
                    "opacity": float(np.clip(cfg.get("opacity", merged[layer]["opacity"]), 0.0, 100.0)),
                    "blend_mode": str(cfg.get("blend_mode", merged[layer]["blend_mode"])),
                }
            )
        return merged

    def _normalize_layer_order(self, layer_order):
        order = [layer for layer in (layer_order or []) if layer in MASK_ORDER]
        for layer in MASK_ORDER:
            if layer not in order:
                order.append(layer)
        return order

    def _get_selective_params(self):
        return {
            layer: {k: v.get() for k, v in self._sliders[layer].items()}
            for layer in MASK_ORDER
            if layer in self._sliders
        }

    def _apply_selective_params(self, params):
        for layer in MASK_ORDER:
            if layer not in self._sliders:
                continue
            layer_params = params.get(layer, {})
            defaults = {k: d for k, _, _, _, d in ALL_LAYERS[layer]}
            for key, var in self._sliders[layer].items():
                var.set(layer_params.get(key, defaults[key]))

    def _copy_masks_dict(self, masks):
        if masks is None:
            return None
        return {key: value.copy() for key, value in masks.items()}

    def _current_profile_key(self):
        if len(self._detected_faces) == 0:
            return -1
        return int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))

    def _store_active_face_profile(self):
        if self.preview_masks is None:
            return

        idx = self._current_profile_key()
        self._face_profiles[idx] = {
            "params": self._get_selective_params(),
            "layer_options": self._get_layer_options(),
            "layer_order": list(self._layer_order),
            "preview_masks": self._copy_masks_dict(self.preview_masks),
            "full_masks": self._copy_masks_dict(self.full_masks),
            "auto_preview_masks": self._copy_masks_dict(self._auto_preview_masks),
            "auto_full_masks": self._copy_masks_dict(self._auto_full_masks),
        }

    def _restore_face_profile(self, face_index):
        key = -1 if len(self._detected_faces) == 0 else face_index
        profile = self._face_profiles.get(key)
        if profile is None:
            return False

        self._sync_profile_load = True
        try:
            self._apply_selective_params(profile.get("params", {}))
            self._layer_options = self._normalize_layer_options(profile.get("layer_options"))
            self._layer_order = self._normalize_layer_order(profile.get("layer_order", MASK_ORDER))
            self.preview_masks = self._copy_masks_dict(profile.get("preview_masks"))
            self.full_masks = self._copy_masks_dict(profile.get("full_masks"))
            self._auto_preview_masks = self._copy_masks_dict(profile.get("auto_preview_masks"))
            self._auto_full_masks = self._copy_masks_dict(profile.get("auto_full_masks"))
        finally:
            self._sync_profile_load = False

        self._update_layer_order_label()
        self._load_active_layer_options()
        self._clear_mask_history()
        self._push_mask_history()
        return True

    def _update_face_picker_controls(self):
        if len(self._detected_faces) == 0:
            self._face_choices = ["Auto"]
            self._face_choice_var.set("Auto")
            self._face_picker.configure(values=self._face_choices, state="disabled")
            self._active_face_index = 0
            return

        self._face_choices = []
        for idx, (x, y, w, h) in enumerate(self._detected_faces):
            self._face_choices.append(f"Face {idx + 1} ({w}x{h} @ {x},{y})")

        self._active_face_index = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        self._face_choice_var.set(self._face_choices[self._active_face_index])
        self._face_picker.configure(values=self._face_choices, state="readonly")

    def _on_face_picker_changed(self, _event=None):
        choice = self._face_choice_var.get()
        if choice not in self._face_choices:
            return

        idx = self._face_choices.index(choice)
        if idx == self._active_face_index:
            return

        self._store_active_face_profile()
        self._active_face_index = idx
        if self.preview_array is not None:
            self.status_var.set("Switching face target ...")
            self._run_segmentation(self._image_revision)

    def _open_image_path(self, path, project_state=None):
        self.file_path = path
        self.status_var.set(f"Loading {os.path.basename(path)} ...")
        self.update()
        threading.Thread(target=self._load_thread, args=(path, project_state), daemon=True).start()

    def _open_file(self):
        types = [
            (
                "Images",
                "*.cr2 *.CR2 *.nef *.NEF *.arw *.ARW *.dng *.DNG *.jpg *.jpeg *.png *.bmp *.tiff",
            ),
            ("All", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=types)
        if not path:
            return
        self._open_image_path(path)

    def _open_project(self):
        path = filedialog.askopenfilename(
            filetypes=[("Portrait Project", "*.peproj"), ("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                project = json.load(fh)
            image_path = project.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Source image not found: {image_path}")
            self._open_image_path(image_path, project_state=project)
        except Exception as ex:
            messagebox.showerror("Project Load Error", str(ex))

    def _open_preset(self):
        path = filedialog.askopenfilename(
            filetypes=[("Portrait Preset", "*.pepreset"), ("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                preset = json.load(fh)
            self._apply_preset_state(preset)
            self.status_var.set(f"Preset loaded -> {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Preset Load Error", str(ex))

    def _load_preset_file(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _default_preset_meta(self):
        return {
            "name": "",
            "category": "general",
            "tags": [],
            "saved_at": "",
        }

    def _normalize_preset_meta(self, meta, path=""):
        normalized = self._default_preset_meta()
        normalized.update(meta or {})
        normalized["name"] = str(normalized.get("name") or "").strip() or Path(path).stem or "Untitled"
        normalized["category"] = str(normalized.get("category") or "general").strip().lower() or "general"
        tags = normalized.get("tags", [])
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]
        normalized["tags"] = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        normalized["saved_at"] = str(normalized.get("saved_at") or "")
        return normalized

    def _preset_label(self, entry):
        meta = entry["meta"]
        category = meta.get("category", "general")
        return f"{meta.get('name', entry['path'].stem)} [{category}]"

    def _preset_meta_summary(self, entry):
        if not entry:
            return "No preset selected"
        meta = entry["meta"]
        saved_at = meta.get("saved_at") or "unknown"
        tags = ", ".join(meta.get("tags") or []) or "none"
        return (
            f"Name: {meta.get('name', entry['path'].stem)}\n"
            f"Category: {meta.get('category', 'general')}\n"
            f"Tags: {tags}\n"
            f"Saved: {saved_at}"
        )

    def _filtered_preset_entries(self, entries):
        search = self._preset_search_var.get().strip().lower()
        category = self._preset_category_var.get().strip().lower() or "all"
        filtered = []
        for entry in entries:
            meta = entry["meta"]
            haystack = " ".join(
                [
                    meta.get("name", ""),
                    meta.get("category", ""),
                    " ".join(meta.get("tags", [])),
                    entry["path"].name,
                ]
            ).lower()
            if search and search not in haystack:
                continue
            if category != "all" and meta.get("category", "general") != category:
                continue
            filtered.append(entry)
        return filtered

    def _update_preset_category_choices(self, entries):
        categories = ["all"]
        categories.extend(sorted({entry["meta"].get("category", "general") for entry in entries}))
        self._preset_category_combo.configure(values=tuple(categories))
        current = self._preset_category_var.get().strip().lower() or "all"
        if current not in categories:
            self._preset_category_var.set("all")

    def _preset_library_dir(self):
        path = Path(os.getcwd()) / "presets"
        path.mkdir(exist_ok=True)
        return path

    def _refresh_preset_browser(self):
        preset_dir = self._preset_library_dir()
        entries = []
        for path in sorted(preset_dir.glob("*.pepreset")):
            try:
                preset = self._load_preset_file(path)
            except Exception:
                preset = {}
            entries.append(
                {
                    "path": path,
                    "preset": preset,
                    "meta": self._normalize_preset_meta(preset.get("meta"), path=str(path)),
                }
            )
        self._preset_library_entries = entries
        self._update_preset_category_choices(entries)
        self._preset_listbox.delete(0, tk.END)
        for entry in self._filtered_preset_entries(entries):
            self._preset_listbox.insert(tk.END, self._preset_label(entry))
        self._update_selected_preset_meta()

    def _on_preset_filter_changed(self, *_args):
        if hasattr(self, "_preset_listbox"):
            self._refresh_preset_browser()

    def _selected_browser_preset_path(self):
        selection = self._preset_listbox.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        filtered = self._filtered_preset_entries(self._preset_library_entries)
        if idx < 0 or idx >= len(filtered):
            return None
        return filtered[idx]

    def _update_selected_preset_meta(self):
        entry = self._selected_browser_preset_path()
        self._preset_meta_var.set(self._preset_meta_summary(entry))

    def _apply_selected_browser_preset(self):
        entry = self._selected_browser_preset_path()
        if not entry:
            messagebox.showwarning("Preset Browser", "Select a preset from the library first.")
            return
        try:
            preset = entry["preset"] if entry.get("preset") else self._load_preset_file(entry["path"])
            self._apply_preset_state(preset)
            self.status_var.set(f"Preset loaded -> {entry['path'].name}")
        except Exception as ex:
            messagebox.showerror("Preset Load Error", str(ex))

    def _save_preset_to_library(self):
        name = simpledialog.askstring("Preset Name", "Preset filename:", initialvalue="portrait_recipe", parent=self)
        if not name:
            return
        category = simpledialog.askstring("Preset Category", "Category:", initialvalue="portrait", parent=self)
        if category is None:
            return
        tags_text = simpledialog.askstring(
            "Preset Tags",
            "Comma-separated tags:",
            initialvalue="skin, portrait",
            parent=self,
        )
        if tags_text is None:
            return
        filename = name if name.endswith(".pepreset") else f"{name}.pepreset"
        out = self._preset_library_dir() / filename
        try:
            preset = self._serialize_preset_state()
            preset["meta"] = self._normalize_preset_meta(
                {
                    "name": Path(filename).stem,
                    "category": category,
                    "tags": [part.strip() for part in tags_text.split(",") if part.strip()],
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                path=str(out),
            )
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self._refresh_preset_browser()
            self.status_var.set(f"Preset saved -> {out.name}")
        except Exception as ex:
            messagebox.showerror("Preset Save Error", str(ex))

    def _toggle_split_view(self):
        mode_order = ["off", "split", "before", "side_by_side"]
        current = self._compare_mode_var.get()
        try:
            idx = mode_order.index(current)
        except ValueError:
            idx = 0
        self._compare_mode_var.set(mode_order[(idx + 1) % len(mode_order)])

    def _on_compare_mode_changed(self, *_args):
        self._compare_mode = self._compare_mode_var.get().strip().lower() or "off"
        if self.preview_image:
            self._render_canvas()

    def _prompt_batch_options(self):
        suffix = simpledialog.askstring(
            "Batch Output Suffix",
            "Output filename suffix:",
            initialvalue="_enhanced",
            parent=self,
        )
        if suffix is None:
            return None

        dialog = tk.Toplevel(self)
        dialog.title("Batch Export Options")
        dialog.configure(bg=PANEL)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        format_var = tk.StringVar(value="jpeg")
        skip_completed_var = tk.BooleanVar(value=True)
        result = {"confirmed": False}

        tk.Label(dialog, text="Output Format", font=("Helvetica", 9, "bold"), fg=TEXT, bg=PANEL).pack(
            anchor="w", padx=12, pady=(12, 4)
        )
        fmt_combo = ttk.Combobox(
            dialog,
            values=tuple(self.BATCH_OUTPUT_FORMATS.keys()),
            textvariable=format_var,
            state="readonly",
            width=16,
        )
        fmt_combo.pack(fill=tk.X, padx=12)

        tk.Checkbutton(
            dialog,
            text="Skip already completed files from prior logs",
            variable=skip_completed_var,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL,
            activeforeground=TEXT,
            selectcolor=CARD,
            relief=tk.FLAT,
        ).pack(anchor="w", padx=12, pady=(10, 8))

        btns = tk.Frame(dialog, bg=PANEL)
        btns.pack(fill=tk.X, padx=12, pady=(0, 12))

        def confirm():
            result["confirmed"] = True
            dialog.destroy()

        def cancel():
            dialog.destroy()

        self._btn(btns, "Cancel", cancel).pack(side=tk.RIGHT, padx=(4, 0))
        self._btn(btns, "Start", confirm, style="accent").pack(side=tk.RIGHT)

        dialog.wait_window()
        if not result["confirmed"]:
            return None

        return {
            "suffix": suffix,
            "format": format_var.get(),
            "skip_completed": bool(skip_completed_var.get()),
        }

    def _load_thread(self, path, project_state=None):
        try:
            ext = os.path.splitext(path)[1].lower()
            full, metadata = self._read_image_file(path)

            preview, scale = self._build_preview_proxy(full)

            def commit_loaded():
                self._image_revision += 1
                self.full_array = full
                self.preview_array = preview
                self.preview_scale = scale
                self._source_native_array = metadata.get("source_native_array")
                self._source_is_raw = bool(metadata.get("source_is_raw", False))
                self._detected_faces = []
                self._active_face_index = 0
                self._face_profiles = {}
                self.preview_masks = None
                self.full_masks = None
                self._auto_preview_masks = None
                self._auto_full_masks = None
                self.preview_image = None
                self._source_metadata = {k: v for k, v in metadata.items() if k != "source_native_array"}
                self._update_source_profile_label()
                self._mask_edit_enabled = False
                self._clear_mask_history()
                self._update_face_picker_controls()
                self._refresh_mask_tool_state()
                if project_state is not None:
                    self._apply_project_state(project_state)
                else:
                    self._run_segmentation(self._image_revision)

            self.after(0, commit_loaded)
        except Exception as ex:
            self.after(0, lambda: messagebox.showerror("Load Error", str(ex)))

    def _read_image_file(self, path, color_settings=None):
        ext = os.path.splitext(path)[1].lower()
        metadata = {}
        if ext in (".cr2", ".nef", ".arw", ".dng", ".raw"):
            if not HAS_RAWPY:
                raise RuntimeError("RAW support requires rawpy. Install dependencies from requirements.txt.")
            raw_cfg = self._normalize_color_settings(color_settings or self._color_settings)
            with rawpy.imread(path) as raw:
                postprocess_kwargs = {
                    "output_bps": 16,
                    "no_auto_bright": False,
                }
                if raw_cfg["raw_white_balance"] == "auto":
                    postprocess_kwargs["use_auto_wb"] = True
                    postprocess_kwargs["use_camera_wb"] = False
                else:
                    postprocess_kwargs["use_camera_wb"] = True
                    postprocess_kwargs["use_auto_wb"] = False

                color_space_map = {
                    "srgb": "sRGB",
                    "adobe": "Adobe",
                    "prophoto": "ProPhoto",
                    "xyz": "XYZ",
                    "raw": "raw",
                }
                cs_name = color_space_map.get(raw_cfg["raw_colorspace"], "sRGB")
                cs_value = getattr(rawpy.ColorSpace, cs_name, None)
                if cs_value is not None:
                    postprocess_kwargs["output_color"] = cs_value

                rgb = raw.postprocess(**postprocess_kwargs)
            full = rgb.astype(np.float32) / 65535.0
            if raw_cfg.get("raw_lut_enabled") and raw_cfg.get("raw_lut_path"):
                lut = self._get_raw_lut(raw_cfg["raw_lut_path"])
                full = apply_cube_lut(full, lut)
                metadata["raw_lut_title"] = lut.get("title", os.path.basename(raw_cfg["raw_lut_path"]))
            metadata["input_profile_applied"] = (
                f"raw_decode:{raw_cfg['raw_white_balance']}:{raw_cfg['raw_colorspace']}:"
                f"lut={'on' if raw_cfg.get('raw_lut_enabled') and raw_cfg.get('raw_lut_path') else 'off'}"
            )
            metadata["source_is_raw"] = True
            return full, metadata

        pil = Image.open(path)
        metadata = self._capture_image_metadata(pil)
        pil_rgb = ImageOps.exif_transpose(pil).convert("RGB")
        native = np.array(pil_rgb, dtype=np.float32) / 255.0
        full, metadata = self._apply_input_profile_to_array(native, metadata, color_settings=color_settings)
        metadata["source_native_array"] = native
        metadata["source_is_raw"] = False
        return full, metadata

    def _capture_image_metadata(self, pil_image):
        exif_bytes = b""
        try:
            exif = pil_image.getexif()
            if exif:
                exif_bytes = exif.tobytes()
        except Exception:
            exif_bytes = b""

        metadata = {}
        if exif_bytes:
            metadata["exif"] = exif_bytes
        icc_profile = pil_image.info.get("icc_profile")
        if icc_profile:
            metadata["icc_profile"] = icc_profile
        dpi = pil_image.info.get("dpi")
        if dpi:
            metadata["dpi"] = dpi
        return metadata

    def _supported_image_paths(self, directory):
        paths = []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in self.SUPPORTED_IMAGE_EXTS:
                continue
            paths.append(path)
        return paths

    def _combine_face_masks(self, image_array):
        faces = self.segmenter.list_faces(image_array)
        if not faces:
            masks, guides = self.segmenter.segment_with_guides(image_array, face_index=0)
            return masks, guides, 0

        combined = None
        guide_list = []
        for idx in range(len(faces)):
            masks, guides = self.segmenter.segment_with_guides(image_array, face_index=idx)
            if combined is None:
                combined = {key: value.copy() for key, value in masks.items()}
            else:
                for key, value in masks.items():
                    if key not in combined:
                        combined[key] = value.copy()
                    else:
                        combined[key] = np.maximum(combined[key], value)
            if guides:
                guide_list.append(guides)
        return combined, guide_list or None, len(faces)

    def _build_preview_proxy(self, full_array, max_dim=1600):
        height, width = full_array.shape[:2]
        scale = min(1.0, float(max_dim) / float(max(height, width)))
        if scale >= 0.999:
            return full_array.copy(), 1.0

        preview_w = max(1, int(width * scale))
        preview_h = max(1, int(height * scale))
        preview = resize_image(
            full_array,
            (preview_w, preview_h),
            interpolation=cv2.INTER_AREA,
            acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
        )
        return preview.astype(np.float32), scale

    def _run_segmentation(self, revision):
        if self.preview_array is None or revision != self._image_revision:
            return

        full_h, full_w = self.full_array.shape[:2]
        preview_h, preview_w = self.preview_array.shape[:2]
        backend = self.segmenter.backend_label
        detector = getattr(self.segmenter, "detector_backend_label", "detect=haar")
        subjects = getattr(self.segmenter, "subject_backend_label", "subjects=heuristic")
        facial_hair = getattr(self.segmenter, "facial_hair_backend_label", "f_hair=fallback")
        face_note = f", face {self._active_face_index + 1}" if len(self._detected_faces) > 1 else ""
        self.status_var.set(
            f"Analyzing faces ({backend}; {detector}; {subjects}; {facial_hair}{face_note}) ... preview {preview_w}x{preview_h} (source {full_w}x{full_h})"
        )
        self.update()
        threading.Thread(target=self._segmentation_thread, args=(revision,), daemon=True).start()

    def _segmentation_thread(self, revision):
        try:
            faces = self.segmenter.list_faces(self.preview_array)
            masks, guides = self.segmenter.segment_with_guides(self.preview_array, face_index=self._active_face_index)
            self.after(0, lambda: self._on_segmented(revision, masks, guides, faces))
        except Exception as ex:
            self.after(0, lambda: self._on_segmentation_error(revision, ex))

    def _on_segmentation_error(self, revision, error):
        if revision != self._image_revision:
            return
        self.status_var.set(f"Segmentation error: {error}")
        self._detected_faces = []
        self.preview_masks = None
        self.full_masks = None
        self.preview_guides = None
        self.full_guides = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self._clear_mask_history()
        self._update_face_picker_controls()
        self._refresh_mask_tool_state()
        self._enqueue_render()

    def _on_segmented(self, revision, masks, guides, faces):
        if revision != self._image_revision or self.preview_array is None:
            return

        self._detected_faces = faces or []
        if len(self._detected_faces) > 0:
            self._active_face_index = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        else:
            self._active_face_index = 0
        self._update_face_picker_controls()

        self.preview_masks = masks
        self.preview_guides = guides
        self.full_masks = self._upsample_masks_to_full(masks)
        scale = 1.0 / max(self.preview_scale, 1e-6)
        self.full_guides = _scale_expression_guides(self.preview_guides, scale, scale)
        self._auto_preview_masks = {k: v.copy() for k, v in self.preview_masks.items()}
        self._auto_full_masks = {k: v.copy() for k, v in self.full_masks.items()} if self.full_masks else None
        if not self._restore_face_profile(self._active_face_index):
            self._clear_mask_history()
            self._push_mask_history()

        found = any(masks[k].max() > 0.1 for k in ("face", "eyes"))
        status = "Face detected · All layers ready" if found else "No face detected · Global adjustments only"
        name = os.path.basename(self.file_path)
        full_h, full_w = self.full_array.shape[:2]
        preview_h, preview_w = self.preview_array.shape[:2]
        backend = self.segmenter.backend_label
        detector = getattr(self.segmenter, "detector_backend_label", "detect=haar")
        subjects = getattr(self.segmenter, "subject_backend_label", "subjects=heuristic")
        facial_hair = getattr(self.segmenter, "facial_hair_backend_label", "f_hair=fallback")
        reason = getattr(self.segmenter, "reason_unavailable", "")
        reason_note = f" · {reason}" if reason else ""
        face_note = f"faces={len(self._detected_faces)}, target={self._active_face_index + 1}" if self._detected_faces else "faces=0"
        self.status_var.set(
            f"{name}  {full_w}x{full_h}  -  {status}  [{backend}; {detector}; {subjects}; {facial_hair}{reason_note}; {face_note}]  (preview {preview_w}x{preview_h})"
        )
        self._refresh_mask_tool_state()
        self._enqueue_render()

    def _upsample_masks_to_full(self, preview_masks):
        if self.full_array is None or self.preview_array is None or preview_masks is None:
            return None

        full_h, full_w = self.full_array.shape[:2]
        preview_h, preview_w = self.preview_array.shape[:2]
        if (full_h, full_w) == (preview_h, preview_w):
            return {key: mask.copy() for key, mask in preview_masks.items()}

        full_masks = {}
        for key, mask in preview_masks.items():
            upscaled = resize_image(
                mask,
                (full_w, full_h),
                interpolation=cv2.INTER_LINEAR,
                acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
            )
            full_masks[key] = np.clip(upscaled.astype(np.float32), 0.0, 1.0)
        return full_masks

    def _encode_array(self, array):
        if array is None:
            return None
        payload = zlib.compress(array.astype(array.dtype, copy=False).tobytes())
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "data": base64.b64encode(payload).decode("ascii"),
        }

    def _decode_array(self, payload):
        if payload is None:
            return None
        raw = zlib.decompress(base64.b64decode(payload["data"].encode("ascii")))
        array = np.frombuffer(raw, dtype=np.dtype(payload["dtype"]))
        return array.reshape(payload["shape"]).copy()

    def _encode_masks(self, masks):
        if masks is None:
            return None
        return {key: self._encode_array(value) for key, value in masks.items()}

    def _decode_masks(self, payload):
        if payload is None:
            return None
        return {key: self._decode_array(value) for key, value in payload.items()}

    def _serialize_face_profiles(self):
        out = {}
        for key, profile in self._face_profiles.items():
            out[str(key)] = {
                "params": profile.get("params", {}),
                "layer_options": profile.get("layer_options", {}),
                "layer_order": profile.get("layer_order", list(MASK_ORDER)),
                "preview_masks": self._encode_masks(profile.get("preview_masks")),
                "full_masks": self._encode_masks(profile.get("full_masks")),
                "auto_preview_masks": self._encode_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._encode_masks(profile.get("auto_full_masks")),
            }
        return out

    def _deserialize_face_profiles(self, payload):
        out = {}
        for key, profile in (payload or {}).items():
            out[int(key)] = {
                "params": profile.get("params", {}),
                "layer_options": profile.get("layer_options", {}),
                "layer_order": profile.get("layer_order", list(MASK_ORDER)),
                "preview_masks": self._decode_masks(profile.get("preview_masks")),
                "full_masks": self._decode_masks(profile.get("full_masks")),
                "auto_preview_masks": self._decode_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._decode_masks(profile.get("auto_full_masks")),
            }
        return out

    def _set_global_params(self, params):
        self._sync_profile_load = True
        try:
            layer = "global"
            defaults = {k: d for k, _, _, _, d in ALL_LAYERS[layer]}
            for key, var in self._sliders[layer].items():
                var.set(params.get(key, defaults[key]))
        finally:
            self._sync_profile_load = False

    def _serialize_project_state(self):
        self._store_active_face_profile()
        return {
            "version": 1,
            "image_path": os.path.abspath(self.file_path) if self.file_path else "",
            "global_params": {k: v.get() for k, v in self._sliders.get("global", {}).items()},
            "color_settings": dict(self._color_settings),
            "active_face_index": int(self._active_face_index),
            "detected_faces": [list(face) for face in self._detected_faces],
            "face_profiles": self._serialize_face_profiles(),
        }

    def _serialize_preset_state(self):
        return {
            "version": 1,
            "global_params": {k: v.get() for k, v in self._sliders.get("global", {}).items()},
            "color_settings": dict(self._color_settings),
            "selective_params": self._get_selective_params(),
            "layer_options": self._get_layer_options(),
            "layer_order": list(self._layer_order),
            "meta": self._default_preset_meta(),
        }

    def _apply_preset_state(self, preset):
        self._sync_profile_load = True
        try:
            self._set_global_params(preset.get("global_params", {}))
            self._color_settings = self._normalize_color_settings(preset.get("color_settings"))
            self._apply_selective_params(preset.get("selective_params", {}))
            self._layer_options = self._normalize_layer_options(preset.get("layer_options"))
            self._layer_order = self._normalize_layer_order(preset.get("layer_order", MASK_ORDER))
        finally:
            self._sync_profile_load = False

        self._load_color_controls()
        self._update_layer_order_label()
        self._load_active_layer_options()
        if self.preview_array is not None:
            self._store_active_face_profile()
            self._enqueue_render()

    def _get_preset_render_params(self, preset):
        params = {"global": dict(preset.get("global_params", {}))}
        selective = preset.get("selective_params", {})
        for layer in MASK_ORDER:
            params[layer] = dict(selective.get(layer, {}))
        return params

    def _get_preset_color_settings(self, preset):
        return self._normalize_color_settings(preset.get("color_settings"))

    def _apply_project_state(self, project):
        self._set_global_params(project.get("global_params", {}))
        self._color_settings = self._normalize_color_settings(project.get("color_settings"))
        self._load_color_controls()
        self._detected_faces = [tuple(int(v) for v in face) for face in project.get("detected_faces", [])]
        self._active_face_index = int(project.get("active_face_index", 0))
        if self._detected_faces:
            self._active_face_index = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        else:
            self._active_face_index = 0

        self._face_profiles = self._deserialize_face_profiles(project.get("face_profiles", {}))
        self._update_face_picker_controls()

        if not self._restore_face_profile(self._active_face_index):
            self._run_segmentation(self._image_revision)
            return

        status = f"Project loaded  [{len(self._face_profiles)} face profiles]"
        self.status_var.set(status)
        self._refresh_mask_tool_state()
        self._enqueue_render()

    def _reanalyze(self):
        if self.preview_array is None:
            return
        self._store_active_face_profile()
        self.status_var.set("Re-analyzing ...")
        self._run_segmentation(self._image_revision)

    def _get_all_params(self):
        return {layer: {k: v.get() for k, v in kvs.items()} for layer, kvs in self._sliders.items()}

    def _get_layer_options(self):
        return {layer: dict(cfg) for layer, cfg in self._layer_options.items()}

    def _schedule_update(self):
        if self.preview_array is None:
            return
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(55, self._enqueue_render)

    def _enqueue_render(self):
        if self.preview_array is None:
            return

        if not self._sync_profile_load:
            self._store_active_face_profile()

        params = self._get_all_params()
        layer_opts = self._get_layer_options()
        layer_order = tuple(self._layer_order)

        self._latest_render_request += 1
        request_id = self._latest_render_request
        revision = self._image_revision

        while True:
            try:
                self._render_queue.get_nowait()
                self._render_queue.task_done()
            except Empty:
                break

        self._render_queue.put((revision, request_id, params, layer_opts, layer_order))

    def _render_worker(self):
        while True:
            revision, request_id, params, layer_opts, layer_order = self._render_queue.get()
            try:
                if revision != self._image_revision or self.preview_array is None:
                    continue

                # Skip stale jobs that were superseded before processing began.
                if request_id < self._latest_render_request:
                    continue

                result = process_all_layers(
                    self.preview_array,
                    params,
                    self.preview_masks,
                    geometry=self.preview_guides,
                    layer_order=layer_order,
                    layer_options=layer_opts,
                    color_settings=self._color_settings,
                    runtime_settings=self._runtime_settings,
                )

                if revision != self._image_revision or request_id < self._latest_render_request:
                    continue

                self.preview_image = result
                self.after(0, self._render_canvas)
            finally:
                self._render_queue.task_done()

    def _render_canvas(self):
        if not self.preview_image:
            return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            return

        img = self.preview_image.copy()

        if self._show_mask and self.preview_masks:
            layer = self._active_layer
            if layer != "global" and layer in self.preview_masks:
                mask = self.preview_masks[layer]
                color_hex = LAYER_COLORS.get(layer, "#ffffff")
                cr, cg, cb = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[:, :, 0] = cr / 255.0
                overlay[:, :, 1] = cg / 255.0
                overlay[:, :, 2] = cb / 255.0
                overlay[:, :, 3] = mask * 0.45
                base = np.array(img, dtype=np.float32) / 255.0
                alpha = overlay[:, :, 3:4]
                blended = base * (1 - alpha) + overlay[:, :, :3] * alpha
                img = Image.fromarray(to_uint8(blended))

        original_img = None
        if self._compare_mode in {"split", "before", "side_by_side"} and self.preview_array is not None:
            original_img = Image.fromarray(to_uint8(self.preview_array))

        compare_mode = self._compare_mode
        if compare_mode == "before" and original_img is not None:
            img = original_img
            img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
        elif compare_mode == "side_by_side" and original_img is not None:
            target_w = max(canvas_w // 2 - 8, 1)
            target_h = max(canvas_h, 1)
            left_img = original_img.copy()
            right_img = img.copy()
            left_img.thumbnail((target_w, target_h), Image.LANCZOS)
            right_img.thumbnail((target_w, target_h), Image.LANCZOS)
            pane_h = max(left_img.size[1], right_img.size[1])
            composed = Image.new("RGB", (left_img.size[0] + right_img.size[0] + 12, pane_h), color=BG)
            composed.paste(left_img, (0, (pane_h - left_img.size[1]) // 2))
            composed.paste(right_img, (left_img.size[0] + 12, (pane_h - right_img.size[1]) // 2))
            img = composed
        else:
            img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
            if compare_mode == "split" and original_img is not None:
                original_img.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
                if original_img.size != img.size:
                    original_img = original_img.resize(img.size, Image.LANCZOS)
                split_px = int(np.clip(self._split_position, 0.0, 1.0) * img.size[0])
                split_px = int(np.clip(split_px, 0, img.size[0]))
                composed = original_img.copy()
                if split_px < img.size[0]:
                    right = img.crop((split_px, 0, img.size[0], img.size[1]))
                    composed.paste(right, (split_px, 0))
                img = composed

        disp_w, disp_h = img.size
        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2
        self._display_origin = (ox, oy)
        self._display_size = (disp_w, disp_h)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("preview")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor="center", tags="preview")
        self.canvas.delete("split")
        self.canvas.delete("compare_label")
        if compare_mode == "split":
            split_canvas_x = ox + int(np.clip(self._split_position, 0.0, 1.0) * disp_w)
            self.canvas.create_line(split_canvas_x, oy, split_canvas_x, oy + disp_h, fill=ACCENT, width=2, tags="split")
        elif compare_mode == "side_by_side":
            gap_x = ox + disp_w // 2
            self.canvas.create_text(
                ox + max(disp_w // 4, 40),
                max(oy - 12, 12),
                text="Before",
                fill=TEXT_DIM,
                font=("Helvetica", 9, "bold"),
                tags="compare_label",
            )
            self.canvas.create_text(
                ox + min((disp_w * 3) // 4, disp_w - 40),
                max(oy - 12, 12),
                text="After",
                fill=TEXT_DIM,
                font=("Helvetica", 9, "bold"),
                tags="compare_label",
            )
            self.canvas.create_line(gap_x, oy, gap_x, oy + disp_h, fill=SEP, width=1, tags="split")
        self.canvas.itemconfig(self.hint_id, text="")

    def _toggle_mask_view(self):
        self._show_mask = not self._show_mask
        self._mask_btn.config(bg=ACCENT if self._show_mask else CARD, fg=BG if self._show_mask else TEXT)
        if self.preview_image:
            self._render_canvas()

    def _capture_mask_snapshot(self):
        if self.preview_masks is None:
            return None
        snap = {}
        for layer in self._layer_options:
            if layer in self.preview_masks:
                snap[layer] = self.preview_masks[layer].copy()
        return snap

    def _restore_mask_snapshot(self, snapshot):
        if self.preview_masks is None or snapshot is None:
            return
        for layer, mask in snapshot.items():
            if layer in self.preview_masks:
                self.preview_masks[layer] = mask.copy()
                self._sync_full_mask_from_preview(layer)

    def _clear_mask_history(self):
        self._mask_history = []
        self._mask_history_index = -1
        self._update_mask_history_buttons()

    def _push_mask_history(self):
        snapshot = self._capture_mask_snapshot()
        if snapshot is None:
            return

        if self._mask_history_index < len(self._mask_history) - 1:
            self._mask_history = self._mask_history[: self._mask_history_index + 1]

        self._mask_history.append(snapshot)
        if len(self._mask_history) > self._max_mask_history:
            overflow = len(self._mask_history) - self._max_mask_history
            self._mask_history = self._mask_history[overflow:]
        self._mask_history_index = len(self._mask_history) - 1
        self._update_mask_history_buttons()

    def _update_mask_history_buttons(self):
        can_edit = self._active_layer in self._layer_options and self.preview_masks is not None
        can_undo = can_edit and self._mask_history_index > 0
        can_redo = can_edit and self._mask_history_index >= 0 and self._mask_history_index < len(self._mask_history) - 1
        self._undo_mask_btn.configure(state="normal" if can_undo else "disabled")
        self._redo_mask_btn.configure(state="normal" if can_redo else "disabled")

    def _undo_mask_edit(self):
        if self._mask_history_index <= 0:
            return
        self._mask_history_index -= 1
        self._restore_mask_snapshot(self._mask_history[self._mask_history_index])
        self._update_mask_history_buttons()
        self._enqueue_render()

    def _redo_mask_edit(self):
        if self._mask_history_index < 0 or self._mask_history_index >= len(self._mask_history) - 1:
            return
        self._mask_history_index += 1
        self._restore_mask_snapshot(self._mask_history[self._mask_history_index])
        self._update_mask_history_buttons()
        self._enqueue_render()

    def _refresh_mask_tool_state(self):
        editable = self._active_layer in self._layer_options and self.preview_masks is not None
        state = "normal" if editable else "disabled"

        if not editable and self._mask_edit_enabled:
            self._mask_edit_enabled = False

        self._mask_edit_btn.configure(state=state, bg=ACCENT if self._mask_edit_enabled and editable else CARD)
        self._paint_radio.configure(state=state)
        self._erase_radio.configure(state=state)
        self._brush_scale.configure(state=state)
        self._feather_btn.configure(state=state)
        self._reset_mask_btn.configure(state=state)
        if not editable:
            self._undo_mask_btn.configure(state="disabled")
            self._redo_mask_btn.configure(state="disabled")
        else:
            self._update_mask_history_buttons()

    def _toggle_mask_edit(self):
        if self._active_layer not in self._layer_options or self.preview_masks is None:
            return
        self._mask_edit_enabled = not self._mask_edit_enabled
        if self._mask_edit_enabled and self._mask_history_index < 0:
            self._push_mask_history()
        self._refresh_mask_tool_state()

    def _on_canvas_press(self, event):
        if self._compare_mode == "split" and not self._mask_edit_enabled:
            self._set_split_position_from_canvas(event.x)
            return
        if self._mask_edit_enabled:
            self._mask_stroke_active = True
            self._push_mask_history()
        self._paint_active_mask(event)

    def _on_canvas_drag(self, event):
        if self._compare_mode == "split" and not self._mask_edit_enabled:
            self._set_split_position_from_canvas(event.x)
            return
        self._paint_active_mask(event)

    def _on_canvas_release(self, _event):
        self._mask_stroke_active = False

    def _canvas_to_preview_xy(self, x, y):
        if self.preview_array is None:
            return None

        disp_w, disp_h = self._display_size
        if disp_w <= 0 or disp_h <= 0:
            return None

        ox, oy = self._display_origin
        if x < ox or y < oy or x >= ox + disp_w or y >= oy + disp_h:
            return None

        px = (x - ox) / float(disp_w)
        py = (y - oy) / float(disp_h)
        ih, iw = self.preview_array.shape[:2]
        ix = int(np.clip(px * iw, 0, iw - 1))
        iy = int(np.clip(py * ih, 0, ih - 1))
        return ix, iy

    def _set_split_position_from_canvas(self, x):
        disp_w, _disp_h = self._display_size
        if disp_w <= 0:
            return
        ox, _oy = self._display_origin
        pos = (x - ox) / float(disp_w)
        self._split_position = float(np.clip(pos, 0.0, 1.0))
        if self.preview_image:
            self._render_canvas()

    def _paint_active_mask(self, event):
        if not self._mask_edit_enabled:
            return
        if self._active_layer not in self._layer_options:
            return
        if self.preview_masks is None or self._active_layer not in self.preview_masks:
            return

        coord = self._canvas_to_preview_xy(event.x, event.y)
        if coord is None:
            return

        layer = self._active_layer
        mask = self.preview_masks[layer]
        radius = max(1, int(self._mask_brush_size_var.get()))
        mode = self._mask_paint_mode_var.get()
        value = 1.0 if mode == "paint" else 0.0

        cv2.circle(mask, coord, radius, value, thickness=-1, lineType=cv2.LINE_AA)
        self.preview_masks[layer] = np.clip(mask.astype(np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(layer)
        self._update_mask_history_buttons()
        self._enqueue_render()

    def _sync_full_mask_from_preview(self, layer):
        if self.preview_masks is None or layer not in self.preview_masks:
            return
        if self.full_array is None or self.preview_array is None:
            return

        if self.full_masks is None:
            self.full_masks = {}

        p_mask = self.preview_masks[layer]
        fh, fw = self.full_array.shape[:2]
        ph, pw = self.preview_array.shape[:2]
        if (fh, fw) == (ph, pw):
            self.full_masks[layer] = p_mask.copy()
        else:
            upscaled = cv2.resize(p_mask, (fw, fh), interpolation=cv2.INTER_LINEAR)
            self.full_masks[layer] = np.clip(upscaled.astype(np.float32), 0.0, 1.0)

    def _feather_active_mask(self):
        if self._active_layer not in self._layer_options or self.preview_masks is None:
            return
        layer = self._active_layer
        if layer not in self.preview_masks:
            return

        self._push_mask_history()
        sigma = max(1.0, float(self._mask_brush_size_var.get()) / 6.0)
        feathered = smooth_mask(
            self.preview_masks[layer],
            sigma=sigma,
            acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
        )
        self.preview_masks[layer] = np.clip(feathered.astype(np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(layer)
        self._update_mask_history_buttons()
        self._enqueue_render()

    def _reset_active_mask(self):
        if self._active_layer not in self._layer_options:
            return
        layer = self._active_layer
        if self._auto_preview_masks is None or layer not in self._auto_preview_masks:
            return

        self._push_mask_history()
        self.preview_masks[layer] = self._auto_preview_masks[layer].copy()

        if self._auto_full_masks is not None and layer in self._auto_full_masks:
            if self.full_masks is None:
                self.full_masks = {}
            self.full_masks[layer] = self._auto_full_masks[layer].copy()
        else:
            self._sync_full_mask_from_preview(layer)

        self._update_mask_history_buttons()
        self._enqueue_render()

    def _on_canvas_resize(self, event):
        if self.preview_image:
            self._render_canvas()
        else:
            self.canvas.coords(self.hint_id, event.width // 2, event.height // 2)

    def _export(self):
        if self.full_array is None:
            messagebox.showwarning("No Image", "Open an image first.")
            return

        self._store_active_face_profile()

        out = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tiff")],
            initialfile="portrait_enhanced_v2",
        )
        if not out:
            return

        params = self._get_all_params()
        layer_opts = self._get_layer_options()
        layer_order = tuple(self._layer_order)
        full = self.full_array
        masks = self.full_masks

        self.status_var.set("Exporting full resolution ...")
        self.update()

        def work():
            result = process_all_layers(
                full,
                params,
                masks,
                geometry=self.full_guides,
                layer_order=layer_order,
                layer_options=layer_opts,
                color_settings=self._color_settings,
                runtime_settings=self._runtime_settings,
            )
            ext = os.path.splitext(out)[1].lower()
            save_kwargs = self._build_export_metadata_kwargs(ext, self._color_settings)
            if ext in (".jpg", ".jpeg"):
                result.save(out, "JPEG", quality=96, subsampling=0, **save_kwargs)
            else:
                result.save(out, **save_kwargs)
            self.after(0, lambda: self.status_var.set(f"Exported -> {out}"))
            self.after(0, lambda: messagebox.showinfo("Exported", f"Saved:\n{out}"))

        threading.Thread(target=work, daemon=True).start()

    def _build_export_metadata_kwargs(self, ext, color_settings=None):
        return self._build_export_metadata_kwargs_from(self._source_metadata, ext, color_settings=color_settings)

    def _save_rendered_image(self, result, out_path, source_metadata, color_settings=None):
        ext = os.path.splitext(out_path)[1].lower()
        save_kwargs = self._build_export_metadata_kwargs_from(source_metadata, ext, color_settings=color_settings)
        if ext in (".jpg", ".jpeg"):
            result.save(out_path, "JPEG", quality=96, subsampling=0, **save_kwargs)
        else:
            result.save(out_path, **save_kwargs)

    def _build_export_metadata_kwargs_from(self, metadata, ext, color_settings=None):
        source_metadata = dict(metadata or {})
        color_cfg = self._normalize_color_settings(color_settings or self._color_settings)
        save_kwargs = {}
        exif_bytes = source_metadata.get("exif")
        dpi = source_metadata.get("dpi")
        icc_profile = None
        input_applied = str(source_metadata.get("input_profile_applied", ""))

        if color_cfg["icc_policy"] == "preserve_source":
            if input_applied.startswith("embedded_to_srgb"):
                if color_cfg["output_transform"] == "srgb":
                    icc_profile = _SRGB_ICC_BYTES
            else:
                icc_profile = source_metadata.get("icc_profile")
        elif color_cfg["icc_policy"] == "srgb" and color_cfg["output_transform"] == "srgb":
            icc_profile = _SRGB_ICC_BYTES

        if exif_bytes and ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            save_kwargs["exif"] = exif_bytes
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if dpi:
            save_kwargs["dpi"] = dpi
        return save_kwargs

    def _batch_export(self):
        preset_path = filedialog.askopenfilename(
            title="Select Preset",
            filetypes=[("Portrait Preset", "*.pepreset"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if not preset_path:
            return

        input_dir = filedialog.askdirectory(title="Select Input Folder")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return

        options = self._prompt_batch_options()
        if options is None:
            return

        try:
            preset = self._load_preset_file(preset_path)
        except Exception as ex:
            messagebox.showerror("Preset Load Error", str(ex))
            return

        image_paths = self._supported_image_paths(input_dir)
        if not image_paths:
            messagebox.showwarning("No Images", "No supported images found in the selected input folder.")
            return

        self.status_var.set(f"Batch exporting {len(image_paths)} image(s) ...")
        self.update()
        threading.Thread(
            target=self._batch_export_worker,
            args=(image_paths, output_dir, preset, os.path.basename(preset_path), options),
            daemon=True,
        ).start()

    def _batch_export_worker(self, image_paths, output_dir, preset, preset_name, options):
        render_params = self._get_preset_render_params(preset)
        color_settings = self._get_preset_color_settings(preset)
        layer_options = self._normalize_layer_options(preset.get("layer_options"))
        layer_order = tuple(self._normalize_layer_order(preset.get("layer_order", MASK_ORDER)))
        output_ext = self.BATCH_OUTPUT_FORMATS[options["format"]]
        suffix = options["suffix"]
        log_path = os.path.join(output_dir, "batch_export_log.jsonl")
        completed_index = self._load_batch_log_index(log_path) if options.get("skip_completed") else {}
        preset_hash = self._preset_hash(preset)
        batch_key = self._batch_config_key(preset_hash, suffix, options["format"])
        completed = 0
        skipped = 0
        failures = []

        for idx, path in enumerate(image_paths, start=1):
            out_path = self._batch_output_path(path, output_dir, suffix=suffix, output_ext=output_ext)
            if options.get("skip_completed") and self._batch_should_skip(completed_index, batch_key, path, out_path):
                skipped += 1
                self._append_batch_log(
                    log_path,
                    {
                        "ts": self._utc_timestamp(),
                        "status": "skipped",
                        "reason": "already_completed",
                        "source_path": os.path.abspath(path),
                        "output_path": os.path.abspath(out_path),
                        "batch_key": batch_key,
                        "preset_hash": preset_hash,
                        "preset_name": preset_name,
                        "suffix": suffix,
                        "output_format": options["format"],
                    },
                )
                continue
            try:
                self.after(0, lambda i=idx, n=len(image_paths), p=path: self.status_var.set(f"Batch {i}/{n}: {os.path.basename(p)}"))
                full, metadata = self._read_image_file(path, color_settings=color_settings)
                masks, guides, _face_count = self._combine_face_masks(full)
                result = process_all_layers(
                    full,
                    render_params,
                    masks,
                    geometry=guides,
                    layer_order=layer_order,
                    layer_options=layer_options,
                    color_settings=color_settings,
                    runtime_settings=self._runtime_settings,
                )
                self._save_rendered_image(result, out_path, metadata, color_settings=color_settings)
                completed += 1
                self._append_batch_log(
                    log_path,
                    {
                        "ts": self._utc_timestamp(),
                        "status": "success",
                        "source_path": os.path.abspath(path),
                        "output_path": os.path.abspath(out_path),
                        "batch_key": batch_key,
                        "preset_hash": preset_hash,
                        "preset_name": preset_name,
                        "suffix": suffix,
                        "output_format": options["format"],
                    },
                )
            except Exception as ex:
                failures.append(f"{os.path.basename(path)}: {ex}")
                self._append_batch_log(
                    log_path,
                    {
                        "ts": self._utc_timestamp(),
                        "status": "error",
                        "error": str(ex),
                        "source_path": os.path.abspath(path),
                        "output_path": os.path.abspath(out_path),
                        "batch_key": batch_key,
                        "preset_hash": preset_hash,
                        "preset_name": preset_name,
                        "suffix": suffix,
                        "output_format": options["format"],
                    },
                )

        def finish():
            if failures:
                preview = "\n".join(failures[:8])
                extra = "" if len(failures) <= 8 else f"\n... and {len(failures) - 8} more"
                self.status_var.set(f"Batch complete: {completed} succeeded, {skipped} skipped, {len(failures)} failed")
                messagebox.showwarning(
                    "Batch Export Complete",
                    f"Preset: {preset_name}\nSucceeded: {completed}\nSkipped: {skipped}\nFailed: {len(failures)}\nLog: {log_path}\n\n{preview}{extra}",
                )
                return

            self.status_var.set(f"Batch complete: {completed} exported, {skipped} skipped")
            messagebox.showinfo(
                "Batch Export Complete",
                f"Preset: {preset_name}\nExported: {completed}\nSkipped: {skipped}\nOutput: {output_dir}\nLog: {log_path}",
            )

        self.after(0, finish)

    def _batch_output_path(self, src_path, output_dir, suffix="_enhanced", output_ext=".jpg"):
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return os.path.join(output_dir, f"{stem}{suffix}{output_ext}")

    def _preset_hash(self, preset):
        payload = json.dumps(preset, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _batch_config_key(self, preset_hash, suffix, output_format):
        return json.dumps({"preset_hash": preset_hash, "suffix": suffix, "format": output_format}, sort_keys=True)

    def _utc_timestamp(self):
        return datetime.now(timezone.utc).isoformat()

    def _load_batch_log_index(self, log_path):
        index = {}
        if not os.path.exists(log_path):
            return index
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("status") != "success":
                        continue
                    key = (record.get("batch_key"), record.get("source_path"))
                    if all(key):
                        index[key] = record
        except OSError:
            return {}
        return index

    def _batch_should_skip(self, completed_index, batch_key, source_path, output_path):
        key = (batch_key, os.path.abspath(source_path))
        if key not in completed_index:
            return False
        return os.path.exists(output_path)

    def _append_batch_log(self, log_path, record):
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass

    def _load_batch_log_records(self, log_path):
        records = []
        if not os.path.exists(log_path):
            return records
        try:
            with open(log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return records

    def _latest_failed_batch_group(self, records):
        groups = {}
        order = []
        for record in records:
            key = record.get("batch_key")
            if not key:
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(record)

        for key in reversed(order):
            failed = [r for r in groups[key] if r.get("status") == "error" and r.get("source_path")]
            if failed:
                return key, failed[-1], failed
        return None, None, []

    def _retry_failed_batch(self):
        log_path = filedialog.askopenfilename(
            title="Select Batch Log",
            filetypes=[("JSON Lines", "*.jsonl"), ("All", "*.*")],
        )
        if not log_path:
            return

        preset_path = filedialog.askopenfilename(
            title="Select Preset For Retry",
            filetypes=[("Portrait Preset", "*.pepreset"), ("JSON", "*.json"), ("All", "*.*")],
        )
        if not preset_path:
            return

        try:
            preset = self._load_preset_file(preset_path)
        except Exception as ex:
            messagebox.showerror("Preset Load Error", str(ex))
            return

        records = self._load_batch_log_records(log_path)
        batch_key, sample, failed_records = self._latest_failed_batch_group(records)
        if not batch_key or not failed_records:
            messagebox.showinfo("Retry Failed", "No failed batch entries were found in the selected log.")
            return

        preset_hash = self._preset_hash(preset)
        if sample.get("preset_hash") and sample["preset_hash"] != preset_hash:
            messagebox.showerror(
                "Retry Failed",
                "The selected preset does not match the preset content hash recorded in the latest failed batch.",
            )
            return

        output_format = sample.get("output_format", "jpeg")
        suffix = sample.get("suffix", "_enhanced")
        output_dir = os.path.dirname(os.path.abspath(log_path))
        image_paths = []
        seen = set()
        for record in failed_records:
            src = record.get("source_path")
            if not src or src in seen:
                continue
            if not os.path.exists(src):
                continue
            seen.add(src)
            image_paths.append(src)

        if not image_paths:
            messagebox.showwarning("Retry Failed", "No retryable source files exist for the latest failed batch.")
            return

        options = {
            "suffix": suffix,
            "format": output_format,
            "skip_completed": True,
        }
        self.status_var.set(f"Retrying {len(image_paths)} failed image(s) ...")
        self.update()
        threading.Thread(
            target=self._batch_export_worker,
            args=(image_paths, output_dir, preset, os.path.basename(preset_path), options),
            daemon=True,
        ).start()

    def _save_project(self):
        if self.full_array is None or not self.file_path:
            messagebox.showwarning("No Image", "Open an image first.")
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".peproj",
            filetypes=[("Portrait Project", "*.peproj"), ("JSON", "*.json")],
            initialfile="portrait_enhancer_project",
        )
        if not out:
            return

        try:
            project = self._serialize_project_state()
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(project, fh)
            self.status_var.set(f"Project saved -> {out}")
            messagebox.showinfo("Project Saved", f"Saved:\n{out}")
        except Exception as ex:
            messagebox.showerror("Project Save Error", str(ex))

    def _save_preset(self):
        out = filedialog.asksaveasfilename(
            defaultextension=".pepreset",
            filetypes=[("Portrait Preset", "*.pepreset"), ("JSON", "*.json")],
            initialfile="portrait_enhancer_preset",
        )
        if not out:
            return

        try:
            preset = self._serialize_preset_state()
            preset["meta"] = self._normalize_preset_meta(
                {
                    "name": Path(out).stem,
                    "category": "general",
                    "tags": [],
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                },
                path=out,
            )
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self.status_var.set(f"Preset saved -> {out}")
            messagebox.showinfo("Preset Saved", f"Saved:\n{out}")
        except Exception as ex:
            messagebox.showerror("Preset Save Error", str(ex))

    def _reset_layer(self):
        layer = self._active_layer
        defaults = {k: d for k, _, _, _, d in ALL_LAYERS[layer]}
        for key, var in self._sliders[layer].items():
            var.set(defaults[key])
        if layer in self._layer_options:
            if self._auto_preview_masks is not None and layer in self._auto_preview_masks:
                self.preview_masks[layer] = self._auto_preview_masks[layer].copy()
            if self._auto_full_masks is not None and layer in self._auto_full_masks:
                if self.full_masks is None:
                    self.full_masks = {}
                self.full_masks[layer] = self._auto_full_masks[layer].copy()
            self._clear_mask_history()
            self._push_mask_history()
        if self.preview_array is not None:
            self._enqueue_render()

    def _reset_all(self):
        self._sync_profile_load = True
        for layer, sliders in ALL_LAYERS.items():
            defaults = {k: d for k, _, _, _, d in sliders}
            for key, var in self._sliders[layer].items():
                var.set(defaults[key])
        self._sync_profile_load = False

        self._layer_options = self._default_layer_options()
        self._color_settings = self._default_color_settings()
        self._load_color_controls()

        self._layer_order = list(MASK_ORDER)
        self._update_layer_order_label()
        self._load_active_layer_options()

        if self._auto_preview_masks is not None:
            self.preview_masks = self._copy_masks_dict(self._auto_preview_masks)
        if self._auto_full_masks is not None:
            self.full_masks = self._copy_masks_dict(self._auto_full_masks)
        self.preview_guides = None
        self.full_guides = None
        self._face_profiles = {}
        self._clear_mask_history()
        if self.preview_masks is not None:
            self._push_mask_history()

        if self.preview_array is not None:
            self._enqueue_render()


def run_app():
    app = PortraitEnhancerV2()
    app.mainloop()
