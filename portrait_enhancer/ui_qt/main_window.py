"""PySide6 frontend for Portrait Enhancer."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import base64
import subprocess
import signal
import sys
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QSize, QEvent, QPointF, QRectF, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QInputDialog,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QToolButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from portrait_enhancer.config import ALL_LAYERS, LAYER_COLORS, LAYER_NAMES, LAYER_SLIDER_GROUPS, MASK_ORDER
from portrait_enhancer.core import framing as framing_ops
from portrait_enhancer.core.framing import apply_framing, default_framing, normalize_framing
from portrait_enhancer.core.histogram import compute_histogram
from portrait_enhancer.core import white_balance as wb_ops
from portrait_enhancer.core import culling as culling_ops
from portrait_enhancer.core import face_swap as face_swap_ops
from portrait_enhancer.core import tone_curve as tc_ops
from portrait_enhancer.core import color_mixer as cm_ops
from portrait_enhancer.core.masks import apply_mask_adjustments, default_mask_adjustments, normalize_mask_adjustments
from portrait_enhancer.core.processing import (
    process_all_layers,
    StagePipelineCache,
    apply_output_sharpening,
    _offset_expression_guides,
    _scale_expression_guides,
    expression_warp_mode,
    suggest_global_auto_values,
    suggest_region_noise_red,
    suggest_auto_tone,
    suggest_auto_tone_for_face,
    suggest_auto_subject,
    suggest_auto_crop,
)
from portrait_enhancer.core.refine import get_face_refiner
from portrait_enhancer.core.segmentation import FaceSegmenter, MASK_KEYS
from portrait_enhancer.core.analysis_cache import (
    CACHE_VERSION as ANALYSIS_CACHE_VERSION,
    CACHE_KIND_SINGLE_FACE,
    load_analysis,
    save_analysis,
    segmenter_cache_signature,
)
from portrait_enhancer.core.denoise import MLDenoiser, DeepDenoiser
from portrait_enhancer.core.raw_decode import RAW_EXTS, decode_raw, is_raw_path

# Openable files = every RAW format LibRaw handles + the standard 8-bit/16-bit stills.
_NON_RAW_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SUPPORTED_IMAGE_EXTS = tuple(RAW_EXTS) + _NON_RAW_IMAGE_EXTS
from portrait_enhancer.core.utils import resize_image, smooth_mask, to_uint8


class BatchExportDialog(QDialog):
    SUPPORTED_IMAGE_EXTS = SUPPORTED_IMAGE_EXTS  # module-level set (RAW formats + standard stills)

    def __init__(self, parent=None, preset_path="", input_dir="", output_dir="", suffix="_enhanced", output_format="jpeg", skip_completed=True, deep_denoise=False, profiles=None, selected_profile=""):
        super().__init__(parent)
        self.setWindowTitle("Create Batch Export")
        self.setModal(True)
        self.resize(760, 300)
        self._profiles = dict(profiles or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Custom")
        for name in sorted(self._profiles):
            self.profile_combo.addItem(name)
        self.profile_combo.currentTextChanged.connect(self._on_profile_selected)
        profile_row = QWidget(self)
        profile_layout = QHBoxLayout(profile_row)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.addWidget(self.profile_combo, 1)
        save_profile_btn = QPushButton("Save Profile")
        save_profile_btn.clicked.connect(self._save_profile)
        profile_layout.addWidget(save_profile_btn)
        rename_profile_btn = QPushButton("Rename")
        rename_profile_btn.clicked.connect(self._rename_profile)
        profile_layout.addWidget(rename_profile_btn)
        duplicate_profile_btn = QPushButton("Duplicate")
        duplicate_profile_btn.clicked.connect(self._duplicate_profile)
        profile_layout.addWidget(duplicate_profile_btn)
        delete_profile_btn = QPushButton("Delete")
        delete_profile_btn.clicked.connect(self._delete_profile)
        profile_layout.addWidget(delete_profile_btn)
        form.addRow("Profile", profile_row)

        self.preset_edit = QLineEdit(preset_path)
        preset_row = self._path_row(self.preset_edit, self._browse_preset)
        form.addRow("Preset", preset_row)

        self.input_edit = QLineEdit(input_dir)
        self.input_edit.textChanged.connect(self._update_estimate)
        input_row = self._path_row(self.input_edit, self._browse_input_dir)
        form.addRow("Input Folder", input_row)

        self.output_edit = QLineEdit(output_dir)
        self.output_edit.textChanged.connect(self._update_estimate)
        output_row = self._path_row(self.output_edit, self._browse_output_dir)
        form.addRow("Output Folder", output_row)

        self.suffix_edit = QLineEdit(suffix)
        self.suffix_edit.textChanged.connect(self._update_estimate)
        form.addRow("Suffix", self.suffix_edit)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpeg", "png", "tiff"])
        self.format_combo.setCurrentText(output_format)
        self.format_combo.currentTextChanged.connect(self._update_estimate)
        form.addRow("Output Format", self.format_combo)

        self.skip_completed_check = QCheckBox("Skip already completed files from prior runs")
        self.skip_completed_check.setChecked(bool(skip_completed))
        self.skip_completed_check.toggled.connect(self._update_estimate)
        layout.addWidget(self.skip_completed_check)

        self.deep_denoise_check = QCheckBox("Apply Deep Denoise during export", self)
        self.deep_denoise_check.setChecked(bool(deep_denoise))
        self.deep_denoise_check.setToolTip(
            "Run the heavy full-resolution Deep Denoise model on every exported image. "
            "This can add minutes per image."
        )
        layout.addWidget(self.deep_denoise_check)

        self.estimate_label = QLabel("Estimate: select an input folder")
        self.estimate_label.setWordWrap(True)
        layout.addWidget(self.estimate_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if selected_profile and selected_profile in self._profiles:
            self.profile_combo.setCurrentText(selected_profile)
        self._update_estimate()

    def _current_profile_payload(self):
        return {
            "preset_path": self.preset_edit.text().strip(),
            "input_dir": self.input_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "suffix": self.suffix_edit.text(),
            "format": self.format_combo.currentText(),
            "skip_completed": self.skip_completed_check.isChecked(),
            "deep_denoise": self.deep_denoise_check.isChecked(),
        }

    def _apply_profile(self, payload):
        if not isinstance(payload, dict):
            return
        self.preset_edit.setText(str(payload.get("preset_path", "")).strip())
        self.input_edit.setText(str(payload.get("input_dir", "")).strip())
        self.output_edit.setText(str(payload.get("output_dir", "")).strip())
        self.suffix_edit.setText(str(payload.get("suffix", "_enhanced")))
        self.format_combo.setCurrentText(str(payload.get("format", "jpeg")))
        self.skip_completed_check.setChecked(bool(payload.get("skip_completed", True)))
        self.deep_denoise_check.setChecked(bool(payload.get("deep_denoise", False)))
        self._update_estimate()

    def _on_profile_selected(self, name: str):
        if name == "Custom":
            return
        self._apply_profile(self._profiles.get(name, {}))

    def _save_profile(self):
        name, ok = QInputDialog.getText(self, "Save Batch Profile", "Profile Name", text=self.profile_combo.currentText() if self.profile_combo.currentText() != "Custom" else "")
        name = str(name).strip()
        if not ok or not name:
            return
        self._profiles[name] = self._current_profile_payload()
        if self.profile_combo.findText(name) < 0:
            self.profile_combo.addItem(name)
        self.profile_combo.setCurrentText(name)

    def _rename_profile(self):
        current_name = self.profile_combo.currentText()
        if not current_name or current_name == "Custom":
            return
        new_name, ok = QInputDialog.getText(self, "Rename Batch Profile", "Profile Name", text=current_name)
        new_name = str(new_name).strip()
        if not ok or not new_name or new_name == current_name:
            return
        if new_name in self._profiles:
            QMessageBox.warning(self, "Rename Batch Profile", "A batch profile with that name already exists.")
            return
        payload = self._profiles.pop(current_name, None)
        if payload is None:
            return
        self._profiles[new_name] = payload
        idx = self.profile_combo.findText(current_name)
        if idx >= 0:
            self.profile_combo.setItemText(idx, new_name)
            self.profile_combo.setCurrentText(new_name)

    def _duplicate_profile(self):
        base_name = self.profile_combo.currentText()
        if not base_name or base_name == "Custom":
            base_name = "Batch Profile"
        new_name, ok = QInputDialog.getText(self, "Duplicate Batch Profile", "New Profile Name", text=f"{base_name} Copy")
        new_name = str(new_name).strip()
        if not ok or not new_name:
            return
        if new_name in self._profiles:
            QMessageBox.warning(self, "Duplicate Batch Profile", "A batch profile with that name already exists.")
            return
        self._profiles[new_name] = self._current_profile_payload()
        self.profile_combo.addItem(new_name)
        self.profile_combo.setCurrentText(new_name)

    def _delete_profile(self):
        name = self.profile_combo.currentText()
        if not name or name == "Custom":
            return
        self._profiles.pop(name, None)
        idx = self.profile_combo.findText(name)
        if idx >= 0:
            self.profile_combo.removeItem(idx)
        self.profile_combo.setCurrentText("Custom")

    def _supported_image_paths(self, input_dir):
        paths = []
        if not input_dir or not os.path.isdir(input_dir):
            return paths
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() not in self.SUPPORTED_IMAGE_EXTS:
                continue
            paths.append(path)
        return paths

    def _update_estimate(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        suffix = self.suffix_edit.text()
        output_ext = {"jpeg": ".jpg", "png": ".png", "tiff": ".tiff"}.get(self.format_combo.currentText(), ".jpg")
        image_paths = self._supported_image_paths(input_dir)
        if not image_paths:
            self.estimate_label.setText("Estimate: 0 supported images found")
            return
        skipped = 0
        if self.skip_completed_check.isChecked() and output_dir:
            for source_path in image_paths:
                stem = Path(source_path).stem
                output_path = os.path.join(output_dir, f"{stem}{suffix}{output_ext}")
                if os.path.exists(output_path):
                    skipped += 1
        process_count = max(0, len(image_paths) - skipped)
        self.estimate_label.setText(
            f"Estimate: {len(image_paths)} supported images | "
            f"{process_count} to process | "
            f"{skipped} likely skipped"
        )

    def _path_row(self, line_edit, browse_fn):
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit, 1)
        button = QPushButton("Browse")
        button.clicked.connect(browse_fn)
        row_layout.addWidget(button)
        return row

    def _browse_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Preset", "", "Portrait Preset (*.pepreset *.json)")
        if path:
            self.preset_edit.setText(path)

    def _browse_input_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.input_edit.text())
        if path:
            self.input_edit.setText(path)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def options(self):
        return {
            "preset_path": self.preset_edit.text().strip(),
            "input_dir": self.input_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "suffix": self.suffix_edit.text(),
            "format": self.format_combo.currentText(),
            "skip_completed": self.skip_completed_check.isChecked(),
            "deep_denoise": self.deep_denoise_check.isChecked(),
            "profiles": dict(self._profiles),
            "selected_profile": self.profile_combo.currentText() if self.profile_combo.currentText() != "Custom" else "",
        }


class ExportDialog(QDialog):
    """Single-image export: format, quality, resize, metadata, destination."""

    FORMATS = (
        ("JPEG", "jpeg", ".jpg"),
        ("PNG", "png", ".png"),
        ("TIFF", "tiff", ".tif"),
    )

    OUTPUT_SHARPENING_OPTIONS = (
        ("Off", "off"),
        ("Low", "low"),
        ("Standard", "standard"),
        ("High", "high"),
    )

    def __init__(self, parent=None, output_dir="", stem="export", output_format="jpeg",
                 quality=92, has_metadata=False, source_w=0, source_h=0, output_sharpening="standard",
                 deep_denoise=False):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.setModal(True)
        self.resize(560, 0)
        self._stem = stem or "export"
        self._source_w = int(source_w)
        self._source_h = int(source_h)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.format_combo = QComboBox(self)
        for label, value, _ext in self.FORMATS:
            self.format_combo.addItem(label, value)
        idx = self.format_combo.findData(output_format)
        self.format_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Format", self.format_combo)

        quality_row = QWidget(self)
        quality_layout = QHBoxLayout(quality_row)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        self.quality_slider = QSlider(Qt.Horizontal, self)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(int(quality))
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value_label.setText(str(int(v))))
        quality_layout.addWidget(self.quality_slider, 1)
        self.quality_value_label = QLabel(str(int(quality)), self)
        quality_layout.addWidget(self.quality_value_label)
        self.quality_row = quality_row
        form.addRow("JPEG Quality", quality_row)

        resize_row = QWidget(self)
        resize_layout = QHBoxLayout(resize_row)
        resize_layout.setContentsMargins(0, 0, 0, 0)
        self.resize_check = QCheckBox("Limit long edge to", self)
        self.resize_check.toggled.connect(self._on_resize_toggled)
        resize_layout.addWidget(self.resize_check)
        self.resize_spin = QSpinBox(self)
        self.resize_spin.setRange(64, 20000)
        self.resize_spin.setSingleStep(64)
        longest = max(self._source_w, self._source_h) or 2048
        self.resize_spin.setValue(min(longest, 2048))
        self.resize_spin.setSuffix(" px")
        self.resize_spin.setEnabled(False)
        self.resize_spin.valueChanged.connect(lambda _v: self._update_preview())
        resize_layout.addWidget(self.resize_spin)
        resize_layout.addStretch(1)
        form.addRow("Resize", resize_row)

        self.metadata_check = QCheckBox("Keep EXIF / ICC / DPI from source", self)
        self.metadata_check.setChecked(bool(has_metadata))
        self.metadata_check.setEnabled(bool(has_metadata))
        if not has_metadata:
            self.metadata_check.setToolTip("Source has no embeddable metadata (or is a RAW file)")
        form.addRow("Metadata", self.metadata_check)

        self.output_sharpening_combo = QComboBox(self)
        for label, value in self.OUTPUT_SHARPENING_OPTIONS:
            self.output_sharpening_combo.addItem(label, value)
        sharp_idx = self.output_sharpening_combo.findData(output_sharpening)
        self.output_sharpening_combo.setCurrentIndex(sharp_idx if sharp_idx >= 0 else 2)
        self.output_sharpening_combo.setToolTip(
            "A final sharpening pass calibrated for this export's actual pixel size -- distinct "
            "from the Sharpness slider, which is tuned for the working/preview resolution."
        )
        form.addRow("Output Sharpening", self.output_sharpening_combo)

        self.deep_denoise_check = QCheckBox("Apply Deep Denoise during export", self)
        self.deep_denoise_check.setChecked(bool(deep_denoise))
        self.deep_denoise_check.setToolTip(
            "Run the full-resolution Deep Denoise model for this export without changing the live preview toggle."
        )
        form.addRow("Deep Denoise", self.deep_denoise_check)

        self.dest_edit = QLineEdit(output_dir, self)
        self.dest_edit.textChanged.connect(lambda _t: self._update_preview())
        dest_row = QWidget(self)
        dest_layout = QHBoxLayout(dest_row)
        dest_layout.setContentsMargins(0, 0, 0, 0)
        dest_layout.addWidget(self.dest_edit, 1)
        browse_btn = QPushButton("Browse", self)
        browse_btn.clicked.connect(self._browse_dest)
        dest_layout.addWidget(browse_btn)
        form.addRow("Destination", dest_row)

        self.name_edit = QLineEdit(f"{self._stem}_export", self)
        self.name_edit.textChanged.connect(lambda _t: self._update_preview())
        form.addRow("Filename", self.name_edit)

        self.preview_label = QLabel(self)
        self.preview_label.setObjectName("MutedLabel")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.export_btn = buttons.addButton("Export", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_format_changed()

    def _current_format(self):
        return self.format_combo.currentData()

    def _current_ext(self):
        value = self._current_format()
        for _label, fmt, ext in self.FORMATS:
            if fmt == value:
                return ext
        return ".jpg"

    def _on_format_changed(self):
        is_jpeg = self._current_format() == "jpeg"
        self.quality_row.setEnabled(is_jpeg)
        self._update_preview()

    def _on_resize_toggled(self, checked: bool):
        self.resize_spin.setEnabled(bool(checked))
        self._update_preview()

    def _browse_dest(self):
        path = QFileDialog.getExistingDirectory(self, "Select Destination Folder", self.dest_edit.text())
        if path:
            self.dest_edit.setText(path)

    def _update_preview(self):
        ext = self._current_ext()
        name = self.name_edit.text().strip() or f"{self._stem}_export"
        dims = ""
        if self._source_w and self._source_h:
            w, h = self._source_w, self._source_h
            if self.resize_check.isChecked():
                limit = int(self.resize_spin.value())
                longest = max(w, h)
                if longest > limit:
                    scale = limit / float(longest)
                    w, h = max(1, round(w * scale)), max(1, round(h * scale))
            dims = f"  ·  {w}×{h}px"
        self.preview_label.setText(f"Will save: {name}{ext}{dims}")

    def options(self):
        ext = self._current_ext()
        name = self.name_edit.text().strip() or f"{self._stem}_export"
        return {
            "format": self._current_format(),
            "ext": ext,
            "quality": int(self.quality_slider.value()),
            "resize_long_edge": int(self.resize_spin.value()) if self.resize_check.isChecked() else 0,
            "keep_metadata": self.metadata_check.isChecked() and self.metadata_check.isEnabled(),
            "output_sharpening": self.output_sharpening_combo.currentData(),
            "deep_denoise": self.deep_denoise_check.isChecked(),
            "out_path": os.path.join(self.dest_edit.text().strip(), f"{name}{ext}"),
            "dest_dir": self.dest_edit.text().strip(),
        }


class BatchJobsDialog(QDialog):
    def __init__(self, parent=None, output_dir="", load_jobs_callback=None, cancel_job_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Jobs")
        self.setModal(True)
        self.resize(980, 620)
        self._jobs = []
        self._load_jobs_callback = load_jobs_callback
        self._cancel_job_callback = cancel_job_callback
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(2000)
        self._refresh_timer.timeout.connect(self.refresh_jobs)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.output_edit = QLineEdit(output_dir)
        output_row = QWidget(self)
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.addWidget(self.output_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_output_dir)
        output_row_layout.addWidget(browse_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_jobs)
        output_row_layout.addWidget(refresh_btn)
        form.addRow("Output Folder", output_row)

        refresh_options_row = QWidget(self)
        refresh_options_layout = QHBoxLayout(refresh_options_row)
        refresh_options_layout.setContentsMargins(0, 0, 0, 0)
        self.auto_refresh_check = QCheckBox("Auto Refresh")
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.toggled.connect(self._on_auto_refresh_toggled)
        refresh_options_layout.addWidget(self.auto_refresh_check)
        refresh_options_layout.addWidget(QLabel("Every 2s"))
        refresh_options_layout.addStretch(1)
        form.addRow("Refresh", refresh_options_row)

        body = QHBoxLayout()
        layout.addLayout(body, 1)

        self.jobs_list = QListWidget()
        self.jobs_list.currentRowChanged.connect(self._on_job_selected)
        body.addWidget(self.jobs_list, 1)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        body.addWidget(self.details, 2)

        action_row = QHBoxLayout()
        self.cancel_job_btn = QPushButton("Cancel Selected Export")
        self.cancel_job_btn.setEnabled(False)
        self.cancel_job_btn.clicked.connect(self._cancel_selected_job)
        action_row.addWidget(self.cancel_job_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)
        self._on_auto_refresh_toggled(True)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_edit.text())
        if path:
            self.output_edit.setText(path)
            self.refresh_jobs()

    def showEvent(self, event):
        super().showEvent(event)
        if self.auto_refresh_check.isChecked():
            self._refresh_timer.start()
        self.refresh_jobs()

    def closeEvent(self, event):
        self._refresh_timer.stop()
        super().closeEvent(event)

    def output_dir(self):
        return self.output_edit.text().strip()

    def set_jobs(self, jobs):
        selected_job_id = None
        current_index = self.jobs_list.currentRow()
        if 0 <= current_index < len(self._jobs):
            selected_job_id = self._jobs[current_index].get("job_id")
        self._jobs = list(jobs)
        self.jobs_list.clear()
        selected_index = -1
        for job in self._jobs:
            resource = job.get("resource_usage") or {}
            resource_text = ""
            if resource:
                resource_text = f" cpu={resource.get('cpu_percent', 0.0):.0f}% rss={resource.get('rss_mb', 0.0):.0f}MB"
            summary = (
                f"{job.get('created_at', 'unknown')} | "
                f"{job.get('mode', 'batch')} | "
                f"{job.get('status', 'unknown')} | "
                f"ok={job.get('success', 0)} skip={job.get('skipped', 0)} "
                f"err={job.get('error', 0)} cancel={job.get('canceled', 0)}{resource_text}"
            )
            self.jobs_list.addItem(summary)
            if selected_job_id and job.get("job_id") == selected_job_id:
                selected_index = self.jobs_list.count() - 1
        if self._jobs:
            self.jobs_list.setCurrentRow(selected_index if selected_index >= 0 else 0)
        else:
            self.details.setPlainText("No batch jobs found in the selected output folder.")
            self.cancel_job_btn.setEnabled(False)

    def _on_job_selected(self, index):
        if index < 0 or index >= len(self._jobs):
            self.details.clear()
            self.cancel_job_btn.setEnabled(False)
            return
        job = self._jobs[index]
        self.cancel_job_btn.setEnabled(job.get("status") in {"queued", "running"} and self._cancel_job_callback is not None)
        lines = [
            f"Job ID: {job.get('job_id', 'unknown')}",
            f"Mode: {job.get('mode', 'batch')}",
            f"Created: {job.get('created_at', 'unknown')}",
            f"Status: {job.get('status', 'unknown')}",
            f"Preset: {job.get('preset_path', '')}",
            f"Input Dir: {job.get('input_dir', '')}",
            f"Output Dir: {job.get('output_dir', '')}",
            f"Suffix: {job.get('suffix', '')}",
            f"Format: {job.get('output_format', '')}",
            f"Skip Completed: {job.get('skip_completed', False)}",
            f"Source Count: {job.get('source_count', 0)}",
            f"Success: {job.get('success', 0)}",
            f"Skipped: {job.get('skipped', 0)}",
            f"Errors: {job.get('error', 0)}",
            f"Canceled: {job.get('canceled', 0)}",
            f"Runner PID: {job.get('runner_pid', '')}",
            "",
            f"Job File: {job.get('job_path', '')}",
            f"Batch Log: {job.get('log_path', '')}",
            f"Runner Log: {job.get('runner_log_path', '')}",
        ]
        resource = job.get("resource_usage") or {}
        worker_plan = job.get("worker_plan") or {}
        system_resource = job.get("system_resource") or {}
        denoise_progress = job.get("deep_denoise_progress") or {}
        if denoise_progress:
            tile_done = int(denoise_progress.get("deep_denoise_done") or 0)
            tile_total = int(denoise_progress.get("deep_denoise_total") or 0)
            provider = denoise_progress.get("deep_denoise_provider") or ""
            source_name = os.path.basename(denoise_progress.get("source_path", "") or "")
            parts = []
            if tile_total > 0:
                pct = int(round(tile_done * 100.0 / max(1, tile_total)))
                parts.append(f"{tile_done}/{tile_total} tiles ({pct}%)")
            if provider:
                parts.append(str(provider))
            if source_name:
                parts.append(source_name)
            lines.extend(["", "Deep Denoise:", "  " + " | ".join(parts)])
        if resource or worker_plan or system_resource:
            lines.extend(["", "Resource Utilization:"])
            if resource:
                lines.append(
                    f"  Export processes: {resource.get('process_count', 0)} | "
                    f"CPU {resource.get('cpu_percent', 0.0):.1f}% | "
                    f"RSS {resource.get('rss_mb', 0.0):.0f} MB"
                )
                for proc in resource.get("processes", [])[:6]:
                    lines.append(
                        f"    PID {proc.get('pid')} {proc.get('role', 'process')}: "
                        f"CPU {proc.get('cpu_percent', 0.0):.1f}% | "
                        f"RSS {proc.get('rss_mb', 0.0):.0f} MB | {proc.get('state', '')}"
                    )
            if system_resource:
                pressure = system_resource.get("memory_pressure")
                free_pct = system_resource.get("memory_free_percent")
                available_gb = system_resource.get("available_memory_gb")
                parts = []
                if pressure:
                    parts.append(f"memory pressure {pressure}")
                if free_pct is not None:
                    parts.append(f"{free_pct}% free")
                if available_gb is not None:
                    parts.append(f"{available_gb:.1f} GB available")
                if parts:
                    lines.append("  System: " + " | ".join(parts))
            if worker_plan:
                plan_parts = [
                    f"selected workers {worker_plan.get('selected_workers')}",
                    f"wave {worker_plan.get('wave_index')} size {worker_plan.get('wave_size')}",
                    f"remaining {worker_plan.get('remaining_tasks')}",
                ]
                if worker_plan.get("deep_denoise_tile_workers") is not None:
                    plan_parts.append(f"denoise tile workers {worker_plan.get('deep_denoise_tile_workers')}")
                if worker_plan.get("available_memory_gb") is not None:
                    plan_parts.append(f"planner memory {worker_plan.get('available_memory_gb')} GB")
                if worker_plan.get("memory_cap") is not None:
                    plan_parts.append(f"memory cap {worker_plan.get('memory_cap')}")
                if worker_plan.get("memory_pressure"):
                    plan_parts.append(f"planner pressure {worker_plan.get('memory_pressure')}")
                lines.append("  Worker plan: " + " | ".join(str(p) for p in plan_parts if p))
        error_details = job.get("error_details") or []
        if error_details:
            by_message = {}
            for entry in error_details:
                message = str(entry.get("error", "") or "(no error message)")
                by_message.setdefault(message, []).append(entry.get("source_path", ""))
            lines.extend(["", f"Errors ({len(error_details)} image(s), {len(by_message)} distinct cause(s)):"])
            for message, paths in by_message.items():
                lines.append(f"  [{len(paths)}x] {message}")
                for path in paths[:5]:
                    lines.append(f"      {os.path.basename(path) if path else '(unknown file)'}")
                if len(paths) > 5:
                    lines.append(f"      ... and {len(paths) - 5} more")
        elif job.get("latest_error"):
            lines.extend(["", "Latest Error:", str(job["latest_error"])])
        self.details.setPlainText("\n".join(lines))

    def _cancel_selected_job(self):
        index = self.jobs_list.currentRow()
        if index < 0 or index >= len(self._jobs) or self._cancel_job_callback is None:
            return
        job = self._jobs[index]
        reply = QMessageBox.question(
            self,
            "Cancel Export",
            f"Cancel export job {job.get('job_id', 'unknown')}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        ok, message = self._cancel_job_callback(job)
        if ok:
            QMessageBox.information(self, "Export Canceled", message)
        else:
            QMessageBox.warning(self, "Cancel Failed", message)
        self.refresh_jobs()

    def _on_auto_refresh_toggled(self, checked: bool):
        if checked:
            self._refresh_timer.start()
        else:
            self._refresh_timer.stop()

    def refresh_jobs(self):
        if self._load_jobs_callback is None:
            return
        output_dir = self.output_dir()
        if not output_dir or not os.path.isdir(output_dir):
            self.set_jobs([])
            return
        self.set_jobs(self._load_jobs_callback(output_dir))


class ReadinessDialog(QDialog):
    """Actionable model/runtime readiness panel with per-component status."""

    STATUS_STYLE = {
        "ready": ("Ready", "#4f9a5f", "#16241a"),
        "fallback": ("Fallback", "#caa14a", "#2a2412"),
        "off": ("Not installed", "#7c8590", "#1b1f26"),
    }

    def __init__(self, items, parent=None, title="System Check", recheck_callback=None, models_dir=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(620, 460)
        self._recheck_callback = recheck_callback
        self._models_dir = str(models_dir) if models_dir else ""

        layout = QVBoxLayout(self)
        heading = QLabel("Portrait AI — runtime &amp; model status", self)
        heading.setStyleSheet("font-size: 11pt; font-weight: 700;")
        layout.addWidget(heading)
        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #9ea4ad;")
        layout.addWidget(self.summary_label)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._rows_host = QWidget(self)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 4, 0, 4)
        self._rows_layout.setSpacing(6)
        scroll.setWidget(self._rows_host)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        open_btn = buttons.addButton("Open Models Folder", QDialogButtonBox.ActionRole)
        open_btn.clicked.connect(self._open_models_folder)
        if recheck_callback is not None:
            recheck_btn = buttons.addButton("Re-check", QDialogButtonBox.ActionRole)
            recheck_btn.clicked.connect(self._recheck)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._populate(items)

    def _clear_rows(self):
        while self._rows_layout.count():
            child = self._rows_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()

    def _make_chip(self, status):
        label_text, fg, bg = self.STATUS_STYLE.get(status, self.STATUS_STYLE["off"])
        chip = QLabel(label_text)
        chip.setAlignment(Qt.AlignCenter)
        chip.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {fg};"
            "border-radius: 6px; padding: 2px 8px; font-weight: 600;"
        )
        return chip

    def _populate(self, items):
        self._clear_rows()
        issues = sum(1 for item in items if item.get("status") != "ready")
        if issues:
            self.summary_label.setText(
                f"{issues} of {len(items)} components are in fallback or not installed. "
                "The app still works — install the optional models below to enable them."
            )
        else:
            self.summary_label.setText("All components are ready.")
        for item in items:
            row = QFrame(self)
            row.setStyleSheet("QFrame { background: #15171d; border: 1px solid #292f3a; border-radius: 8px; }")
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(2)
            top = QHBoxLayout()
            name = QLabel(item.get("name", ""))
            name.setStyleSheet("font-weight: 600; border: 0;")
            top.addWidget(name)
            top.addStretch(1)
            top.addWidget(self._make_chip(item.get("status", "off")))
            row_layout.addLayout(top)
            message = QLabel(item.get("message", ""))
            message.setWordWrap(True)
            message.setStyleSheet("color: #9ea4ad; border: 0;")
            detail = item.get("detail", "")
            if detail:
                message.setToolTip(detail)
            row_layout.addWidget(message)
            self._rows_layout.addWidget(row)
        self._rows_layout.addStretch(1)

    def _open_models_folder(self):
        if not self._models_dir:
            return
        try:
            os.makedirs(self._models_dir, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._models_dir))

    def _recheck(self):
        if self._recheck_callback is None:
            return
        items, _has_issues, models_dir = self._recheck_callback()
        if models_dir:
            self._models_dir = str(models_dir)
        self._populate(items)


class PreferencesDialog(QDialog):
    """App-wide settings that previously had no UI at all -- acceleration mode could only be
    changed by editing code, and the export folder had to be re-picked from scratch every time."""

    ACCELERATION_OPTIONS = [
        ("Auto (prefer GPU, fall back to CPU)", "auto"),
        ("GPU (CUDA) only", "cuda"),
        ("CPU only", "cpu"),
    ]

    def __init__(self, parent=None, preferences: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(480, 220)
        preferences = preferences or {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(10)

        self.acceleration_combo = QComboBox(self)
        for label, _value in self.ACCELERATION_OPTIONS:
            self.acceleration_combo.addItem(label)
        self.acceleration_combo.setToolTip(
            "Controls whether GPU (CUDA) acceleration is used for resize/blur-heavy operations. "
            "Switch to CPU only if you're seeing GPU-related crashes or slowdowns."
        )
        current_accel = preferences.get("acceleration_mode", "auto")
        for idx, (_label, value) in enumerate(self.ACCELERATION_OPTIONS):
            if value == current_accel:
                self.acceleration_combo.setCurrentIndex(idx)
                break
        form.addRow("Acceleration:", self.acceleration_combo)

        export_row = QHBoxLayout()
        self.export_folder_edit = QLineEdit(self)
        self.export_folder_edit.setText(preferences.get("default_export_folder", ""))
        self.export_folder_edit.setPlaceholderText("(ask every time)")
        export_row.addWidget(self.export_folder_edit, 1)
        browse_btn = QPushButton("Browse...", self)
        browse_btn.clicked.connect(self._browse_export_folder)
        export_row.addWidget(browse_btn)
        form.addRow("Default export folder:", export_row)

        self.auto_enhance_checkbox = QCheckBox("Auto-enhance new images on open", self)
        self.auto_enhance_checkbox.setChecked(bool(preferences.get("auto_enhance_on_open", False)))
        self.auto_enhance_checkbox.setToolTip(
            "Applies the first Quick Enhance recipe automatically when opening an image that has "
            "no saved settings yet (an image you've already customized is never touched). "
            "Always one Ctrl+Z away, and off by default."
        )
        form.addRow("", self.auto_enhance_checkbox)

        layout.addLayout(form)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_export_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Default Export Folder", self.export_folder_edit.text())
        if path:
            self.export_folder_edit.setText(path)

    def values(self) -> dict:
        idx = self.acceleration_combo.currentIndex()
        accel_value = self.ACCELERATION_OPTIONS[idx][1] if 0 <= idx < len(self.ACCELERATION_OPTIONS) else "auto"
        return {
            "acceleration_mode": accel_value,
            "default_export_folder": self.export_folder_edit.text().strip(),
            "auto_enhance_on_open": self.auto_enhance_checkbox.isChecked(),
        }


class RecipeDialog(QDialog):
    def __init__(self, recipes: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guided Recipes")
        self.setModal(True)
        self.resize(820, 480)
        self._recipes = list(recipes)
        self._selected_index = 0

        layout = QHBoxLayout(self)
        self.recipe_list = QListWidget(self)
        self.recipe_list.currentRowChanged.connect(self._on_selected)
        layout.addWidget(self.recipe_list, 1)

        right = QVBoxLayout()
        self.title_label = QLabel("Select a recipe")
        right.addWidget(self.title_label)
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        right.addWidget(self.detail_label)
        self.recipe_preview = QTextEdit(self)
        self.recipe_preview.setReadOnly(True)
        right.addWidget(self.recipe_preview, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)
        layout.addLayout(right, 2)

        for recipe in self._recipes:
            self.recipe_list.addItem(recipe.get("name", "Recipe"))
        if self._recipes:
            self.recipe_list.setCurrentRow(0)

    def _on_selected(self, index: int):
        if index < 0 or index >= len(self._recipes):
            self._selected_index = -1
            self.title_label.setText("Select a recipe")
            self.detail_label.setText("")
            self.recipe_preview.clear()
            return
        self._selected_index = index
        recipe = self._recipes[index]
        self.title_label.setText(recipe.get("name", "Recipe"))
        self.detail_label.setText(recipe.get("description", ""))
        highlights = recipe.get("highlights", [])
        if highlights:
            self.recipe_preview.setPlainText("\n".join(f"- {line}" for line in highlights))
        else:
            self.recipe_preview.clear()

    def selected_recipe(self):
        if self._selected_index < 0 or self._selected_index >= len(self._recipes):
            return None
        return self._recipes[self._selected_index]


class MaskDiagnosticsDialog(QDialog):
    """Read-only report from MaskDiagnosticsTask -- one column per detected face, one row per
    mask layer, so an inconsistent/empty mask on a specific face or image can be pinned down
    directly instead of inferred from how Mask View happens to look on screen."""

    # Below this, a "detected" mask is indistinguishable from noise -- flag it as empty rather
    # than show a technically-nonzero but meaningless coverage number.
    EMPTY_THRESHOLD_PCT = 0.05

    def __init__(self, faces: list, report: dict, parent=None, live_face_index=None, live_masks=None, live_state=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Diagnostics")
        self.setModal(True)
        self.resize(820, 560)

        layout = QVBoxLayout(self)

        if live_state:
            diag = self._diagnose_display_block(live_state)
            if diag:
                culprit = QLabel(diag, self)
                culprit.setWordWrap(True)
                culprit.setStyleSheet("color: #e8c07a; background: #2a2418; padding: 6px; border-radius: 4px;")
                layout.addWidget(culprit)

        face_count = len(faces) if faces else 1
        summary = QLabel(self)
        summary.setWordWrap(True)
        if faces:
            box_text = "; ".join(f"Face {i + 1}: {w}x{h} @ ({x},{y})" for i, (x, y, w, h) in enumerate(faces))
            summary.setText(f"{len(faces)} face(s) detected. {box_text}")
        else:
            summary.setText("0 faces detected -- showing the Auto (whole-scene) pass.")
        layout.addWidget(summary)

        confidences = [report.get(idx, {}).get("_person_split_confidence", "") for idx in range(face_count)]
        if any(confidences):
            conf_text = ", ".join(f"Face {i + 1}: {c or 'n/a'}" for i, c in enumerate(confidences))
            conf_label = QLabel(f"Person-split confidence -- {conf_text}", self)
            conf_label.setObjectName("MutedLabel")
            conf_label.setWordWrap(True)
            layout.addWidget(conf_label)

        has_live = live_masks is not None and live_face_index is not None
        if has_live:
            live_mask_face_index = live_state.get("live_mask_face_index") if isinstance(live_state, dict) else None
            mask_tag_note = ""
            if live_mask_face_index is not None and int(live_mask_face_index) != int(live_face_index):
                mask_tag_note = (
                    f" The live mask data is tagged as Face {int(live_mask_face_index) + 1}, "
                    "so the open document is out of sync with the Face Target selector."
                )
            live_label = QLabel(
                f"\"Live\" column -- what's actually held in the open document right now for "
                f"Face {live_face_index + 1} (this is what Mask View paints from). A mismatch "
                f"against the fresh column here means the live document's data is stale, "
                f"separate from whether the segmenter itself can produce good data.{mask_tag_note}",
                self,
            )
            live_label.setObjectName("MutedLabel")
            live_label.setWordWrap(True)
            layout.addWidget(live_label)

        rows = [key for key in MASK_KEYS]
        col_count = face_count + (1 if has_live else 0)
        table = QTableWidget(len(rows), col_count, self)
        headers = [f"Face {i + 1}" for i in range(face_count)] if faces else ["Auto"]
        if has_live:
            headers.append(f"Live (Face {live_face_index + 1})")
        table.setHorizontalHeaderLabels(headers)
        table.setVerticalHeaderLabels(rows)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        def make_item(stats, missing_text):
            """Returns (QTableWidgetItem, is_empty) -- missing_text covers both "no such
            face" (--) and "key absent from this dict entirely" (KEY MISSING), the latter
            being the live-document symptom this comparison column exists to catch."""
            if stats is None:
                item = QTableWidgetItem(missing_text)
                if missing_text == "KEY MISSING":
                    item.setBackground(QColor(120, 40, 40))
                    item.setForeground(QColor(255, 220, 220))
                return item, False
            cov = stats["coverage_10"]
            is_empty = cov < self.EMPTY_THRESHOLD_PCT
            text = f"EMPTY (max={stats['max']:.3f})" if is_empty else f"{cov:.2f}% (max={stats['max']:.2f})"
            item = QTableWidgetItem(text)
            if is_empty:
                item.setBackground(QColor(120, 40, 40))
                item.setForeground(QColor(255, 220, 220))
            return item, is_empty

        empty_count = 0
        for row, key in enumerate(rows):
            for col in range(face_count):
                stats = report.get(col, {}).get(key)
                item, is_empty = make_item(stats, "--")
                if is_empty:
                    empty_count += 1
                table.setItem(row, col, item)
            if has_live:
                live_dict = live_masks or {}
                live_stats = _mask_stats(live_dict[key]) if key in live_dict else None
                missing_text = "KEY MISSING" if (live_masks is not None and key not in live_dict) else "--"
                item, is_empty = make_item(live_stats, missing_text)
                table.setItem(row, face_count, item)
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        if empty_count:
            warning = QLabel(f"⚠ {empty_count} face/layer combination(s) came out completely empty.", self)
            warning.setStyleSheet("color: #e07a7a;")
            layout.addWidget(warning)

        if live_state:
            detail = ", ".join(f"{k}={v}" for k, v in live_state.items())
            state_label = QLabel(f"Live display state: {detail}", self)
            state_label.setObjectName("MutedLabel")
            state_label.setWordWrap(True)
            layout.addWidget(state_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _diagnose_display_block(self, s: dict) -> str:
        """Given the captured live display-gating flags, name the first reason Mask View would
        paint nothing. Order matters -- these are checked top-down in the render path, so the
        first failing gate is the operative one."""
        layer = s.get("active_layer")
        if not s.get("show_mask"):
            return ("⚠ Mask View is OFF in the live document (show_mask=False). The overlay only "
                    "draws while Mask View is toggled on. If the button looks pressed but this "
                    "says off, toggle it off and on once.")
        if not s.get("active_layer_in_MASK_ORDER"):
            return (f"⚠ The active layer is '{layer}', which is not a maskable layer (e.g. Global). "
                    "Mask View only shows per-region layers -- select Face/Skin/Eyes/Lips/Hair/"
                    "Person/Subjects/Background.")
        if not s.get("active_layer_in_preview_masks"):
            return (f"⚠ The active layer '{layer}' has no entry in the live document's masks, even "
                    "though the segmenter can produce one -- the live document is out of sync. "
                    "Reopen the image.")
        if s.get("preview_image_is_None"):
            return ("⚠ No rendered preview image exists yet (preview_image=None), so toggling Mask "
                    "View can't repaint. Wait for the render to finish, or nudge any slider.")
        if s.get("compare_mode") == "before":
            return ("⚠ Compare mode is 'before' -- the canvas is showing the original, unedited "
                    "image and discards every overlay. Set Compare to 'off'.")
        strength = s.get("mask_strength")
        if strength is not None and strength <= 0:
            return (f"⚠ The '{layer}' layer's mask Strength is {strength}% -- at 0 the overlay (and "
                    "the effect) multiply to nothing. Raise Strength or click Reset Settings.")
        return ("All display-gating flags look correct for the overlay to show. If it still "
                "doesn't paint, the masks ARE present (see the Live column) -- capture exactly "
                "what's on screen for the next step.")


class _ResettableSlider(QSlider):
    """A QSlider that resets to its default value on double-click -- otherwise the only way
    to undo one slider is dragging it back by hand or using a layer-wide Reset button."""

    def __init__(self, orientation, default_value=0, parent=None):
        super().__init__(orientation, parent)
        self._default_value = int(default_value)

    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default_value)
        event.accept()


class ImagePreviewLabel(QLabel):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._pixmap = None
        self._scaled_pixmap = None
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 32.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_last = None
        self._compare_mode = "off"
        self._split_callback = None
        self._dragging_split = False
        self._edit_enabled = False
        self._paint_callback = None
        self._paint_start_callback = None
        self._paint_commit_callback = None
        self._dragging_paint = False
        self._hover_rel = None
        self._brush_radius = 24
        self._brush_hardness = 100
        self._source_image_size = None
        self._crop_overlay = None
        self._crop_change_callback = None
        self._crop_commit_callback = None
        self._crop_start_callback = None
        self._crop_drag_handle = None
        self._crop_drag_last = None
        self._crop_aspect_lock = None  # target width:height ratio in true image pixels, or None
        self._wb_pick_enabled = False
        self._wb_pick_callback = None
        self._click_mask_enabled = False
        self._click_mask_callback = None
        self._denoise_point_enabled = False
        self._denoise_point_callback = None
        self._pixmap_image = None
        self._wb_hover_src = None
        self._wb_hover_widget = None
        self._face_highlight = None
        # Click-to-target: a plain click on a detected face (when no modal tool owns the
        # click) selects that face, instead of requiring the Face Target dropdown. Always
        # live (not a toggleable tool) -- lowest-priority gesture, checked only once every
        # other tool/drag has had a chance to claim the click. Targets are (face_index, x, y,
        # w, h) rects in the same normalized display space as _face_highlight.
        self._face_click_targets = []
        self._face_click_callback = None
        self._pan_moved = False
        # True full-resolution pixel zoom: past a magnification threshold, swap in a tile
        # rendered from the real full-res source (through the full edit pipeline) instead of
        # the upscaled, capped-at-1600px editing proxy. _full_to_proxy_ratio is
        # max(full_dim)/max(proxy_dim) (>=1); the threshold is crossed once a real full-res
        # pixel maps to >=1 screen pixel.
        self._hires_enabled = True
        self._hires_active = False
        self._full_to_proxy_ratio = 1.0
        self._hires_pixmap = None
        self._hires_rect_norm = None  # normalized display-space rect the cached tile covers
        self._hires_request_callback = None
        self._hires_pending_rect = None
        self._hires_timer = QTimer(self)
        self._hires_timer.setInterval(150)
        self._hires_timer.setSingleShot(True)
        self._hires_timer.timeout.connect(self._on_hires_timer_fired)

    def set_preview_pixmap(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self._pixmap_image = pixmap.toImage() if pixmap is not None else None
        self._apply_scaled_pixmap()

    def reset_view(self):
        """Reset zoom/pan to fit-the-window."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()
        self.zoomChanged.emit(self._zoom)

    def set_compare_state(self, mode: str, split_callback=None):
        self._compare_mode = str(mode or "off")
        self._split_callback = split_callback

    def set_edit_state(self, enabled: bool, paint_callback=None, paint_start_callback=None, paint_commit_callback=None):
        self._edit_enabled = bool(enabled)
        self._paint_callback = paint_callback
        self._paint_start_callback = paint_start_callback
        self._paint_commit_callback = paint_commit_callback
        if not self._edit_enabled:
            self._hover_rel = None
        self.update()

    def set_brush_preview(self, radius: int, source_image_size=None, hardness: int = 100):
        self._brush_radius = max(1, int(radius))
        self._brush_hardness = max(0, min(100, int(hardness)))
        self._source_image_size = source_image_size
        self.update()

    def set_crop_overlay(self, crop_rect):
        """Show an interactive crop box (normalized rect), or None to hide it."""
        self._crop_overlay = list(crop_rect) if crop_rect is not None else None
        if crop_rect is None:
            self._crop_drag_handle = None
            self._crop_drag_last = None
        self.update()

    def set_crop_aspect_lock(self, aspect):
        """Target width:height ratio (true image pixels) to preserve while dragging crop
        handles, or None for free-form resize."""
        self._crop_aspect_lock = float(aspect) if aspect else None

    def set_face_highlight(self, rect):
        """Outline the currently-targeted face/person (normalized rect), or None to hide it --
        keeps the canvas in sync with the Face Target combo so a forgotten selection doesn't
        silently get edited instead of the one the user thinks is active."""
        self._face_highlight = list(rect) if rect is not None else None
        self.update()

    def set_face_click_targets(self, targets, callback):
        """All detected faces' hit-test rects (normalized, same space as set_face_highlight) and
        the callback to fire with a face_index when one is clicked -- lets the canvas double as a
        face selector instead of requiring the Face Target dropdown."""
        self._face_click_targets = list(targets) if targets else []
        self._face_click_callback = callback

    def _hit_test_face_click(self, widget_x: float, widget_y: float):
        norm = self._widget_to_norm(widget_x, widget_y)
        if norm is None or not self._face_click_targets:
            return None
        nx, ny = norm
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            return None
        for face_index, rx, ry, rw, rh in self._face_click_targets:
            if rx <= nx <= rx + rw and ry <= ny <= ry + rh:
                return face_index
        return None

    def set_crop_callbacks(self, change_callback=None, commit_callback=None, start_callback=None):
        self._crop_change_callback = change_callback
        self._crop_commit_callback = commit_callback
        self._crop_start_callback = start_callback

    def set_wb_pick_state(self, enabled: bool, callback=None):
        self._wb_pick_enabled = bool(enabled)
        self._wb_pick_callback = callback
        if not self._wb_pick_enabled:
            self._wb_hover_src = None
            self._wb_hover_widget = None
        self.setCursor(Qt.CrossCursor if self._wb_pick_enabled else Qt.ArrowCursor)

    def set_click_mask_state(self, enabled: bool, callback=None):
        """AI Select tool: a left click emits a normalized point for the active layer's
        SAM-assisted mask, same shape as set_wb_pick_state."""
        self._click_mask_enabled = bool(enabled)
        self._click_mask_callback = callback
        self.setCursor(Qt.CrossCursor if self._click_mask_enabled else Qt.ArrowCursor)
        self.update()

    def set_denoise_point_state(self, enabled: bool, callback=None):
        """Denoise Point tool: a left click emits a normalized point for a small live-preview
        deep-denoise square centered there, same shape as set_wb_pick_state."""
        self._denoise_point_enabled = bool(enabled)
        self._denoise_point_callback = callback
        self.setCursor(Qt.CrossCursor if self._denoise_point_enabled else Qt.ArrowCursor)
        self.update()

    def set_full_res_ratio(self, ratio: float):
        """max(full_dim)/max(proxy_dim) for the currently loaded image -- 1.0 means the
        proxy already *is* full resolution, so hi-res tiles are never needed."""
        self._full_to_proxy_ratio = max(1.0, float(ratio))

    def set_hires_enabled(self, enabled: bool):
        """Disabled while a mode that already shows something other than the plain edited
        image is active (mask edit, WB/click-mask/denoise-point picking, crop edit, mask
        overlay, expression guides, before/after compare) -- those are out of scope for v1
        and the proxy is correct/expected there."""
        enabled = bool(enabled)
        if self._hires_enabled == enabled:
            return
        self._hires_enabled = enabled
        if not enabled:
            self._hires_active = False
        self.update()

    def set_hires_request_callback(self, callback):
        self._hires_request_callback = callback

    def set_hires_tile(self, pixmap: QPixmap | None, rect_norm=None):
        """Cache the latest rendered hi-res tile and the normalized display-space rect it
        covers, or clear it (pixmap=None) when the document changes and the cache goes
        stale."""
        self._hires_pixmap = pixmap
        self._hires_rect_norm = tuple(rect_norm) if pixmap is not None and rect_norm is not None else None
        self.update()

    def _visible_norm_rect(self):
        """The widget's own content rect, mapped back to normalized display-image space and
        clipped to [0, 1] -- the region of the image actually on screen right now."""
        geom = self._pixmap_rect()
        if geom is None:
            return None
        rect = self.contentsRect()
        top_left = self._widget_to_norm(rect.x(), rect.y())
        bottom_right = self._widget_to_norm(rect.x() + rect.width(), rect.y() + rect.height())
        if top_left is None or bottom_right is None:
            return None
        x0, y0 = top_left
        x1, y1 = bottom_right
        x0, x1 = max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))
        y0, y1 = max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))
        if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
            return None
        return x0, y0, x1, y1

    def _hires_tile_covers(self, visible) -> bool:
        if self._hires_rect_norm is None:
            return False
        hx0, hy0, hx1, hy1 = self._hires_rect_norm
        vx0, vy0, vx1, vy1 = visible
        eps = 1e-6
        return hx0 <= vx0 + eps and hy0 <= vy0 + eps and hx1 >= vx1 - eps and hy1 >= vy1 - eps

    def _request_hires_tile(self, visible_norm):
        if self._hires_request_callback is None:
            return
        x0, y0, x1, y1 = visible_norm
        # Pad by half the visible extent on each side -- covers small pans without a fresh
        # request, and gives slack for the rotation-skew bounding box on the window side.
        pad_x, pad_y = (x1 - x0) * 0.5, (y1 - y0) * 0.5
        self._hires_pending_rect = (
            max(0.0, x0 - pad_x), max(0.0, y0 - pad_y),
            min(1.0, x1 + pad_x), min(1.0, y1 + pad_y),
        )
        self._hires_timer.start()

    def _on_hires_timer_fired(self):
        if self._hires_pending_rect is not None and self._hires_request_callback is not None:
            self._hires_request_callback(self._hires_pending_rect)

    def event(self, e):
        # macOS trackpad pinch arrives as a native zoom gesture.
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:
            self._set_zoom(self._zoom * (1.0 + e.value()), e.position())
            return True
        return super().event(e)

    def wheelEvent(self, event):
        if self._pixmap is None:
            super().wheelEvent(event)
            return
        mods = event.modifiers()
        if mods & (Qt.ControlModifier | Qt.MetaModifier):
            # Ctrl/Cmd + wheel zooms (non-trackpad fallback for pinch).
            self._set_zoom(self._zoom * (1.0 + event.angleDelta().y() / 1200.0), event.position())
            event.accept()
            return
        if self._zoom > 1.0 + 1e-6:
            # Two-finger scroll pans while zoomed in.
            delta = event.pixelDelta()
            if delta.isNull():
                delta = event.angleDelta() / 8
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Double-click resets the view, but only when no edit tool owns the click.
        if (
            not self._edit_enabled
            and self._crop_overlay is None
            and not self._wb_pick_enabled
            and not self._click_mask_enabled
            and not self._denoise_point_enabled
        ):
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def _fit_scale(self) -> float:
        """Scale that fits the source pixmap to the widget (KeepAspectRatio)."""
        if self._pixmap is None:
            return 0.0
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0:
            return 0.0
        rect = self.contentsRect()
        return min(rect.width() / float(iw), rect.height() / float(ih))

    def _pixmap_rect(self):
        """(x0, y0, w, h) of the displayed image inside the widget, including the
        current zoom and pan. Single source of truth for every coordinate map."""
        if self._pixmap is None:
            return None
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if iw <= 0 or ih <= 0:
            return None
        scale = self._fit_scale() * self._zoom
        pw = iw * scale
        ph = ih * scale
        rect = self.contentsRect()
        x0 = rect.x() + (rect.width() - pw) / 2.0 + self._pan_x
        y0 = rect.y() + (rect.height() - ph) / 2.0 + self._pan_y
        return x0, y0, pw, ph

    def _widget_to_norm(self, wx: float, wy: float):
        geom = self._pixmap_rect()
        if geom is None:
            return None
        x0, y0, pw, ph = geom
        if pw <= 0 or ph <= 0:
            return None
        return (float(wx) - x0) / pw, (float(wy) - y0) / ph

    def _clamp_pan(self):
        """Keep the zoomed image covering the viewport (no drift past edges)."""
        if self._pixmap is None or self._zoom <= 1.0 + 1e-6:
            self._pan_x = 0.0
            self._pan_y = 0.0
            return
        rect = self.contentsRect()
        scale = self._fit_scale() * self._zoom
        pw = self._pixmap.width() * scale
        ph = self._pixmap.height() * scale
        ex = max(0.0, (pw - rect.width()) / 2.0)
        ey = max(0.0, (ph - rect.height()) / 2.0)
        self._pan_x = max(-ex, min(ex, self._pan_x))
        self._pan_y = max(-ey, min(ey, self._pan_y))

    def _set_zoom(self, new_zoom: float, center=None):
        new_zoom = max(self._min_zoom, min(self._max_zoom, float(new_zoom)))
        if abs(new_zoom - self._zoom) < 1e-4 or self._pixmap is None:
            return
        # Keep the point under the cursor stationary while zooming.
        norm = self._widget_to_norm(center.x(), center.y()) if center is not None else None
        self._zoom = new_zoom
        if self._zoom <= 1.0 + 1e-6:
            self._pan_x = 0.0
            self._pan_y = 0.0
        elif norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
            rect = self.contentsRect()
            scale = self._fit_scale() * self._zoom
            pw = self._pixmap.width() * scale
            ph = self._pixmap.height() * scale
            base_x0 = rect.x() + (rect.width() - pw) / 2.0
            base_y0 = rect.y() + (rect.height() - ph) / 2.0
            self._pan_x = center.x() - (base_x0 + norm[0] * pw)
            self._pan_y = center.y() - (base_y0 + norm[1] * ph)
        self._clamp_pan()
        self.update()
        self.zoomChanged.emit(self._zoom)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clamp_pan()
        self._apply_scaled_pixmap()

    def leaveEvent(self, event):
        self._hover_rel = None
        self._wb_hover_src = None
        self._wb_hover_widget = None
        self.update()
        super().leaveEvent(event)

    def _paint_crop_overlay(self):
        geom = self._pixmap_rect()
        if geom is None or self._crop_overlay is None:
            return
        x0, y0, pw, ph = geom
        cx, cy, cw, ch = self._crop_overlay
        bx = x0 + cx * pw
        by = y0 + cy * ph
        bw = cw * pw
        bh = ch * ph

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Dim the area outside the crop box.
        dim = QColor(0, 0, 0, 130)
        painter.setPen(Qt.NoPen)
        painter.setBrush(dim)
        painter.drawRect(int(x0), int(y0), int(pw), int(by - y0))
        painter.drawRect(int(x0), int(by + bh), int(pw), int(y0 + ph - (by + bh)))
        painter.drawRect(int(x0), int(by), int(bx - x0), int(bh))
        painter.drawRect(int(bx + bw), int(by), int(x0 + pw - (bx + bw)), int(bh))

        # Box outline and rule-of-thirds guides.
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(int(bx), int(by), int(bw), int(bh))
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
        for i in (1, 2):
            painter.drawLine(int(bx + bw * i / 3), int(by), int(bx + bw * i / 3), int(by + bh))
            painter.drawLine(int(bx), int(by + bh * i / 3), int(bx + bw), int(by + bh * i / 3))

        # Corner/edge handles.
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(212, 168, 83))
        hs = 7
        for nx, ny in (
            (cx, cy), (cx + cw / 2, cy), (cx + cw, cy),
            (cx + cw, cy + ch / 2), (cx + cw, cy + ch),
            (cx + cw / 2, cy + ch), (cx, cy + ch), (cx, cy + ch / 2),
        ):
            hx = x0 + nx * pw
            hy = y0 + ny * ph
            painter.drawRect(int(hx - hs / 2), int(hy - hs / 2), hs, hs)
        painter.end()

    def _paint_face_highlight(self):
        geom = self._pixmap_rect()
        if geom is None or self._face_highlight is None:
            return
        x0, y0, pw, ph = geom
        fx, fy, fw, fh = self._face_highlight
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(212, 168, 83), 2))
        painter.drawRect(int(x0 + fx * pw), int(y0 + fy * ph), int(fw * pw), int(fh * ph))
        painter.end()

    def _paint_wb_loupe(self):
        if self._pixmap_image is None or self._wb_hover_src is None or self._wb_hover_widget is None:
            return
        img = self._pixmap_image
        sx, sy = self._wb_hover_src
        cells, half, zoom = 9, 4, 12
        size = cells * zoom
        wx, wy = self._wb_hover_widget
        rect = self.rect()
        lx = wx + 18 if wx + 18 + size <= rect.right() else wx - 18 - size
        ly = wy + 18 if wy + 18 + size <= rect.bottom() else wy - 18 - size
        lx = max(rect.left(), min(lx, rect.right() - size))
        ly = max(rect.top(), min(ly, rect.bottom() - size))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for j in range(cells):
            for i in range(cells):
                px, py = sx - half + i, sy - half + j
                if 0 <= px < img.width() and 0 <= py < img.height():
                    color = img.pixelColor(px, py)
                else:
                    color = QColor(20, 20, 20)
                painter.fillRect(lx + i * zoom, ly + j * zoom, zoom, zoom, color)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawRect(lx + half * zoom, ly + half * zoom, zoom, zoom)
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(lx, ly, size - 1, size - 1)
        if 0 <= sx < img.width() and 0 <= sy < img.height():
            c = img.pixelColor(sx, sy)
            tb_y = ly + size + 2 if ly + size + 18 <= rect.bottom() else ly - 18
            painter.fillRect(lx, tb_y, size, 16, QColor(0, 0, 0, 210))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(lx + 3, tb_y + 12, f"{c.red()},{c.green()},{c.blue()}")
        painter.end()

    def _paint_hires_tile(self, geom):
        """True full-resolution pixel zoom: once a real full-res pixel maps to >=1 screen
        pixel, swap the corresponding region of the upscaled proxy for a tile rendered from
        the actual full-resolution source. Hysteresis (enter at 1.0x, exit below 0.75x) avoids
        flicker right at the threshold; below the exit point the plain proxy (already drawn by
        the caller) is left alone."""
        if not self._hires_enabled or self._full_to_proxy_ratio <= 1.0 + 1e-6:
            self._hires_active = False
            return
        effective_ratio = (self._fit_scale() * self._zoom) / self._full_to_proxy_ratio
        threshold = 0.75 if self._hires_active else 1.0
        if effective_ratio < threshold:
            self._hires_active = False
            return
        self._hires_active = True
        visible = self._visible_norm_rect()
        if visible is None:
            return
        if self._hires_pixmap is not None and self._hires_tile_covers(visible):
            x0, y0, pw, ph = geom
            hx0, hy0, hx1, hy1 = self._hires_rect_norm
            dest = QRectF(x0 + hx0 * pw, y0 + hy0 * ph, (hx1 - hx0) * pw, (hy1 - hy0) * ph)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.setClipRect(self.contentsRect())
            painter.drawPixmap(dest, self._hires_pixmap, QRectF(self._hires_pixmap.rect()))
            painter.end()
        else:
            self._request_hires_tile(visible)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is not None:
            geom = self._pixmap_rect()
            if geom is not None:
                x0, y0, pw, ph = geom
                painter = QPainter(self)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                painter.setClipRect(self.contentsRect())
                painter.drawPixmap(QRectF(x0, y0, pw, ph), self._pixmap, QRectF(self._pixmap.rect()))
                painter.end()
                self._paint_hires_tile(geom)
        if self._crop_overlay is not None:
            self._paint_crop_overlay()
        if self._face_highlight is not None:
            self._paint_face_highlight()
        if self._wb_pick_enabled:
            self._paint_wb_loupe()
        if not self._edit_enabled or self._hover_rel is None or self._pixmap is None:
            return
        if not self._source_image_size or self._source_image_size[0] <= 0 or self._source_image_size[1] <= 0:
            return

        geom = self._pixmap_rect()
        if geom is None:
            return
        x0, y0, pw, ph = geom
        cx = x0 + self._hover_rel[0] * pw
        cy = y0 + self._hover_rel[1] * ph
        scale_x = pw / float(self._source_image_size[0])
        scale_y = ph / float(self._source_image_size[1])
        radius = max(2.0, self._brush_radius * min(scale_x, scale_y))
        hardness = max(0.0, min(1.0, self._brush_hardness / 100.0))
        inner_radius = max(1.0, radius * hardness)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(int(round(cx - radius)), int(round(cy - radius)), int(round(radius * 2)), int(round(radius * 2)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        inner_d = max(2.0, radius * 2 - 2)
        painter.drawEllipse(
            int(round(cx - radius + 1)),
            int(round(cy - radius + 1)),
            int(round(inner_d)),
            int(round(inner_d)),
        )
        if inner_radius < radius - 1.0:
            painter.setPen(QPen(QColor(91, 155, 213), 1, Qt.DashLine))
            painter.drawEllipse(
                int(round(cx - inner_radius)),
                int(round(cy - inner_radius)),
                int(round(max(2.0, inner_radius * 2))),
                int(round(max(2.0, inner_radius * 2))),
            )
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._wb_pick_enabled and self._wb_pick_callback is not None:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
                self._wb_pick_callback(norm[0], norm[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and self._click_mask_enabled and self._click_mask_callback is not None:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
                self._click_mask_callback(norm[0], norm[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and self._denoise_point_enabled and self._denoise_point_callback is not None:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
                self._denoise_point_callback(norm[0], norm[1])
                event.accept()
                return
        if event.button() == Qt.LeftButton and self._crop_overlay is not None:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None:
                tol = self._crop_handle_tolerance()
                handle = framing_ops.hit_test_handle(self._crop_overlay, norm[0], norm[1], tol)
                if handle is not None:
                    self._crop_drag_handle = handle
                    self._crop_drag_last = norm
                    if self._crop_start_callback is not None:
                        self._crop_start_callback()
                    event.accept()
                    return
        if event.button() == Qt.LeftButton and self._edit_enabled and self._paint_callback is not None:
            self._update_hover_rel(event.position().x(), event.position().y())
            if self._paint_start_callback is not None:
                self._paint_start_callback()
            self._dragging_paint = True
            self._emit_paint_point(event.position().x(), event.position().y())
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._compare_mode == "split":
            self._dragging_split = True
            self._emit_split_position(event.position().x())
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._zoom > 1.0 + 1e-6:
            # No edit tool active and zoomed in: left-drag pans the canvas. A plain click
            # (zero/negligible movement before release) is distinguished from a pan-drag in
            # mouseReleaseEvent, so face click-to-target still works while zoomed in.
            self._panning = True
            self._pan_last = event.position()
            self._pan_moved = False
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._face_click_callback is not None:
            face_index = self._hit_test_face_click(event.position().x(), event.position().y())
            if face_index is not None:
                self._face_click_callback(face_index)
                event.accept()
                return
        super().mousePressEvent(event)

    def _crop_handle_tolerance(self) -> float:
        geom = self._pixmap_rect()
        if geom is None:
            return 0.03
        _x0, _y0, pw, _ph = geom
        return max(0.02, 10.0 / max(1.0, pw))

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_last is not None:
            pos = event.position()
            self._pan_x += pos.x() - self._pan_last.x()
            self._pan_y += pos.y() - self._pan_last.y()
            self._pan_last = pos
            self._pan_moved = True
            self._clamp_pan()
            self.update()
            event.accept()
            return
        if self._wb_pick_enabled:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            pix = self._pixmap
            if norm is not None and pix is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
                self._wb_hover_src = (
                    int(round(norm[0] * (pix.width() - 1))),
                    int(round(norm[1] * (pix.height() - 1))),
                )
                self._wb_hover_widget = (event.position().x(), event.position().y())
            else:
                self._wb_hover_src = None
                self._wb_hover_widget = None
            self.update()
            event.accept()
            return
        if self._crop_drag_handle is not None and self._crop_overlay is not None:
            norm = self._widget_to_norm(event.position().x(), event.position().y())
            if norm is not None and self._crop_drag_last is not None:
                dx = norm[0] - self._crop_drag_last[0]
                dy = norm[1] - self._crop_drag_last[1]
                # In crop-edit mode the displayed pixmap is the full straightened (uncropped)
                # frame, so its own width:height ratio is exactly the source image's --
                # needed to convert a true-pixel aspect lock into the matching fraction-space
                # ratio for the normalized crop box.
                canvas_aspect = 1.0
                if self._pixmap is not None and self._pixmap.height() > 0:
                    canvas_aspect = self._pixmap.width() / self._pixmap.height()
                self._crop_overlay = framing_ops.resize_crop(
                    self._crop_overlay, self._crop_drag_handle, dx, dy,
                    aspect=self._crop_aspect_lock, canvas_aspect=canvas_aspect,
                )
                self._crop_drag_last = norm
                if self._crop_change_callback is not None:
                    self._crop_change_callback(list(self._crop_overlay))
                self.update()
            event.accept()
            return
        self._update_hover_rel(event.position().x(), event.position().y())
        if self._dragging_paint and self._edit_enabled and self._paint_callback is not None:
            self._emit_paint_point(event.position().x(), event.position().y())
            event.accept()
            return
        if self._dragging_split and self._compare_mode == "split":
            self._emit_split_position(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._crop_drag_handle is not None:
                self._crop_drag_handle = None
                self._crop_drag_last = None
                if self._crop_commit_callback is not None:
                    self._crop_commit_callback()
            if self._panning:
                was_click = not self._pan_moved
                self._panning = False
                self._pan_last = None
                self.setCursor(Qt.CrossCursor if self._wb_pick_enabled else Qt.ArrowCursor)
                # Zoomed in, button released without ever dragging -- this was a tap, not a
                # pan, so it can still target a face (mousePressEvent only checks face targets
                # at zoom<=1, since above that every press provisionally starts a pan).
                if was_click and self._face_click_callback is not None:
                    face_index = self._hit_test_face_click(event.position().x(), event.position().y())
                    if face_index is not None:
                        self._face_click_callback(face_index)
            if self._dragging_paint and self._paint_commit_callback is not None:
                self._paint_commit_callback()
            self._dragging_split = False
            self._dragging_paint = False
        super().mouseReleaseEvent(event)

    def _apply_scaled_pixmap(self):
        # The image is painted manually in paintEvent (so it can be zoomed/panned);
        # QLabel only shows the placeholder text when there is no image.
        if self._pixmap is None:
            self._scaled_pixmap = None
            self.setText("")
        else:
            self.setText("")
        self.update()

    def _update_hover_rel(self, widget_x: float, widget_y: float):
        norm = self._widget_to_norm(widget_x, widget_y)
        if norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
            self._hover_rel = norm
        else:
            self._hover_rel = None
        self.update()

    def _emit_split_position(self, widget_x: float):
        if self._split_callback is None:
            return
        geom = self._pixmap_rect()
        if geom is None:
            return
        x0, _y0, pw, _ph = geom
        rel_x = (float(widget_x) - x0) / pw
        self._split_callback(max(0.0, min(1.0, rel_x)))

    def _emit_paint_point(self, widget_x: float, widget_y: float):
        if self._paint_callback is None:
            return
        norm = self._widget_to_norm(widget_x, widget_y)
        if norm is not None and 0.0 <= norm[0] <= 1.0 and 0.0 <= norm[1] <= 1.0:
            self._paint_callback(norm[0], norm[1])


class HistogramWidget(QWidget):
    """Compact RGB + luminance histogram with clipping indicators."""

    _CLIP_THRESHOLD = 0.005  # >0.5% of pixels clipped lights the warning marker

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Histogram of the edited preview (R/G/B + luma)")

    def set_histogram(self, data: dict | None):
        self._data = data
        self.update()

    def clear(self):
        self.set_histogram(None)

    def paintEvent(self, event):  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(self.rect(), QColor("#0c0c0e"))

        if not self._data:
            painter.setPen(QColor("#5a5650"))
            painter.drawText(rect, Qt.AlignCenter, "Histogram")
            painter.end()
            return

        levels = int(self._data.get("levels", 256))
        width = rect.width()
        height = rect.height()

        channels = (
            ("luma", QColor(200, 200, 200, 150)),
            ("red", QColor(212, 88, 88, 150)),
            ("green", QColor(120, 200, 120, 150)),
            ("blue", QColor(110, 150, 220, 160)),
        )

        # Shared scale (skip pure-black/white spikes so they don't flatten the curve).
        peak = 1.0
        for name, _ in channels:
            counts = np.asarray(self._data.get(name), dtype=np.float64)
            if counts.size >= levels and levels > 2:
                interior = counts[1 : levels - 1]
                if interior.size:
                    peak = max(peak, float(interior.max()))
        log_peak = np.log1p(peak)

        for name, color in channels:
            counts = np.asarray(self._data.get(name), dtype=np.float64)
            if counts.size < levels:
                continue
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            scaled = np.log1p(counts) / log_peak if log_peak > 0 else counts * 0.0
            for x in range(width):
                lo = int(x * levels / width)
                hi = max(lo + 1, int((x + 1) * levels / width))
                value = float(scaled[lo:hi].max()) if hi <= levels else 0.0
                bar_h = int(min(1.0, value) * height)
                if bar_h > 0:
                    painter.drawRect(rect.left() + x, rect.bottom() - bar_h, 1, bar_h)

        # Clipping markers: bottom-left for crushed shadows, bottom-right for blown highlights.
        # Distinguished by shape (circle vs. square), not just color, so the difference isn't
        # lost on a colorblind user -- and the tooltip spells it out in text either way.
        shadow = self._data.get("shadow_clip", {})
        highlight = self._data.get("highlight_clip", {})
        shadow_clipped = any(v > self._CLIP_THRESHOLD for v in shadow.values())
        highlight_clipped = any(v > self._CLIP_THRESHOLD for v in highlight.values())
        marker = 6
        if shadow_clipped:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(91, 155, 213))
            painter.drawEllipse(rect.left(), rect.top(), marker, marker)
        if highlight_clipped:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(212, 168, 83))
            painter.drawRect(rect.right() - marker + 1, rect.top(), marker, marker)
        tooltip = "Histogram of the edited preview (R/G/B + luma)"
        if shadow_clipped or highlight_clipped:
            parts = []
            if shadow_clipped:
                parts.append("shadows clipped (circle, blue)")
            if highlight_clipped:
                parts.append("highlights clipped (square, gold)")
            tooltip += " -- " + " and ".join(parts)
        self.setToolTip(tooltip)
        painter.end()


class ToneCurveWidget(QWidget):
    """Interactive tone curve: drag points, click to add, double-click to remove."""

    _TOL = 0.045

    def __init__(self, parent=None):
        super().__init__(parent)
        self._points = tc_ops.default_curve()
        self._hist = None
        self._drag_index = None
        self._change_cb = None
        self._commit_cb = None
        self._start_cb = None
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Drag points to shape tones · click to add · double-click to remove")

    def set_points(self, points):
        self._points = [tuple(p) for p in tc_ops.normalize_curve(points)]
        self.update()

    def set_histogram(self, hist):
        self._hist = hist
        self.update()

    def set_callbacks(self, change=None, commit=None, start=None):
        self._change_cb = change
        self._commit_cb = commit
        self._start_cb = start

    def _plot_rect(self):
        return self.contentsRect().adjusted(6, 6, -6, -6)

    def _to_widget(self, x, y):
        r = self._plot_rect()
        return r.left() + x * r.width(), r.bottom() - y * r.height()

    def _to_norm(self, wx, wy):
        r = self._plot_rect()
        if r.width() <= 0 or r.height() <= 0:
            return 0.0, 0.0
        x = (float(wx) - r.left()) / r.width()
        y = (r.bottom() - float(wy)) / r.height()
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#141318"))
        r = self._plot_rect()

        # Faint luminance histogram backdrop.
        if self._hist is not None:
            luma = np.asarray(self._hist.get("luma"), dtype=np.float64)
            if luma.size:
                interior = luma[1:-1] if luma.size > 2 else luma
                peak = np.log1p(float(interior.max())) if interior.size else 1.0
                if peak > 0:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QColor(90, 90, 100, 90))
                    levels = luma.size
                    for x in range(r.width()):
                        lo = int(x * levels / r.width())
                        hi = max(lo + 1, int((x + 1) * levels / r.width()))
                        val = float(np.log1p(luma[lo:hi].max())) / peak
                        bar = int(min(1.0, val) * r.height())
                        if bar > 0:
                            painter.drawRect(r.left() + x, r.bottom() - bar, 1, bar)

        # Grid (thirds) and identity diagonal.
        painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
        for i in (1, 2):
            painter.drawLine(int(r.left() + r.width() * i / 3), r.top(), int(r.left() + r.width() * i / 3), r.bottom())
            painter.drawLine(r.left(), int(r.top() + r.height() * i / 3), r.right(), int(r.top() + r.height() * i / 3))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
        painter.drawLine(r.left(), r.bottom(), r.right(), r.top())

        # The curve, sampled from the LUT.
        lut = tc_ops.curve_to_lut(self._points, size=max(2, r.width()))
        painter.setPen(QPen(QColor(212, 168, 83), 2))
        prev = None
        for x in range(r.width()):
            wx = r.left() + x
            wy = r.bottom() - float(lut[x]) * r.height()
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(wx), int(wy))
            prev = (wx, wy)

        # Control points.
        for px, py in self._points:
            wx, wy = self._to_widget(px, py)
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(QColor(40, 40, 40), 1))
            painter.drawEllipse(int(wx - 4), int(wy - 4), 8, 8)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        x, y = self._to_norm(event.position().x(), event.position().y())
        idx = tc_ops.nearest_point(self._points, x, y, self._TOL)
        if self._start_cb is not None:
            self._start_cb()
        if idx is None:
            self._points, idx = tc_ops.add_point(self._points, x, y)
            self._emit_change()
        self._drag_index = idx
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_index is None:
            return super().mouseMoveEvent(event)
        x, y = self._to_norm(event.position().x(), event.position().y())
        self._points = tc_ops.move_point(self._points, self._drag_index, x, y)
        self._emit_change()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_index is not None:
            self._drag_index = None
            if self._commit_cb is not None:
                self._commit_cb()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        x, y = self._to_norm(event.position().x(), event.position().y())
        idx = tc_ops.nearest_point(self._points, x, y, self._TOL)
        if idx is not None:
            new_points = tc_ops.remove_point(self._points, idx)
            if new_points != self._points:
                if self._start_cb is not None:
                    self._start_cb()
                self._points = new_points
                self._emit_change()
                if self._commit_cb is not None:
                    self._commit_cb()
        self.update()
        event.accept()

    def _emit_change(self):
        if self._change_cb is not None:
            self._change_cb([list(p) for p in self._points])


def _copy_masks_data(masks):
    """Pure-data deep copy of a mask dict -- no Qt/self dependency, safe to call from a
    worker thread. The window's _copy_masks delegates here so existing call sites are
    unaffected; SegmentationTask (which holds no `self` reference) calls this directly."""
    if masks is None:
        return None
    return {key: value.copy() for key, value in masks.items()}


def _masks_usable(masks) -> bool:
    """A mask dict is usable only if it actually carries the per-region layer arrays. A face
    profile deserialized from a saved collection/project carries the per-face slider params but
    an EMPTY mask dict (masks are too large to persist, and are meant to be regenerated on
    load) -- restoring that empty dict as-is leaves every layer's Mask View painting nothing
    (KEY MISSING). Such a profile must trigger a fresh segmentation, not be treated as fully
    restored."""
    if not masks:
        return False
    return any(key in masks for key in MASK_ORDER)


def _decode_image_file(path: str, color_settings=None, on_preview=None):
    """Pure decode: RAW/image file -> (float array, metadata dict), no window-state side
    effects -- safe to call from a worker thread. _read_image_file (instance method) wraps
    this for callers that still expect the self._source_metadata side effect.

    RAW decode is delegated to the shared core.raw_decode.decode_raw so Qt, batch, and Tk
    honor the same RAW white-balance/colorspace/LUT/auto-brightness settings and capture the
    same camera metadata. color_settings may be None (camera-WB defaults) for callers like
    donor-frame decoding that don't carry the document's settings.

    on_preview, if given, is forwarded to decode_raw for a fast half-resolution preview pass
    on RAW sources (see decode_raw's docstring); ignored for non-RAW sources, which decode via
    PIL and are already fast enough that a staged preview isn't worth the complexity."""
    if is_raw_path(path):
        return decode_raw(path, color_settings, on_preview=on_preview)

    pil = Image.open(path)
    metadata = {}
    exif_bytes = pil.info.get("exif")
    if exif_bytes:
        metadata["exif"] = exif_bytes
    icc_profile = pil.info.get("icc_profile")
    if icc_profile:
        metadata["icc_profile"] = icc_profile
    dpi = pil.info.get("dpi")
    if dpi:
        metadata["dpi"] = dpi
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.asarray(pil, dtype=np.float32) / 255.0, metadata


class ImageLoadSignals(QObject):
    finished = Signal(int, str, object, dict)  # job_id, path, array, metadata
    # Fired once, before `finished`, only for RAW sources: a fast half-resolution decode so
    # the canvas can show a pixel-accurate preview well before the full decode completes.
    preview_ready = Signal(int, str, object, dict)  # job_id, path, array, metadata
    failed = Signal(int, str, str)  # job_id, path, message


class ImageLoadTask(QRunnable):
    """Decodes a RAW/image file off the GUI thread -- a large RAW file's demosaic can take
    several seconds; previously this ran synchronously in _load_image_path and froze the
    window for that whole time on every image open/switch."""

    def __init__(self, job_id: int, path: str, color_settings=None):
        super().__init__()
        self.job_id = job_id
        self.path = path
        # Snapshot the document's color settings so a RAW file is decoded with the WB/
        # colorspace/LUT/auto-brightness in effect at enqueue time, even if the user changes
        # them before the worker thread runs.
        self.color_settings = color_settings
        self.signals = ImageLoadSignals()

    def run(self):
        try:
            # emit() from this worker thread is safe -- Qt auto-queues delivery to the
            # receiver's (GUI) thread, the same pattern the finished/failed signals use below.
            on_preview = lambda arr, meta: self.signals.preview_ready.emit(self.job_id, self.path, arr, meta)
            full, metadata = _decode_image_file(self.path, self.color_settings, on_preview=on_preview)
            self.signals.finished.emit(self.job_id, self.path, full, metadata)
        except Exception as ex:
            self.signals.failed.emit(self.job_id, self.path, str(ex))


class FixEyesSignals(QObject):
    finished = Signal(int, object, int, int)  # job_id, result_array, fixed_count, other_paths_count
    no_result = Signal(int, str)  # job_id, status message (no blink / no match -- not an error)
    failed = Signal(int, str)  # job_id, message


class FixEyesTask(QRunnable):
    """Analyzes the active image's faces for blinks and swaps in open-eyes pixels from other
    burst frames, off the GUI thread -- decoding + analyzing every donor frame synchronously
    previously froze the window (a WaitCursor was the only feedback) for however long that
    took. Holds no `self` window reference; the segmenter is read-only here (face detection
    inference), matching SegmentationTask's existing pattern."""

    def __init__(self, job_id: int, full_array: np.ndarray, other_paths: list, segmenter):
        super().__init__()
        self.job_id = job_id
        self.full_array = full_array
        self.other_paths = other_paths
        self.segmenter = segmenter
        self.signals = FixEyesSignals()

    def run(self):
        try:
            target_faces = culling_ops.analyze_faces_detailed(self.segmenter, self.full_array)
            blinking = [f for f in target_faces if f["open"] is False]
            if not blinking:
                self.signals.no_result.emit(self.job_id, "Fix Eyes: no blink detected in this image")
                return

            result = self.full_array.copy()
            fixed_count = 0
            donor_cache = {}  # path -> per-face details, decoded/analyzed at most once
            for face in blinking:
                for donor_path in self.other_paths:
                    if donor_path not in donor_cache:
                        try:
                            donor_img, _meta = _decode_image_file(donor_path)
                            donor_cache[donor_path] = (donor_img, culling_ops.analyze_faces_detailed(self.segmenter, donor_img))
                        except Exception:
                            donor_cache[donor_path] = (None, [])
                    donor_img, donor_faces = donor_cache[donor_path]
                    if donor_img is None:
                        continue
                    match = face_swap_ops.match_face_by_position(face["box"], donor_faces)
                    if match is None or match["open"] is not True or not match["guides"]:
                        continue
                    swapped_img, _sides = face_swap_ops.swap_eyes(
                        result, face["guides"], face["box"], donor_img, match["guides"]
                    )
                    if swapped_img is not None:
                        result = swapped_img
                        fixed_count += 1
                        break  # this face is fixed -- move on to any other blinking face

            if fixed_count == 0:
                self.signals.no_result.emit(
                    self.job_id,
                    "Fix Eyes: found a blink but no open-eyes match for that face in the other burst frames",
                )
                return
            self.signals.finished.emit(self.job_id, result, fixed_count, len(self.other_paths))
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))


class MaskClickSignals(QObject):
    finished = Signal(int, object)  # job_id, mask (np.ndarray) or None
    failed = Signal(int, str)


class MaskClickTask(QRunnable):
    """Computes one SAM click-prompted mask off the GUI thread -- the first click on a fresh
    image pays the encoder's embedding cost (up to a couple seconds on CPU), which would
    otherwise freeze the window with no feedback. Holds no `self` window reference; the
    segmenter's SAM instance is read-only here, matching every other inference task's pattern."""

    def __init__(self, job_id: int, preview_array: np.ndarray, point_xy: tuple[float, float], segmenter):
        super().__init__()
        self.job_id = job_id
        self.preview_array = preview_array
        self.point_xy = point_xy
        self.segmenter = segmenter
        self.signals = MaskClickSignals()

    def run(self):
        try:
            instance = getattr(self.segmenter, "_instance", None)
            if instance is None or not getattr(instance, "available", False):
                self.signals.finished.emit(self.job_id, None)
                return
            mask = instance.click_mask(self.preview_array, self.point_xy)
            self.signals.finished.emit(self.job_id, mask)
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))


class WBAutoAISignals(QObject):
    finished = Signal(int, list)  # job_id, [(path, temp_k, tint, fallback_reason, error_or_None), ...]
    failed = Signal(int, str)


class WBAutoAITask(QRunnable):
    """Decodes + estimates AI white balance for each selected image off the GUI thread. Takes
    bound methods for the decode-proxy/estimate steps rather than reimplementing them --
    neither touches mutable window state (build_preview_proxy reads only its argument;
    estimate_wb only reads the read-only-after-startup WB estimator model), so calling them
    from a worker thread is safe; only plain data crosses back to the GUI thread, where the
    original code's widget updates and disk writes still happen."""

    def __init__(self, job_id: int, items: list, build_preview_proxy, estimate_wb):
        super().__init__()
        self.job_id = job_id
        self.items = items  # list of (path, preview_array_or_None) -- None means "decode this path"
        self.build_preview_proxy = build_preview_proxy
        self.estimate_wb = estimate_wb
        self.signals = WBAutoAISignals()

    def run(self):
        try:
            results = []
            for path, preview in self.items:
                try:
                    if preview is None:
                        full, _meta = _decode_image_file(path)
                        preview, _scale = self.build_preview_proxy(full)
                    temp_k, tint, fallback_reason = self.estimate_wb(preview)
                    results.append((path, temp_k, tint, fallback_reason, None))
                except Exception as ex:
                    results.append((path, None, None, None, str(ex)))
            self.signals.finished.emit(self.job_id, results)
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))


def _copy_guides_data(guides):
    """Pure-data deep copy of a guides structure (dict/list of dicts of points/arrays)."""
    if guides is None:
        return None
    if isinstance(guides, list):
        return [_copy_guides_data(item) for item in guides]
    if isinstance(guides, dict):
        copied = {}
        for key, value in guides.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                copied[key] = [float(value[0]), float(value[1])]
            else:
                copied[key] = value
        return copied
    return guides


def _scale_masks_data(masks, shape_hw):
    """Resize every mask in `masks` to `shape_hw` (h, w). Pure PIL/numpy, no Qt dependency."""
    if masks is None:
        return None
    h, w = shape_hw
    scaled = {}
    for key, mask in masks.items():
        pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize((w, h), Image.BILINEAR)
        scaled[key] = np.asarray(pil, dtype=np.float32) / 255.0
    return scaled


class PreviewRenderSignals(QObject):
    finished = Signal(int, object, float)
    failed = Signal(int, str)


class PreviewRenderTask(QRunnable):
    def __init__(
        self,
        job_id: int,
        image: np.ndarray,
        params: dict,
        masks,
        geometry,
        layer_order,
        layer_options,
        color_settings,
        runtime_settings,
        stage_cache=None,
        inputs_token=None,
        want_sharpen_mask: bool = False,
    ):
        super().__init__()
        self.job_id = int(job_id)
        self.image = image
        self.params = params
        self.masks = masks
        self.geometry = geometry
        self.layer_order = layer_order
        self.layer_options = layer_options
        self.color_settings = color_settings
        self.runtime_settings = runtime_settings
        self.stage_cache = stage_cache
        self.inputs_token = inputs_token
        self.want_sharpen_mask = want_sharpen_mask
        self.debug_sink: dict = {}
        self.signals = PreviewRenderSignals()

    def run(self):
        t0 = time.perf_counter()
        try:
            result = process_all_layers(
                self.image,
                self.params,
                self.masks,
                geometry=self.geometry,
                layer_order=self.layer_order,
                layer_options=self.layer_options,
                color_settings=self.color_settings,
                runtime_settings=self.runtime_settings,
                stage_cache=self.stage_cache,
                inputs_token=self.inputs_token,
                debug_sink=self.debug_sink if self.want_sharpen_mask else None,
            )
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.signals.finished.emit(self.job_id, result, float(elapsed_ms))


class HiresTileSignals(QObject):
    finished = Signal(int, object, object)  # job_id, PIL.Image tile (or None), rect_norm
    failed = Signal(int, str)


class HiresTileTask(QRunnable):
    """Renders one screen-resolution tile of the true full-resolution source through the
    full edit pipeline, for the zoomed-in hi-res viewport (`render past the proxy's own
    detail -- a real single source pixel per screen pixel`). `crop`/`masks`/`geometry`
    are already cropped to (and offset into) the sub-region at `crop_origin` within the
    `full_shape` source image; `target_rect_px` is that same viewport rect expressed in
    the rotated-but-uncropped display canvas's pixel space (see
    `core.framing.render_display_tile`)."""

    def __init__(
        self,
        job_id: int,
        crop: np.ndarray,
        masks,
        geometry,
        params: dict,
        layer_order,
        layer_options,
        color_settings,
        runtime_settings,
        crop_origin: tuple[int, int],
        full_shape: tuple[int, int],
        framing,
        target_rect_px: tuple[int, int, int, int],
        rect_norm: tuple[float, float, float, float],
    ):
        super().__init__()
        self.job_id = int(job_id)
        self.crop = crop
        self.masks = masks
        self.geometry = geometry
        self.params = params
        self.layer_order = layer_order
        self.layer_options = layer_options
        self.color_settings = color_settings
        self.runtime_settings = runtime_settings
        self.crop_origin = crop_origin
        self.full_shape = full_shape
        self.framing = framing
        self.target_rect_px = target_rect_px
        self.rect_norm = rect_norm
        self.signals = HiresTileSignals()

    def run(self):
        try:
            result = process_all_layers(
                self.crop,
                self.params,
                self.masks,
                geometry=self.geometry,
                layer_order=self.layer_order,
                layer_options=self.layer_options,
                color_settings=self.color_settings,
                runtime_settings=self.runtime_settings,
                crop_origin=self.crop_origin,
                full_shape=self.full_shape,
            )
            full_h, full_w = self.full_shape
            tile = framing_ops.render_display_tile(
                result, self.framing, full_w, full_h, self.crop_origin, self.target_rect_px
            )
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))
            return
        self.signals.finished.emit(self.job_id, tile, self.rect_norm)


class SegmentationSignals(QObject):
    finished = Signal(int, object)
    failed = Signal(int, str)


class SegmentationTask(QRunnable):
    """Runs face detection + mask segmentation for the *active* image off the GUI thread,
    on the high-priority pool (default OS priority) -- separate from the low-priority
    background preload pool, so opening/switching the active image isn't stuck waiting
    behind queued-but-not-open filmstrip selections. Holds no `self` window reference (only
    plain data in, plain data out via the finished signal), matching PreviewRenderTask's
    pattern -- the window applies the result to its own state on the GUI thread."""

    def __init__(
        self,
        job_id: int,
        file_path: str,
        preview_array: np.ndarray,
        full_shape: tuple[int, int],
        preview_scale: float,
        active_face_index: int,
        segmenter,
        force_recompute: bool = False,
    ):
        super().__init__()
        self.job_id = int(job_id)
        self.file_path = file_path
        self.preview_array = preview_array
        self.full_shape = full_shape
        self.preview_scale = float(preview_scale)
        self.active_face_index = max(0, int(active_face_index))
        self.segmenter = segmenter
        # When True, skip the disk-cache lookup entirely and recompute from the models, then
        # overwrite the cached record with the fresh result -- used by "Recalculate Mask" to
        # force a genuinely new computation rather than reverting to a possibly-stale cache hit.
        self.force_recompute = bool(force_recompute)
        self.signals = SegmentationSignals()

    def run(self):
        try:
            cache_signature = segmenter_cache_signature(self.segmenter)
            requested_face_index = self.active_face_index
            cached = None
            if self.file_path and not self.force_recompute:
                cached = load_analysis(
                    self.file_path,
                    kind=CACHE_KIND_SINGLE_FACE,
                    image_shape=self.preview_array.shape[:2],
                    face_index=requested_face_index,
                    backend_signature=cache_signature,
                )
                if cached is not None:
                    cached_faces = cached.get("faces") or []
                    if cached_faces and requested_face_index >= len(cached_faces):
                        cached = None

            detect_ms = 0.0
            segment_ms = 0.0
            if cached is not None:
                cache_status = "hit"
                detected_faces = cached.get("faces") or []
                preview_masks = cached.get("masks")
                preview_guides = cached.get("guides")
                face_index = min(requested_face_index, len(detected_faces) - 1) if detected_faces else 0
            else:
                cache_status = "stored"
                t0 = time.perf_counter()
                detected_faces = self.segmenter.list_faces(self.preview_array)
                detect_ms = (time.perf_counter() - t0) * 1000.0
                face_index = self.active_face_index if detected_faces else 0
                if detected_faces:
                    face_index = min(face_index, len(detected_faces) - 1)
                t1 = time.perf_counter()
                preview_masks, preview_guides = self.segmenter.segment_with_guides(
                    self.preview_array, face_index=face_index
                )
                segment_ms = (time.perf_counter() - t1) * 1000.0

            preview_masks_copy = _copy_masks_data(preview_masks)
            full_masks = _scale_masks_data(preview_masks_copy, self.full_shape)
            scale = 1.0 / max(self.preview_scale, 1e-6)
            full_guides = _scale_expression_guides(_copy_guides_data(preview_guides), scale, scale)

            if cached is None and self.file_path:
                save_analysis(
                    self.file_path,
                    kind=CACHE_KIND_SINGLE_FACE,
                    image_shape=self.preview_array.shape[:2],
                    masks=preview_masks_copy,
                    faces=detected_faces,
                    guides=preview_guides,
                    face_index=face_index,
                    backend_signature=segmenter_cache_signature(self.segmenter),
                )

            result = {
                "detected_faces": list(detected_faces or []),
                "face_index": int(face_index),
                "mask_face_index": int(face_index),
                "preview_masks": preview_masks_copy,
                "preview_guides": _copy_guides_data(preview_guides),
                "full_masks": full_masks,
                "full_guides": full_guides,
                "detect_ms": detect_ms,
                "segment_ms": segment_ms,
                "cache_status": cache_status,
            }
            self.signals.finished.emit(self.job_id, result)
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))


class MaskDiagnosticsSignals(QObject):
    finished = Signal(object, object)  # detected_faces (list), {face_index: {mask_key: stats_dict}}
    failed = Signal(str)


def _mask_stats(mask) -> dict:
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "coverage_10": float((arr > 0.10).mean()) * 100.0 if arr.size else 0.0,
        "coverage_50": float((arr > 0.50).mean()) * 100.0 if arr.size else 0.0,
    }


class MaskDiagnosticsTask(QRunnable):
    """Recomputes segmentation for *every* detected face (not just the active one) and reports
    per-layer coverage stats off the GUI thread -- a deliberate, on-demand diagnostic so a user
    hitting an inconsistent/empty mask on some face or image can see exactly which face/layer
    combination is at fault, instead of guessing from how Mask View looks on screen. Each face
    costs roughly as much as a normal segmentation pass, so this is never run automatically."""

    def __init__(self, preview_array: np.ndarray, segmenter):
        super().__init__()
        self.preview_array = preview_array
        self.segmenter = segmenter
        self.signals = MaskDiagnosticsSignals()

    def run(self):
        try:
            faces = self.segmenter.list_faces(self.preview_array)
            report = {}
            face_count = len(faces) if faces else 1
            for idx in range(face_count):
                masks, _guides = self.segmenter.segment_with_guides(self.preview_array, face_index=idx)
                stats = {key: _mask_stats(mask) for key, mask in masks.items()}
                stats["_person_split_confidence"] = getattr(self.segmenter, "person_split_confidence", "")
                report[idx] = stats
            self.signals.finished.emit(list(faces or []), report)
        except Exception as ex:
            self.signals.failed.emit(str(ex))


class FolderScanSignals(QObject):
    finished = Signal(list)
    failed = Signal(str)


class FolderScanTask(QRunnable):
    """Scans a folder (optionally recursive) for supported images off the GUI thread -- a
    folder with thousands of files (or on slow/network storage) could otherwise freeze the
    Import dialog for several seconds with no feedback."""

    def __init__(self, folder: str, supported_exts: set, recursive: bool):
        super().__init__()
        self.folder = folder
        self.supported_exts = supported_exts
        self.recursive = recursive
        self.signals = FolderScanSignals()

    def run(self):
        try:
            folder_path = Path(self.folder)
            iterator = folder_path.rglob("*") if self.recursive else folder_path.iterdir()
            images = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in self.supported_exts)
            self.signals.finished.emit([str(p) for p in images])
        except Exception as ex:
            self.signals.failed.emit(str(ex))


class ImportDialog(QDialog):
    """Dialog to import images from a folder into a named collection."""

    def __init__(self, parent=None, supported_exts=None, existing_collections=None):
        super().__init__(parent)
        self.setWindowTitle("Import Images")
        self.setModal(True)
        self.resize(600, 240)
        self._supported_exts = supported_exts or set(SUPPORTED_IMAGE_EXTS)
        self._selected_folder = None
        self._image_count = 0
        self._collection_name_edited = False
        self._scan_pool = QThreadPool(self)
        self._scanned_paths = []
        self._scan_task = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel("Select a folder to import images from", self)
        info.setStyleSheet("font-weight: 600;")
        layout.addWidget(info)

        folder_layout = QHBoxLayout()
        self.folder_label = QLineEdit(self)
        self.folder_label.setReadOnly(True)
        self.folder_label.setPlaceholderText("No folder selected")
        folder_layout.addWidget(self.folder_label)
        self.browse_btn = QPushButton("Browse...", self)
        self.browse_btn.setMaximumWidth(100)
        self.browse_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self.browse_btn)
        layout.addLayout(folder_layout)

        self.count_label = QLabel("", self)
        self.count_label.setObjectName("MutedLabel")
        layout.addWidget(self.count_label)

        self.recursive_check = QCheckBox("Include subfolders", self)
        self.recursive_check.setToolTip(
            "Scan every subfolder too -- needed for date/RAW-organized camera dumps "
            "(e.g. 2026-06-22/RAW/*.cr2), which otherwise show 0 images."
        )
        self.recursive_check.toggled.connect(self._rescan_folder)
        layout.addWidget(self.recursive_check)

        collection_layout = QHBoxLayout()
        collection_layout.addWidget(QLabel("Collection:", self))
        self.collection_combo = QComboBox(self)
        self.collection_combo.setEditable(True)
        self.collection_combo.addItems(sorted(existing_collections or [], key=str.lower))
        self.collection_combo.setCurrentText("")
        self.collection_combo.editTextChanged.connect(self._on_collection_name_edited)
        collection_layout.addWidget(self.collection_combo, 1)
        layout.addLayout(collection_layout)

        hint = QLabel("Pick an existing collection to merge into it, or type a new name.", self)
        hint.setObjectName("MutedLabel")
        layout.addWidget(hint)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_btn = buttons.button(QDialogButtonBox.Ok)
        self.ok_btn.setEnabled(False)
        layout.addWidget(buttons)

    def _on_collection_name_edited(self, _text):
        self._collection_name_edited = True

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        if not folder:
            return
        self._selected_folder = folder
        self.folder_label.setText(folder)
        self._rescan_folder()

        if not self._collection_name_edited:
            self.collection_combo.setCurrentText(Path(folder).name)
            self._collection_name_edited = False

    def _rescan_folder(self):
        if not self._selected_folder:
            return
        self.count_label.setText("Scanning...")
        self.ok_btn.setEnabled(False)
        task = FolderScanTask(self._selected_folder, self._supported_exts, self.recursive_check.isChecked())
        # Pin on self -- an unpinned local-only task was reproduced to segfault/raise "Signal
        # source has been deleted" under GC pressure once the worker thread tries to emit back.
        self._scan_task = task
        task.signals.finished.connect(self._on_scan_finished)
        task.signals.failed.connect(self._on_scan_failed)
        self._scan_pool.start(task)

    def _on_scan_finished(self, paths):
        self._scan_task = None
        self._scanned_paths = paths
        self._image_count = len(paths)
        self.count_label.setText(f"Found {self._image_count} image{'s' if self._image_count != 1 else ''}")
        self.ok_btn.setEnabled(self._image_count > 0)

    def _on_scan_failed(self, message):
        self._scan_task = None
        self._scanned_paths = []
        self._image_count = 0
        self.count_label.setText(f"Error: {message}")
        self.ok_btn.setEnabled(False)

    def selected_folder(self):
        return self._selected_folder

    def recursive(self) -> bool:
        return self.recursive_check.isChecked()

    def image_count(self):
        return self._image_count

    def scanned_paths(self):
        """Image paths from the most recent scan -- callers should reuse this instead of
        re-scanning the folder themselves."""
        return list(self._scanned_paths)

    def collection_name(self):
        return self.collection_combo.currentText().strip()


class CollectionExportDialog(QDialog):
    """Dialog to batch export every image in a collection using each image's own saved settings."""

    # Speed/quality presets. Each maps to a (disable_segmentation, throughput) pair where
    # throughput is a measured/estimated images-per-minute used only to show a rough ETA.
    # "Fast" skips all face detection (global edits only); "Best" runs full face-aware
    # segmentation so per-face/local edits are applied.
    QUALITY_PRESETS = {
        "fast": {
            "label": "Fast — global edits only (no face detection)",
            "disable_segmentation": True,
            "throughput": 17.0,
        },
        "best": {
            "label": "Best — full face-aware editing (per-face local edits)",
            "disable_segmentation": False,
            "throughput": 2.5,
        },
    }

    OUTPUT_SHARPENING_OPTIONS = (
        ("Off", "off"),
        ("Low", "low"),
        ("Standard", "standard"),
        ("High", "high"),
    )

    def __init__(self, parent=None, output_dir="", suffix="_enhanced", output_format="jpeg", skip_completed=True, image_count=0, overridden_count=0, quality="best", output_sharpening="standard", deep_denoise=False):
        super().__init__(parent)
        self.setWindowTitle("Export Collection")
        self.setModal(True)
        self.resize(640, 320)
        self._image_count = int(image_count)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        plain_count = image_count - overridden_count
        info_text = f"Export {image_count} image(s) from this collection, each using its own saved edits."
        if plain_count:
            info_text += f" {plain_count} image(s) have no saved edits and will export with default settings."
        info = QLabel(info_text, self)
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        layout.addLayout(form)

        self.output_edit = QLineEdit(output_dir)
        form.addRow("Output Folder", self._path_row(self.output_edit, self._browse_output_dir))

        self.suffix_edit = QLineEdit(suffix)
        form.addRow("Suffix", self.suffix_edit)

        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["jpeg", "png", "tiff"])
        self.format_combo.setCurrentText(output_format)
        form.addRow("Output Format", self.format_combo)

        self.quality_combo = QComboBox(self)
        for key, preset in self.QUALITY_PRESETS.items():
            self.quality_combo.addItem(preset["label"], key)
        start_idx = self.quality_combo.findData(quality if quality in self.QUALITY_PRESETS else "best")
        self.quality_combo.setCurrentIndex(max(0, start_idx))
        self.quality_combo.currentIndexChanged.connect(self._update_eta_label)
        form.addRow("Quality", self.quality_combo)

        self.eta_label = QLabel("", self)
        self.eta_label.setObjectName("MutedLabel")
        self.eta_label.setWordWrap(True)
        form.addRow("", self.eta_label)
        self._update_eta_label()

        self.output_sharpening_combo = QComboBox(self)
        for label, value in self.OUTPUT_SHARPENING_OPTIONS:
            self.output_sharpening_combo.addItem(label, value)
        sharp_idx = self.output_sharpening_combo.findData(output_sharpening)
        self.output_sharpening_combo.setCurrentIndex(sharp_idx if sharp_idx >= 0 else 2)
        self.output_sharpening_combo.setToolTip(
            "A final sharpening pass calibrated for each exported image's actual pixel size."
        )
        form.addRow("Output Sharpening", self.output_sharpening_combo)

        self.deep_denoise_check = QCheckBox("Apply Deep Denoise during export", self)
        self.deep_denoise_check.setChecked(bool(deep_denoise))
        self.deep_denoise_check.setToolTip(
            "Run the heavy full-resolution Deep Denoise model on every collection export image."
        )
        layout.addWidget(self.deep_denoise_check)

        self.skip_completed_check = QCheckBox("Skip already completed files from prior runs", self)
        self.skip_completed_check.setChecked(bool(skip_completed))
        layout.addWidget(self.skip_completed_check)

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _path_row(self, line_edit, browse_slot):
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit, 1)
        browse_btn = QPushButton("Browse...", self)
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(browse_slot)
        row_layout.addWidget(browse_btn)
        return row

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_edit.setText(path)

    def output_dir(self):
        return self.output_edit.text().strip()

    def suffix(self):
        return self.suffix_edit.text()

    def output_format(self):
        return self.format_combo.currentText()

    def quality(self):
        return self.quality_combo.currentData() or "best"

    def disable_segmentation(self):
        return bool(self.QUALITY_PRESETS[self.quality()]["disable_segmentation"])

    def output_sharpening(self):
        return self.output_sharpening_combo.currentData() or "standard"

    def deep_denoise(self):
        return self.deep_denoise_check.isChecked()

    def _update_eta_label(self):
        preset = self.QUALITY_PRESETS[self.quality()]
        rate = float(preset.get("throughput", 0.0)) or 1.0
        minutes = max(1, math.ceil(self._image_count / rate))
        self.eta_label.setText(
            f"Estimated ~{minutes} min for {self._image_count} image(s) (8 parallel workers, rough estimate)."
        )

    def skip_completed(self):
        return self.skip_completed_check.isChecked()


class FilmstripThumbnail(QFrame):
    """Clickable thumbnail widget for the image filmstrip."""

    clicked = Signal(str)  # emits the image path
    remove_requested = Signal(str)  # emits the image path

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = str(path)
        self._pixmap = None
        self._is_active = False
        self._is_selected = False
        self._has_override = False
        self._preset_state = ""
        self._preset_name = ""
        self._is_raw = is_raw_path(self.path)
        self._is_culled = False
        self._export_failed = False
        self._export_status = ""

        self.setObjectName("FilmstripThumbnail")
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(80)
        self.setFixedHeight(98)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.thumb_label = QLabel(self)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setMinimumSize(72, 60)
        self.thumb_label.setMaximumSize(72, 60)
        layout.addWidget(self.thumb_label)

        self.name_label = QLabel(self)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setObjectName("MutedLabel")
        self.name_label.setWordWrap(False)
        self.name_label.setStyleSheet("font-size: 7pt;")
        layout.addWidget(self.name_label)
        self.badge_label = QLabel(self)
        self.badge_label.setAlignment(Qt.AlignCenter)
        self.badge_label.setObjectName("MutedLabel")
        self.badge_label.setStyleSheet("font-size: 7pt;")
        layout.addWidget(self.badge_label)
        layout.addStretch(1)

        self._update_name_label()
        self._update_badges()
        self.set_active(False)

    def _update_name_label(self):
        fname = Path(self.path).stem
        truncated = fname[:12] + ("..." if len(fname) > 12 else "")
        if self._has_override:
            # A small dot marks "this image has its own saved edits" -- without it, it's easy
            # to forget which images in a collection were hand-tuned before re-applying a
            # batch preset across the whole thing.
            self.name_label.setText(f'<span style="color:#d4a853;">●</span> {truncated}')
        else:
            self.name_label.setText(truncated)
        self._update_tooltip()

    def _update_badges(self):
        badges = []
        if self._preset_state == "applied":
            badges.append('<span style="color:#79d28f;">APPL</span>')
        elif self._preset_state == "modified":
            badges.append('<span style="color:#d4a853;">MOD</span>')
        elif self._has_override:
            badges.append('<span style="color:#d4a853;">EDIT</span>')
        if self._is_raw:
            badges.append('<span style="color:#7fb4ff;">RAW</span>')
        if self._is_culled:
            badges.append('<span style="color:#d98c9f;">CULL</span>')
        if self._export_status == "queued":
            badges.append('<span style="color:#9aa4b2;">Q</span>')
        elif self._export_status == "exporting":
            badges.append('<span style="color:#7fb4ff;">RUN</span>')
        elif self._export_status == "done":
            badges.append('<span style="color:#79d28f;">DONE</span>')
        elif self._export_failed:
            badges.append('<span style="color:#ff6b6b;">FAIL</span>')
        self.badge_label.setText(" ".join(badges))
        self.badge_label.setVisible(bool(badges))
        self._update_tooltip()

    def _update_tooltip(self):
        lines = [Path(self.path).name]
        if self._preset_state == "applied":
            name = f" ({self._preset_name})" if self._preset_name else ""
            lines.append(f"Preset applied{name}")
        elif self._preset_state == "modified":
            name = f" ({self._preset_name})" if self._preset_name else ""
            lines.append(f"Preset applied and then modified{name}")
        elif self._has_override:
            lines.append("Has saved per-image edits/preset settings")
        if self._is_raw:
            lines.append("RAW source")
        if self._is_culled:
            lines.append("Set aside by Review/Cull")
        if self._export_status == "queued":
            lines.append("Queued for export")
        elif self._export_status == "exporting":
            lines.append("Exporting")
        elif self._export_status == "done":
            lines.append("Export completed")
        elif self._export_failed:
            lines.append("Failed in the latest monitored export/retry")
        tooltip = "\n".join(lines)
        self.setToolTip(tooltip)
        self.name_label.setToolTip(tooltip)
        self.badge_label.setToolTip(tooltip)

    def set_has_override(self, has_override: bool):
        has_override = bool(has_override)
        if has_override == self._has_override:
            return
        self._has_override = has_override
        self._update_name_label()
        self._update_badges()

    def set_preset_state(self, state: str = "", name: str = ""):
        state = state if state in {"applied", "modified"} else ""
        if state == self._preset_state and str(name or "") == self._preset_name:
            return
        self._preset_state = state
        self._preset_name = str(name or "")
        self._update_badges()

    def set_culled(self, culled: bool):
        self._is_culled = bool(culled)
        self._update_badges()

    def set_export_failed(self, failed: bool):
        self._export_failed = bool(failed)
        if failed:
            self._export_status = "failed"
        elif self._export_status == "failed":
            self._export_status = ""
        self._update_badges()

    def set_export_status(self, status: str = ""):
        status = status if status in {"queued", "exporting", "done", "failed"} else ""
        self._export_status = status
        self._export_failed = status == "failed"
        self._update_badges()

    def set_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None:
            self.thumb_label.setText("No preview")
            self._pixmap = None
            return
        # Scale to fit the label while keeping aspect ratio.
        scaled = pixmap.scaled(72, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb_label.setPixmap(scaled)
        self._pixmap = scaled

    def _refresh_style(self):
        if self._is_active and self._is_selected:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 2px solid #d4a853; background: #203044; }"
            )
        elif self._is_active:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 2px solid #d4a853; background: #1a1a1a; }"
            )
        elif self._is_selected:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 2px solid #5b9bd5; background: #182332; }"
            )
        else:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 1px solid #3a3a3a; background: #141414; }"
            )

    def set_active(self, active: bool):
        self._is_active = bool(active)
        self._refresh_style()

    def set_selected(self, selected: bool):
        self._is_selected = bool(selected)
        self._refresh_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)
            event.accept()
        else:
            super().mousePressEvent(event)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        remove_action = menu.addAction("Remove from Collection")
        action = menu.exec(self.mapToGlobal(pos))
        if action == remove_action:
            self.remove_requested.emit(self.path)


class CulledImagesDialog(QDialog):
    """Lists images Review/Cull moved out of a collection -- previously there was no way to
    see this list or get an image back short of editing collections.json by hand."""

    def __init__(self, parent, culled_paths, thumbnail_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Culled Images")
        self.setModal(True)
        self.resize(520, 480)
        self._thumbnail_callback = thumbnail_callback

        layout = QVBoxLayout(self)
        info = QLabel(
            f"{len(culled_paths)} image(s) set aside by Review/Cull -- still on disk, just "
            "removed from this collection's filmstrip. Select any to restore them.",
            self,
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        for path in culled_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            if thumbnail_callback is not None:
                thumb = thumbnail_callback(path)
                if thumb is not None:
                    item.setIcon(QIcon(thumb))
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        restore_btn = QPushButton("Restore Selected", self)
        restore_btn.clicked.connect(self._restore_selected)
        button_row.addWidget(restore_btn)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)

        self._restored = []

    def _restore_selected(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        for item in items:
            self._restored.append(item.data(Qt.UserRole))
            self.list_widget.takeItem(self.list_widget.row(item))

    def restored_paths(self):
        return list(self._restored)


class ReviewCullDialog(QDialog):
    """Review burst/near-duplicate groups and soft frames; pick which to cull.

    Pre-checks the suggested rejects (non-keepers of each burst) so the common case is one
    click, but every checkbox is editable. Soft singles are listed but left unchecked -- they
    are flagged, not auto-culled, since softness alone isn't always a reject.
    """

    def __init__(self, parent, groups, quality, suggestions, thumb_provider):
        super().__init__(parent)
        self.setWindowTitle("Review / Cull")
        self.resize(720, 640)
        self._checks = {}  # path -> QCheckBox

        outer = QVBoxLayout(self)
        bursts = [g for g in groups if len(g) > 1]
        soft = [g[0] for g in groups if len(g) == 1 and suggestions.get(g[0]) == "soft"]
        n_suggested = sum(1 for v in suggestions.values() if v == "cull")
        header = QLabel(
            f"{len(bursts)} burst group(s), {len(soft)} soft frame(s). "
            f"Suggested rejects are pre-checked ({n_suggested}). Files are never deleted — "
            "culled images move to the collection's culled list.",
            self,
        )
        header.setWordWrap(True)
        outer.addWidget(header)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(self)
        body_layout = QVBoxLayout(body)
        body_layout.setSpacing(10)

        for gi, group in enumerate(bursts, 1):
            block = QFrame(self)
            block.setFrameShape(QFrame.StyledPanel)
            bl = QVBoxLayout(block)
            bl.addWidget(QLabel(f"<b>Burst of {len(group)}</b>", self))
            grid = QGridLayout()
            for col, path in enumerate(group):
                grid.addLayout(self._thumb_cell(path, quality, suggestions, thumb_provider), 0, col)
            bl.addLayout(grid)
            body_layout.addWidget(block)

        if soft:
            block = QFrame(self)
            block.setFrameShape(QFrame.StyledPanel)
            bl = QVBoxLayout(block)
            bl.addWidget(QLabel("<b>Soft / blurry frames</b> (flagged, not pre-checked)", self))
            grid = QGridLayout()
            per_row = 4
            for idx, path in enumerate(soft):
                r, c = divmod(idx, per_row)
                grid.addLayout(self._thumb_cell(path, quality, suggestions, thumb_provider), r, c, alignment=Qt.AlignTop)
            bl.addLayout(grid)
            body_layout.addWidget(block)

        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        self._count_label = QLabel("", self)
        outer.addWidget(self._count_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel, self)
        buttons.button(QDialogButtonBox.Apply).setText("Cull Checked")
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._update_count()

    def _thumb_cell(self, path, quality, suggestions, thumb_provider):
        cell = QVBoxLayout()
        thumb = QLabel(self)
        pixmap = thumb_provider(path) if thumb_provider else None
        if pixmap is not None and not pixmap.isNull():
            thumb.setPixmap(pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText(os.path.basename(path))
        thumb.setAlignment(Qt.AlignCenter)
        cell.addWidget(thumb)

        q = quality.get(path, {})
        blur = float(q.get("blur", 0.0))
        tag = suggestions.get(path)
        label = os.path.basename(path)
        blinks = q.get("blinks")
        eye_txt = ""
        if blinks is not None:
            eye_txt = " · eyes ok" if blinks == 0 else f" · {blinks} blinking"
        elif q.get("blink_checked"):
            eye_txt = " · eyes unknown"
        if tag == "keeper":
            info = QLabel(f"<b>KEEP</b> · sharp {blur:.0f}{eye_txt}", self)
        elif tag == "soft":
            info = QLabel(f"soft {blur:.0f}{eye_txt}", self)
        else:
            info = QLabel(f"sharp {blur:.0f}{eye_txt}", self)
        info.setAlignment(Qt.AlignCenter)
        cell.addWidget(info)

        check = QCheckBox("Cull", self)
        check.setChecked(tag == "cull")  # pre-check burst non-keepers only
        check.setEnabled(tag != "keeper")  # don't let the suggested keeper be culled by accident
        check.setToolTip(label)
        check.stateChanged.connect(self._update_count)
        self._checks[path] = check
        cell.addWidget(check, alignment=Qt.AlignCenter)
        return cell

    def _update_count(self):
        n = len(self.culled_paths())
        self._count_label.setText(f"Will cull <b>{n}</b> image(s).")

    def culled_paths(self):
        return [p for p, c in self._checks.items() if c.isChecked()]


# Simple-mode inspector: a curated, flat set of the highest-use controls, each mapping onto an
# existing (layer, key) slider under the hood -- a *view* over the same parameter model used by
# the full 8-tab Advanced surface, not a second pipeline. (layer, key, display_label, tooltip)
# "Warmth" relabels the global "Temperature" slider (the common, friendlier name for it); the
# original design's "Teeth Whiten" was dropped -- no teeth-whitening feature exists yet
# (ROADMAP.md P0) -- in favor of global Sharpness, an equally high-use, already-existing control.
SIMPLE_CONTROLS = [
    ("global", "exposure", "Exposure", "Brightens or darkens the whole image."),
    ("global", "temperature", "Warmth", "Shifts the whole image warmer (toward orange) or cooler (toward blue)."),
    ("global", "sharpness", "Sharpness", "Crisps up fine detail across the whole image."),
    ("skin", "smooth", "Skin Smooth", "Softens skin texture on the targeted face (see Face Target)."),
    ("skin", "blemish", "Blemish Fix", "Reduces the visibility of spots/blemishes on the targeted face."),
    ("eyes", "brightness", "Eye Brighten", "Brightens the eyes on the targeted face."),
    ("eyes", "whites", "Eye Whiten", "Whitens the sclera (whites of the eyes) on the targeted face."),
    ("background", "blur", "Background Blur", "Blurs the background to separate it from the subject(s)."),
]


class PortraitEnhancerQtWindow(QMainWindow):
    # Emitted (from the thumbnail worker thread) once a filmstrip thumbnail has been decoded
    # and written to the disk cache; received on the GUI thread to build the QPixmap there,
    # since QPixmap must not be created off the main thread (doing so yields black thumbnails).
    _thumbnail_ready = Signal(str)
    # Deep Denoise (NAFNet) runs on worker threads; these marshal progress and finished arrays
    # back to the GUI thread (numpy work is fine off-thread, UI updates aren't). The fast proxy
    # pass drives the live preview; the full-res pass runs in the background for export.
    _deep_denoise_progress = Signal(int, int)
    _deep_denoise_done = Signal(object)
    _deep_denoise_region_done = Signal(object)
    _denoise_point_done = Signal(object)
    _denoise_point_preview_ready = Signal(object)  # after crop (np.ndarray) or None on failure
    _deep_denoise_full_done = Signal(str, object)

    SUPPORTED_IMAGE_EXTS = SUPPORTED_IMAGE_EXTS  # module-level set (RAW formats + standard stills)
    BATCH_OUTPUT_FORMATS = {
        "jpeg": ".jpg",
        "png": ".png",
        "tiff": ".tiff",
    }
    ESSENTIAL_SLIDERS = {
        "global": {"exposure", "temperature", "tint", "highlights", "shadows", "clarity", "vibrance"},
        "subjects": {"exposure", "clarity", "saturation", "warmth"},
        # Was missing entirely -- with no entry here, EVERY Person slider was hidden in
        # Essentials Only mode (the default), making the tab look empty. Mirrors "subjects"
        # (Person is "Subjects, scoped to one person") plus the regional Noise Reduc. slider.
        "person": {"exposure", "clarity", "saturation", "warmth", "noise_red"},
        "background": {"blur", "exposure", "saturation", "dehaze", "noise_red"},
        "face": {"exposure", "smile", "eye_open", "brow_lift", "refine", "refine_fidelity"},
        "skin": {"smooth", "blemish", "warmth", "clarity", "noise_red"},
        "eyes": {"brightness", "whites", "sharpen", "iris_pop"},
        "lips": {"brightness", "saturation", "gloss", "hue_shift"},
        "hair": {"brightness", "highlights", "shine", "clarity"},
    }
    AUTO_SUGGEST_SLIDERS = {
        "global": {"noise_red", "color_noise_red", "sharpness", "exposure", "blacks", "whites"},
        # These three layers' "Auto" button re-measures noise within that layer's own mask
        # (not the whole image) -- see _auto_correct_slider's region branch.
        "skin": {"noise_red"},
        "background": {"noise_red"},
        "person": {"noise_red"},
    }
    # Per-(layer, key) tooltip overrides for sliders whose generic "range/default" tooltip
    # isn't enough -- e.g. to distinguish the Global creative Warmth/Tint grade from the
    # corrective White Balance section that shares the temp/tint concept.
    SLIDER_TOOLTIPS = {
        ("global", "temperature"): (
            "Creative warm/cool grade (hue-aware), applied after exposure. A stylistic look -- "
            "separate from the White Balance section, which corrects a color cast in Kelvin."
        ),
        ("global", "tint"): (
            "Creative green/magenta grade, applied after exposure. Separate from the White "
            "Balance Tint, which corrects a cast."
        ),
    }
    BASIC_LAYERS = ("global", "subjects", "person", "background", "face", "skin")
    BROWSER_STATE_FILE = ".preset_browser_state.json"
    # Bumped for the Preset/Template rename ("Template Browser" -> "Preset Browser") so a
    # saved section_state keyed by the old title doesn't silently mismatch the new one --
    # the existing version gate below discards stale state and falls back to defaults.
    INSPECTOR_LAYOUT_VERSION = 3
    INSPECTOR_DEFAULT_SECTION_STATE = {
        "Edit Target": True,
        "Global": True,
        "Layers": True,
        "White Balance": False,
        "Geometry": False,
        "Histogram": False,
        "Tone Curve": False,
        "Color Mixer (HSL)": False,
        "RAW Decode": False,
        "Masks": False,
    }
    PREVIEW_RENDER_CACHE_VERSION = 1
    INITIAL_WINDOW_WIDTH = 1560
    INITIAL_WINDOW_HEIGHT = 940
    INITIAL_WINDOW_SCREEN_MARGIN = 96
    RESPONSIVE_COMPACT_WIDTH = 1500

    def _fit_initial_window_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(self.INITIAL_WINDOW_WIDTH, self.INITIAL_WINDOW_HEIGHT)
            return

        available = screen.availableGeometry()
        margin = self.INITIAL_WINDOW_SCREEN_MARGIN
        frame = self.frameGeometry()
        frame_extra_width = max(0, frame.width() - self.geometry().width())
        frame_extra_height = max(0, frame.height() - self.geometry().height())
        max_width = max(1, available.width() - margin - frame_extra_width)
        max_height = max(1, available.height() - margin - frame_extra_height)
        self.resize(
            min(self.INITIAL_WINDOW_WIDTH, max_width),
            min(self.INITIAL_WINDOW_HEIGHT, max_height),
        )
        self.move(
            available.x() + max(0, (available.width() - self.frameGeometry().width()) // 2),
            available.y() + max(0, (available.height() - self.frameGeometry().height()) // 2),
        )
        self._apply_responsive_layout()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portrait Enhancer Qt")
        self.setFocusPolicy(Qt.StrongFocus)

        self.segmenter = FaceSegmenter()
        self._wb_estimator = wb_ops.WhiteBalanceEstimator()
        self._denoiser = MLDenoiser()
        self._deep_denoiser = DeepDenoiser()
        # Proxy-live deep denoise: the fast pass denoises the PREVIEW proxy (~seconds) for live
        # editing; the heavy full-res NAFNet pass runs in the background (and again, definitively,
        # at export) so editing never waits minutes. All cached per image so toggling is instant.
        self._deep_denoise_running = False         # proxy pass in flight
        self._deep_denoise_enabled = False         # is the denoise currently shown/applied?
        self._deep_denoise_preview = None          # denoised preview proxy (settled res), for live
        self._deep_denoise_interactive = None      # denoised proxy at interactive (drag) res
        self._deep_denoise_full = None             # denoised full_array, for export (background)
        self._deep_denoise_full_running = False    # background full-res pass in flight
        self._deep_denoise_path = None             # guard: only apply results to the image they ran on
        self._deep_denoise_region_running = False  # preview-only masked/cropped test pass
        self._denoise_point_enabled = False        # "Denoise Point" canvas tool armed
        self._denoise_point_running = False        # preview-only point-square test pass
        self._deep_denoise_progress.connect(self._on_deep_denoise_progress)
        self._deep_denoise_done.connect(self._on_deep_denoise_done)
        self._deep_denoise_region_done.connect(self._on_deep_denoise_region_done)
        self._denoise_point_done.connect(self._on_denoise_point_done)
        self._denoise_point_preview_ready.connect(self._on_denoise_point_preview_ready)
        self._deep_denoise_full_done.connect(self._on_deep_denoise_full_done)
        # Fix Eyes: same instant revert/re-apply cache pattern as Deep Denoise. The actual
        # blink analysis/swap runs on its own one-thread pool, same one-active-job-matters
        # pattern as segmentation/image-load above.
        self._fix_eyes_original = None
        self._fix_eyes_result = None
        self._fix_eyes_active = False
        self._fix_eyes_pool = QThreadPool(self)
        self._fix_eyes_pool.setMaxThreadCount(1)
        self._fix_eyes_job_id = 0
        self._fix_eyes_task = None
        # Auto WB (AI) across multiple selected filmstrip images: same pattern, its own pool
        # since it shares no model/state with Fix Eyes' segmenter-based analysis.
        self._wb_auto_ai_pool = QThreadPool(self)
        self._wb_auto_ai_pool.setMaxThreadCount(1)
        self._wb_auto_ai_job_id = 0
        self._wb_auto_ai_task = None
        self.file_path = ""
        self.full_array = None
        self._source_metadata = {}
        self._last_export_format = "jpeg"
        self._last_export_quality = 92
        self._last_export_resize_long_edge = 0
        self._last_export_keep_metadata = True
        self._last_export_output_sharpening = "standard"
        self._last_export_deep_denoise = False
        self._last_export_dest_dir = None  # None = use each image's own folder until overridden
        self.preview_array = None
        self.full_masks = None
        self.preview_masks = None
        self._mask_face_index = None
        self.full_guides = None
        self.preview_guides = None
        self._auto_full_masks = None
        self._auto_preview_masks = None
        self.preview_scale = 1.0
        self._interactive_preview_array = None
        self._interactive_preview_ratio = 1.0
        self.preview_image = None
        self._compare_mode = "off"
        self._split_position = 0.5
        self._compare_restore_mode = None
        self._framing = default_framing()
        self._crop_edit_enabled = False
        self._wb_pick_enabled = False
        self._click_mask_enabled = False
        self._mask_click_job_id = 0
        self._mask_click_task = None
        self._active_layer = "global"
        self._quick_layer_buttons = {}
        self._show_mask = False
        self._mask_debug_mode = "tint"
        self._show_sharpen_mask_preview = False
        self._sharpen_mask_preview = None
        self._show_expression_guides = False
        self._mask_edit_enabled = False
        self._mask_paint_mode = "paint"
        self._mask_brush_size = 24
        self._mask_brush_hardness = 100
        self._mask_adjustments = default_mask_adjustments()
        self._mask_adjustment_sliders = {}
        self._mask_adjustment_labels = {}
        self._mask_review_layers = ("person", "face", "skin", "eyes", "lips", "hair")
        self._mask_review_rows = {}
        self._document_history = []
        self._document_history_index = -1
        self._max_document_history = 30
        self._restoring_document_state = False
        # Set while programmatically syncing the RAW Decode combos to the document's settings,
        # so the change handlers don't fire a spurious re-decode during a load/restore.
        self._syncing_raw_controls = False
        self._raw_decode_section = None
        self._raw_decode_content = None
        self._preset_library_entries = []
        self._preset_browser_compact = False
        self._recent_preset_paths = []
        self._preset_category_filter = "all"
        self._detected_faces = []
        self._active_face_index = 0
        self._face_profiles = {}
        self._sliders = {}
        self._slider_value_labels = {}
        self._simple_sliders = {}
        self._simple_value_labels = {}
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(60)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_preview)
        self._render_pool = QThreadPool(self)
        self._render_pool.setMaxThreadCount(1)
        self._render_job_counter = 0
        self._active_render_job_id = 0
        self._last_completed_render_job_id = 0
        self._render_in_flight = False
        self._render_pending = False
        self._render_task = None
        self._preview_render_cache = {}
        # True full-resolution pixel zoom: a second, low-priority pipeline that renders a
        # screen-sized tile straight from the full-res source on request from the canvas.
        self._hires_pool = QThreadPool(self)
        self._hires_pool.setMaxThreadCount(1)
        self._hires_job_counter = 0
        self._hires_active_job_id = 0
        self._hires_task = None
        self._hires_max_edge = 2200
        self._preview_render_cache_order = []
        self._preview_render_cache_limit = 8
        self._preview_render_disk_cache_limit = 256
        # Cached-graph pipeline (darktable/Lightroom-style): reuses unchanged upstream pipeline
        # stages across renders so a slider drag only recomputes from the first affected stage
        # down. Replaced (not just cleared) on image switch so an in-flight worker render keeps
        # mutating its own now-orphaned cache instead of racing the GUI thread.
        self._stage_pipeline_cache = StagePipelineCache(limit=24)
        self._preview_state_revision = 0
        self._preview_analysis_signature = None
        self._source_pixels_modified = False
        self._slider_drag_active = 0
        # Active-image segmentation: its own pool at default (normal) OS thread priority,
        # separate from the low-priority background preload pool below, so opening/switching
        # the image you're actually looking at isn't stuck waiting behind queued-but-not-open
        # filmstrip selections.
        self._segmentation_pool = QThreadPool(self)
        self._segmentation_pool.setMaxThreadCount(1)
        self._segmentation_job_counter = 0
        self._active_segmentation_job_id = 0
        self._segmentation_in_flight = False
        self._segmentation_pending_callback = None
        self._segmentation_task = None
        self._segmentation_result_cache = {}
        self._segmentation_result_cache_order = []
        self._segmentation_result_cache_bytes = 0
        self._segmentation_result_cache_limit = 8
        self._segmentation_result_cache_byte_limit = 384 * 1024 * 1024
        self._mask_diag_task = None
        self._mask_diag_live_snapshot = (None, None)
        self._mask_diag_state = None
        self._mask_recalc_task = None
        self._mask_recalc_job_id = 0
        # Image decode (RAW demosaic): same one-active-job-matters pattern as segmentation
        # above -- a slow decode for an image the user already clicked past shouldn't matter.
        self._image_load_pool = QThreadPool(self)
        self._image_load_pool.setMaxThreadCount(1)
        self._image_load_job_id = 0
        self._image_load_task = None
        self._image_load_project_state = None
        self._lazy_source_path = None
        self._lazy_source_project_state = None
        self._lazy_source_loading = False
        self._lazy_hydrate_callbacks = []
        self._lazy_hydrate_settings_payload = None
        self._focus_mode = False
        self._focus_restore = None
        self._view_mode = "edit"
        self._view_mode_buttons = {}
        self._mode_restore_panels = None
        self._active_preset_meta = None
        self._pending_preset_preview_meta = None
        self._settings_clipboard = None
        self._collections = {}
        self._active_collection = None
        self._imported_images = []
        self._import_queue = []
        # Worker threads stash per-image cull signals (blur/ahash) here; the GUI thread drains
        # them into the active collection's image_quality in _on_thumbnail_ready.
        self._import_quality_buffer = {}
        self._filmstrip_selected_paths = set()
        self._filmstrip_selection_anchor = None
        self._filmstrip_filter = "all"
        self._filmstrip_failed_paths = set()
        self._filmstrip_export_status = {}
        self._import_worker_timer = QTimer(self)
        self._import_worker_timer.setSingleShot(False)
        self._import_worker_timer.setInterval(500)
        self._import_worker_timer.timeout.connect(self._process_import_queue)
        self._thumbnail_ready.connect(self._on_thumbnail_ready)
        self._import_thread_pool = QThreadPool(self)
        self._import_thread_pool.setMaxThreadCount(1)
        # Background analysis preload runs in a *separate OS process* (analysis_preload_runner),
        # not a same-process thread: a QThreadPool thread can be given a lower OS scheduling
        # priority hint, but it still shares this process's GIL, and measurement showed that
        # alone causes a real ~39% slowdown of active-image segmentation when both run at
        # once. A separate process has no GIL to share, so it can never block the GUI thread.
        self._analysis_preload_in_flight = set()  # paths currently queued or running anywhere
        self._analysis_preload_pending = set()  # accumulated, not yet dispatched to a process
        self._analysis_preload_jobs = {}  # job_id -> {"log_path", "paths" (set), "offset"}
        self._analysis_preload_last_status = 0.0
        self._analysis_preload_job_counter = 0
        self._analysis_preload_debounce_timer = QTimer(self)
        self._analysis_preload_debounce_timer.setSingleShot(True)
        self._analysis_preload_debounce_timer.setInterval(400)
        self._analysis_preload_debounce_timer.timeout.connect(self._flush_analysis_preload_queue)
        self._analysis_preload_poll_timer = QTimer(self)
        self._analysis_preload_poll_timer.setInterval(500)
        self._analysis_preload_poll_timer.timeout.connect(self._poll_analysis_preload_jobs)

        # Debounced refresh of the active image's filmstrip thumbnail so it reflects edits
        # once they settle, instead of staying on the unedited original.
        self._thumb_refresh_timer = QTimer(self)
        self._thumb_refresh_timer.setSingleShot(True)
        self._thumb_refresh_timer.setInterval(500)
        self._thumb_refresh_timer.timeout.connect(self._refresh_active_thumbnail)

        self._color_settings = self._default_color_settings()
        self._preferences = self._load_preferences()
        self._inspector_mode = self._preferences.get("inspector_mode", "simple")
        self._runtime_settings = {"acceleration_mode": self._preferences.get("acceleration_mode", "auto")}
        self._layer_order = list(MASK_ORDER)
        self._layer_options = {layer: {"enabled": True, "opacity": 100.0, "blend_mode": "normal"} for layer in MASK_ORDER}
        self._perf_stats = {"detect_ms": None, "segment_ms": None, "render_ms": None}
        self._compact_layout = None
        self._essentials_only = True
        self._slider_blocks = {}
        self._slider_search_meta = {}
        self._section_buttons = {}
        self._section_contents = {}
        self._section_wrappers = {}
        self._section_state = dict(self.INSPECTOR_DEFAULT_SECTION_STATE)
        self._inspector_search_text = ""

        self._build_ui()
        self._harden_checked_button_styles()
        self._fit_initial_window_to_screen()
        # QTabWidget shows tab 0 ("Subjects") from construction without ever emitting
        # currentChanged for it (Qt only emits on an actual index *change*), so without this,
        # self._active_layer stays at its "global" default while the UI visually shows
        # Subjects selected -- and Mask View silently shows nothing (masks has no "global"
        # key) until the user clicks to a different tab and back. Force the sync
        # currentChanged would have done, now that every widget _on_layer_changed touches
        # (mask_view_btn, etc., all built inside _build_ui) actually exists.
        self._on_layer_changed(self.layer_tabs.currentIndex())
        self._load_browser_state()
        self._load_collections_state()
        self._update_empty_state()
        self._apply_essentials_filter()
        self._refresh_recent_presets()
        self._clear_document_history()
        QTimer.singleShot(0, self._maybe_show_startup_readiness)
        self.statusBar().showMessage("Ready")

    def _apply_window_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #0f1115;
                color: #e7e4dc;
            }
            QWidget {
                color: #e7e4dc;
                font-size: 10pt;
            }
            QFrame#SidePanel,
            QFrame#CanvasHeader,
            QFrame#CanvasSurface,
            QFrame#StatusStrip {
                background: #15171d;
                border: 1px solid #292f3a;
                border-radius: 8px;
            }
            QFrame#CanvasSurface {
                background: #090a0d;
            }
            QScrollArea#InspectorScroll,
            QWidget#InspectorPanel {
                background: #15171d;
                font-size: 9pt;
            }
            QWidget#InspectorPanel QWidget {
                font-size: 9pt;
            }
            QLabel#InspectorTitle {
                color: #fff3d8;
                font-size: 12pt;
                font-weight: 700;
                padding: 9px 8px 3px 8px;
            }
            QScrollArea#InspectorScroll {
                border: 1px solid #292f3a;
                border-radius: 8px;
            }
            QFrame#SidePanel {
                font-size: 9pt;
            }
            QFrame#SidePanel QPushButton,
            QFrame#SidePanel QToolButton,
            QWidget#InspectorPanel QPushButton,
            QWidget#InspectorPanel QToolButton {
                padding: 3px 6px;
                min-height: 22px;
            }
            QFrame#SidePanel QLineEdit,
            QFrame#SidePanel QComboBox,
            QWidget#InspectorPanel QLineEdit,
            QWidget#InspectorPanel QComboBox {
                padding: 2px 5px;
            }
            QPushButton#CompactButton {
                background: #252d3a;
                border-color: #3d4654;
                color: #f1eee7;
                font-size: 8pt;
                padding: 2px 4px;
                min-height: 20px;
            }
            QPushButton#CompactButton:hover {
                background: #303a49;
                border-color: #566274;
            }
            QPushButton#CompactButton:checked {
                background: #8a6a2c;
                border-color: #d4a853;
                color: #fff6e2;
                font-weight: 600;
            }
            QPushButton#CompactButton:checked:hover {
                background: #9a7732;
            }
            QPushButton#SegmentButton {
                background: #222936;
                border: 1px solid #3b4554;
                border-radius: 5px;
                color: #f0f4f8;
                font-size: 8pt;
                padding: 3px 5px;
                min-height: 23px;
            }
            QPushButton#SegmentButton:hover {
                background: #303947;
                border-color: #5a6678;
            }
            QPushButton#SegmentButton:checked {
                background: #d4a853;
                border-color: #e0bd72;
                color: #16130b;
                font-weight: 700;
            }
            QPushButton#SegmentButton:disabled {
                background: #1b2029;
                border-color: #303846;
                color: #9aa4b2;
            }
            QPushButton#SectionHeader {
                background: #222a35;
                border-color: #3b4552;
                border-radius: 5px;
                color: #f0e7d6;
                font-size: 8pt;
                font-weight: 700;
                padding: 4px 7px;
                text-align: left;
                min-height: 22px;
            }
            QPushButton#SectionHeader:hover {
                background: #303846;
                border-color: #586474;
            }
            QPushButton#SectionHeader:checked {
                background: #2a3340;
                border-color: #d4a853;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                color: #fff3d8;
            }
            QWidget#SectionBody {
                background: #101720;
                border: 1px solid #333d4b;
                border-top: 0;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QWidget#SliderBlock {
                border-bottom: 1px solid #26313d;
            }
            QLabel#ControlLabel {
                color: #e0e6ee;
                font-size: 8pt;
                font-weight: 600;
            }
            QLabel#ValueBadge {
                background: #0b1018;
                border: 1px solid #465163;
                border-radius: 4px;
                color: #fff1d6;
                font-size: 8pt;
                padding: 1px 5px;
                qproperty-alignment: AlignRight | AlignVCenter;
            }
            QCheckBox {
                spacing: 6px;
                color: #e4e9ef;
                font-size: 8pt;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                border: 1px solid #3a4452;
                border-radius: 3px;
                background: #10141b;
            }
            QCheckBox::indicator:checked {
                background: #d4a853;
                border-color: #e0bd72;
            }
            QLabel#AppTitle {
                color: #f2eadc;
                font-size: 15pt;
                font-weight: 700;
            }
            QLabel#ImageStatus {
                color: #f2eadc;
                font-weight: 600;
            }
            QLabel#MutedLabel {
                color: #bac2ce;
            }
            QLabel#GroupHeader {
                color: #ccd5e1;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 8px 2px 2px 2px;
                border-bottom: 1px solid #2f3845;
            }
            QLabel#AccentHint {
                color: #f2c66d;
                font-size: 8pt;
                font-weight: 600;
                padding: 0px 2px 2px 2px;
            }
            QLabel#PreviewSwatch {
                background: #101217;
                border: 1px solid #2a2f38;
                border-radius: 6px;
                color: #9ea4ad;
            }
            QLabel#ImagePreview {
                background: #090a0d;
                color: #8f96a3;
            }
            QWidget#CanvasEmptyState {
                background: transparent;
            }
            QLabel#EmptyStateTitle {
                color: #f2eadc;
                font-size: 15pt;
                font-weight: 700;
            }
            QToolBar#MainToolbar {
                background: #15171d;
                border: 0;
                spacing: 6px;
                padding: 6px;
            }
            QToolButton,
            QPushButton {
                background: #252d39;
                border: 1px solid #3d4653;
                border-radius: 6px;
                padding: 6px 9px;
                color: #f2efe8;
            }
            QToolButton:hover,
            QPushButton:hover {
                background: #333d4b;
                border-color: #667386;
            }
            QToolButton:checked,
            QPushButton:checked {
                background: #31465f;
                border-color: #5b9bd5;
                color: #f4f8ff;
            }
            QToolButton:disabled,
            QPushButton:disabled,
            QLineEdit:disabled,
            QComboBox:disabled {
                background: #1b212b;
                border-color: #343d4b;
                color: #9ca6b4;
            }
            QLabel:disabled,
            QCheckBox:disabled {
                color: #9ca6b4;
            }
            QPushButton#PrimaryButton {
                background: #d4a853;
                border-color: #e0bd72;
                color: #15120a;
                font-weight: 700;
            }
            QPushButton#PrimaryButton:hover {
                background: #e0bd72;
            }
            QLineEdit,
            QComboBox,
            QTextEdit,
            QListWidget {
                background: #0b1018;
                border: 1px solid #3a4452;
                border-radius: 6px;
                color: #f2efe8;
                padding: 5px;
                selection-background-color: #5b9bd5;
            }
            QLineEdit:focus,
            QComboBox:focus,
            QTextEdit:focus,
            QListWidget:focus {
                border-color: #d4a853;
            }
            QComboBox QAbstractItemView {
                background: #101722;
                border: 1px solid #4a5666;
                color: #f2efe8;
                selection-background-color: #d4a853;
                selection-color: #16130b;
            }
            QTabWidget::pane {
                border: 1px solid #3a4452;
                border-radius: 6px;
                background: #0f151f;
            }
            QTabBar::tab {
                background: #222b38;
                color: #e1e7ee;
                font-size: 8pt;
                padding: 4px 5px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #d4a853;
                color: #16130b;
                font-weight: 700;
            }
            QTabBar::tab:hover:!selected {
                background: #303947;
                color: #f2efe8;
            }
            /* Primary navigation tabs (Adjust / Retouch / Crop): a prominent, full-width
               segmented control, distinct from the smaller nested per-layer tabs. */
            QTabWidget#InspectorTabs::pane {
                border: none;
                background: transparent;
            }
            QTabWidget#InspectorTabs QTabBar::tab {
                background: #1b2230;
                color: #c2cad6;
                font-size: 9pt;
                font-weight: 600;
                padding: 7px 14px;
                border: 1px solid #2c3543;
                border-radius: 0px;
                margin: 0px;
            }
            QTabWidget#InspectorTabs QTabBar::tab:first {
                border-top-left-radius: 6px;
                border-bottom-left-radius: 6px;
            }
            QTabWidget#InspectorTabs QTabBar::tab:last {
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QTabWidget#InspectorTabs QTabBar::tab:selected {
                background: #d4a853;
                color: #16130b;
                font-weight: 700;
            }
            QTabWidget#InspectorTabs QTabBar::tab:hover:!selected {
                background: #29333f;
                color: #f2efe8;
            }
            QScrollArea {
                background: transparent;
                border: 0;
            }
            QScrollBar:vertical {
                background: #101217;
                width: 8px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #3d4653;
                border-radius: 5px;
                min-height: 28px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #303846;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal,
            QSlider::add-page:horizontal {
                background: #303846;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #d4a853;
                border: 1px solid #f0d08c;
                width: 12px;
                height: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #f0c86b;
                border-color: #fff0bd;
            }
            QSlider::groove:horizontal:disabled,
            QSlider::sub-page:horizontal:disabled,
            QSlider::add-page:horizontal:disabled {
                background: #242b35;
            }
            QSlider::handle:horizontal:disabled {
                background: #5f6875;
                border-color: #707a88;
            }
            QSplitter::handle {
                background: #0f1115;
                width: 8px;
            }
            QStatusBar {
                background: #15171d;
                color: #9ea4ad;
            }
            """
        )

    def _build_empty_state(self) -> QWidget:
        empty = QWidget(self)
        empty.setObjectName("CanvasEmptyState")
        layout = QVBoxLayout(empty)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(14)
        layout.addStretch(1)

        title = QLabel("No Image Loaded", empty)
        title.setObjectName("EmptyStateTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addStretch(1)

        open_btn = QPushButton("Open Image", empty)
        open_btn.setObjectName("PrimaryButton")
        open_btn.clicked.connect(self.open_image)
        actions.addWidget(open_btn)

        import_btn = QPushButton("Import Folder", empty)
        import_btn.clicked.connect(self._import_folder)
        actions.addWidget(import_btn)

        project_btn = QPushButton("Open Project", empty)
        project_btn.clicked.connect(self.open_project)
        actions.addWidget(project_btn)

        self.empty_collection_btn = QPushButton("Open Active Collection", empty)
        self.empty_collection_btn.clicked.connect(self._open_first_collection_image)
        actions.addWidget(self.empty_collection_btn)

        actions.addStretch(1)
        layout.addLayout(actions)
        layout.addStretch(1)
        return empty

    def _update_empty_state(self):
        empty = getattr(self, "_empty_state", None)
        if empty is None:
            return
        has_image = bool(
            self.preview_array is not None
            or self.preview_image is not None
            or getattr(self, "_image_load_task", None) is not None
            or getattr(self, "_lazy_source_path", None)
        )
        empty.setVisible(not has_image)
        stack = getattr(self, "_canvas_stack", None)
        image_label = getattr(self, "image_label", None)
        if stack is not None:
            stack.setCurrentWidget(image_label if has_image and image_label is not None else empty)
        collection_btn = getattr(self, "empty_collection_btn", None)
        if collection_btn is not None:
            has_collection_images = bool(getattr(self, "_imported_images", []))
            collection_btn.setVisible(not has_image and has_collection_images)
            collection_btn.setEnabled(has_collection_images)

    def _open_first_collection_image(self):
        for path in getattr(self, "_imported_images", []):
            if os.path.isfile(path):
                try:
                    self._load_image_path(path)
                except Exception as ex:
                    QMessageBox.critical(self, "Load Error", self._friendly_error_message(ex))
                return
        self.statusBar().showMessage("Active collection has no available images")

    def _set_compact_button_text(self, button_name: str, compact_text: str, full_text: str, compact: bool):
        button = getattr(self, button_name, None)
        if button is not None:
            button.setText(compact_text if compact else full_text)

    def _apply_responsive_layout(self):
        splitter = getattr(self, "_splitter", None)
        if splitter is None:
            return
        compact = self.width() < self.RESPONSIVE_COMPACT_WIDTH
        if getattr(self, "_compact_layout", None) == compact:
            return
        self._compact_layout = compact

        nav_panel = getattr(self, "_nav_panel", None)
        inspector_panel = getattr(self, "_inspector_panel", None)
        if nav_panel is not None:
            nav_panel.setMinimumWidth(220 if compact else 240)
            nav_panel.setMaximumWidth(280 if compact else 320)
        if inspector_panel is not None:
            inspector_panel.setMinimumWidth(280 if compact else 300)
            inspector_panel.setMaximumWidth(340 if compact else 390)
        if hasattr(self, "collection_combo"):
            self.collection_combo.setMinimumWidth(150 if compact else 220)
        if hasattr(self, "split_slider"):
            self.split_slider.setMaximumWidth(120 if compact else 180)

        self._set_compact_button_text("apply_settings_collection_btn", "Apply", "Apply Settings", compact)
        self._set_compact_button_text("review_cull_btn", "Review", "Review / Cull", compact)

        side_width = 500 if compact else 590
        splitter.setSizes([
            230 if compact else 260,
            max(420, self.width() - side_width),
            300 if compact else 330,
        ])
        if getattr(self, "_view_mode", "edit") != "edit":
            self._apply_view_mode()

    def resizeEvent(self, event):  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _build_editor_shell(self):
        open_action = QAction("Open Image", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        import_action = QAction("Import Folder", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self._import_folder)
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self._undo_document_state)
        self.undo_action = undo_action
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self._redo_document_state)
        self.redo_action = redo_action
        open_preset_action = QAction("Open Preset", self)
        open_preset_action.triggered.connect(self.open_preset)
        save_preset_action = QAction("Save Preset", self)
        save_preset_action.triggered.connect(self.save_preset)
        open_project_action = QAction("Open Project", self)
        open_project_action.triggered.connect(self.open_project)
        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        recipes_action = QAction("Recipes", self)
        recipes_action.triggered.connect(self.open_recipe_dialog)
        check_action = QAction("System Check", self)
        check_action.triggered.connect(self.show_system_check)
        preferences_action = QAction("Preferences", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self.show_preferences)
        batch_export_action = QAction("Batch Export", self)
        batch_export_action.triggered.connect(self.batch_export)
        batch_jobs_action = QAction("Batch Jobs", self)
        batch_jobs_action.triggered.connect(self.view_batch_jobs)
        retry_failed_action = QAction("Retry Failed", self)
        retry_failed_action.triggered.connect(self.retry_failed_batch)
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_image)
        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.reset_all)

        copy_settings_action = QAction("Copy Settings", self)
        copy_settings_action.setShortcut("Ctrl+Alt+C")
        copy_settings_action.triggered.connect(self._copy_settings)

        paste_settings_action = QAction("Paste Settings", self)
        paste_settings_action.setShortcut("Ctrl+Alt+V")
        paste_settings_action.triggered.connect(self._paste_settings)
        self._paste_settings_action = paste_settings_action

        self._main_toolbar = None

        # View toggles: collapse the side panels / all chrome to enlarge the canvas.
        self._left_panel_action = QAction("◧ Left", self)
        self._left_panel_action.setCheckable(True)
        self._left_panel_action.setChecked(True)
        self._left_panel_action.setShortcut("Ctrl+Shift+L")
        self._left_panel_action.setToolTip("Show/hide the left panel (Ctrl+Shift+L)")
        self._left_panel_action.toggled.connect(self._on_left_panel_toggled)
        self._right_panel_action = QAction("Right ◨", self)
        self._right_panel_action.setCheckable(True)
        self._right_panel_action.setChecked(True)
        self._right_panel_action.setShortcut("Ctrl+Shift+R")
        self._right_panel_action.setToolTip("Show/hide the right inspector (Ctrl+Shift+R)")
        self._right_panel_action.toggled.connect(self._on_right_panel_toggled)
        self._focus_action = QAction("Focus", self)
        self._focus_action.setCheckable(True)
        self._focus_action.setShortcut("Ctrl+Shift+F")
        self._focus_action.setToolTip("Focus mode — hide all panels and bars (Ctrl+Shift+F)")
        self._focus_action.toggled.connect(self._on_focus_mode_toggled)
        for action in (
            open_action,
            import_action,
            open_project_action,
            save_project_action,
            undo_action,
            redo_action,
            open_preset_action,
            save_preset_action,
            recipes_action,
            check_action,
            preferences_action,
            batch_export_action,
            batch_jobs_action,
            retry_failed_action,
            reset_action,
            export_action,
            copy_settings_action,
            paste_settings_action,
            self._left_panel_action,
            self._right_panel_action,
            self._focus_action,
        ):
            self.addAction(action)

        def make_button(text: str, callback, primary: bool = False):
            button = QPushButton(text, self)
            button.clicked.connect(callback)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            if primary:
                button.setObjectName("PrimaryButton")
            return button

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        nav_panel = QFrame(self)
        self._nav_panel = nav_panel
        nav_panel.setObjectName("SidePanel")
        nav_panel.setMinimumWidth(240)
        nav_panel.setMaximumWidth(320)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.setSpacing(5)

        title = QLabel("Portrait Enhancer", self)
        title.setObjectName("AppTitle")
        nav_layout.addWidget(title)
        nav_layout.addWidget(make_button("Open Image", self.open_image, primary=True))

        project_grid = QHBoxLayout()
        project_grid.setContentsMargins(0, 0, 0, 0)
        project_grid.setSpacing(5)
        project_grid.addWidget(self._make_icon_button(
            "open-project", "Open Project", "Open a saved .peproj project file.", callback=self.open_project))
        project_grid.addWidget(self._make_icon_button(
            "save", "Save Project", "Save the current image, edits, and masks as a .peproj project.",
            callback=self.save_project))
        project_grid.addWidget(self._make_icon_button(
            "preset-open", "Open Preset", "Load saved RAW, white balance, detail, and layer settings from disk.",
            callback=self.open_preset))
        project_grid.addWidget(self._make_icon_button(
            "preset-save", "Save Preset", "Save the current RAW, white balance, detail, and layer settings.",
            callback=self.save_preset))
        project_grid.addWidget(self._make_icon_button(
            "recipes", "Recipes", "Browse all guided recipes with full descriptions.",
            callback=self.open_recipe_dialog))
        project_grid.addWidget(self._make_icon_button(
            "system-check", "System Check", "Review model/runtime readiness for every optional feature.",
            callback=self.show_system_check))
        project_grid.addWidget(self._make_icon_button(
            "preferences", "Preferences", "App-wide preferences.", callback=self.show_preferences))
        nav_layout.addLayout(project_grid)

        # Hidden unless a model/runtime check found a real issue -- stays visible for the rest
        # of the session even after the startup dialog is dismissed, so a degraded feature
        # (e.g. AI refine silently unavailable) doesn't go unnoticed until something seems
        # mysteriously not to work right.
        self.readiness_warning_btn = QPushButton("", self)
        self.readiness_warning_btn.setObjectName("ReadinessWarning")
        self.readiness_warning_btn.setStyleSheet("color: #caa14a; text-align: left;")
        self.readiness_warning_btn.setFlat(True)
        self.readiness_warning_btn.setVisible(False)
        self.readiness_warning_btn.clicked.connect(self.show_system_check)
        nav_layout.addWidget(self.readiness_warning_btn)

        batch_grid = QHBoxLayout()
        batch_grid.setContentsMargins(0, 0, 0, 0)
        batch_grid.setSpacing(5)
        batch_grid.addWidget(self._make_icon_button(
            "batch-export", "Batch Export", "Run a batch export job over a collection.", callback=self.batch_export))
        batch_grid.addWidget(self._make_icon_button(
            "batch-jobs", "Batch Jobs", "View past and in-progress batch export jobs.",
            callback=self.view_batch_jobs))
        batch_grid.addWidget(self._make_icon_button(
            "retry", "Retry Failed", "Retry the images that failed in the most recent batch.",
            callback=self.retry_failed_batch))
        batch_grid.addWidget(self._make_icon_button(
            "export-as", "Export As...", "Choose format/quality/destination/resize before exporting.",
            callback=self.export_image))
        # Primary action: writes the file immediately with whatever settings were used last
        # (or this image's own folder + jpeg/92 on the very first export), no dialog -- "Export
        # As..." stays one click away for anything that needs a different destination/format.
        batch_grid.addWidget(self._make_icon_button(
            "quick-export", "Quick Export", "Export now with the last-used format/quality/destination.",
            "<i>No dialog.</i>", callback=self.quick_export_image, primary=True))
        nav_layout.addLayout(batch_grid)

        preset_content = QWidget(self)
        preset_layout = QVBoxLayout(preset_content)
        preset_layout.setContentsMargins(6, 4, 6, 4)
        preset_layout.setSpacing(6)
        self.preset_search = QLineEdit(self)
        self.preset_search.setPlaceholderText("Search presets")
        self.preset_search.textChanged.connect(self._refresh_preset_browser)
        self.preset_search.installEventFilter(self)
        preset_layout.addWidget(self.preset_search)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(QLabel("Category", self))
        self.preset_category_combo = QComboBox(self)
        self.preset_category_combo.addItem("all")
        self.preset_category_combo.currentTextChanged.connect(self._on_preset_category_changed)
        self.preset_category_combo.installEventFilter(self)
        filter_row.addWidget(self.preset_category_combo, 1)
        preset_layout.addLayout(filter_row)

        preset_layout.addWidget(QLabel("Recent Presets", self))
        self.recent_preset_list = QListWidget(self)
        self.recent_preset_list.setMaximumHeight(88)
        self.recent_preset_list.setMouseTracking(True)
        self.recent_preset_list.installEventFilter(self)
        self.recent_preset_list.itemDoubleClicked.connect(lambda _item: self._apply_selected_recent_preset())
        self.recent_preset_list.currentRowChanged.connect(lambda _row: self._update_selected_preset_meta())
        self.recent_preset_list.itemEntered.connect(lambda _item: self._update_selected_preset_meta())
        preset_layout.addWidget(self.recent_preset_list)

        self.preset_list = QListWidget(self)
        self.preset_list.setMouseTracking(True)
        self.preset_list.installEventFilter(self)
        self.preset_list.setViewMode(QListView.IconMode)
        self.preset_list.setResizeMode(QListView.Adjust)
        self.preset_list.setMovement(QListView.Static)
        self.preset_list.setIconSize(QSize(112, 76))
        self.preset_list.setGridSize(QSize(132, 116))
        self.preset_list.setWordWrap(True)
        self.preset_list.setSpacing(6)
        self.preset_list.itemDoubleClicked.connect(lambda _item: self._apply_selected_browser_preset())
        self.preset_list.currentRowChanged.connect(lambda _row: self._update_selected_preset_meta())
        self.preset_list.itemEntered.connect(lambda _item: self._update_selected_preset_meta())
        preset_layout.addWidget(self.preset_list, 1)

        # Preview swatch + meta: hidden when nothing is selected so the blank placeholder
        # doesn't eat left-panel height when the user hasn't picked a preset yet.
        self.preset_browser_thumbnail = QLabel("", self)
        self.preset_browser_thumbnail.setAlignment(Qt.AlignCenter)
        self.preset_browser_thumbnail.setFixedHeight(96)
        self.preset_browser_thumbnail.setObjectName("PreviewSwatch")
        self.preset_browser_thumbnail.setVisible(False)
        preset_layout.addWidget(self.preset_browser_thumbnail)

        self.preset_meta_label = QLabel("", self)
        self.preset_meta_label.setWordWrap(True)
        self.preset_meta_label.setObjectName("MutedLabel")
        self.preset_meta_label.setVisible(False)
        preset_layout.addWidget(self.preset_meta_label)

        scope_row = QHBoxLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_row.setSpacing(5)
        scope_row.addWidget(QLabel("Apply", self))
        self.preset_apply_scope_combo = QComboBox(self)
        self.preset_apply_scope_combo.addItem("Current image", "current")
        self.preset_apply_scope_combo.addItem("Selected filmstrip images", "selected")
        self.preset_apply_scope_combo.addItem("Entire collection", "collection")
        self.preset_apply_scope_combo.setToolTip("Choose where the selected preset will be applied.")
        scope_row.addWidget(self.preset_apply_scope_combo, 1)
        preset_layout.addLayout(scope_row)

        preset_btn_grid = QHBoxLayout()
        preset_btn_grid.setContentsMargins(0, 0, 0, 0)
        preset_btn_grid.setSpacing(5)
        self.apply_browser_preset_btn = self._make_icon_button(
            "apply", "Apply", "Apply the selected preset using the chosen scope.",
            callback=self._apply_selected_browser_preset)
        preset_btn_grid.addWidget(self.apply_browser_preset_btn)
        self.save_browser_preset_btn = self._make_icon_button(
            "save", "Save Preset",
            "Save current RAW decode, white balance, Detail, and portrait settings for this shoot.",
            callback=self._save_preset_to_library)
        preset_btn_grid.addWidget(self.save_browser_preset_btn)
        self.rename_browser_preset_btn = self._make_icon_button(
            "rename", "Rename", "Rename the selected preset.", callback=self._rename_selected_browser_preset)
        preset_btn_grid.addWidget(self.rename_browser_preset_btn)
        self.delete_browser_preset_btn = self._make_icon_button(
            "delete", "Delete", "Delete the selected preset.", callback=self._delete_selected_browser_preset)
        preset_btn_grid.addWidget(self.delete_browser_preset_btn)
        self.refresh_browser_preset_btn = self._make_icon_button(
            "refresh", "Refresh", "Reload the preset browser from disk.", callback=self._refresh_preset_browser)
        preset_btn_grid.addWidget(self.refresh_browser_preset_btn)
        self.apply_preset_to_collection_btn = self._make_icon_button(
            "collection", "Apply to Collection",
            "Apply this preset to every image in the active collection from the same shoot.",
            "<i>Non-destructive, no export.</i>",
            callback=self._apply_browser_preset_to_collection,
        )
        preset_btn_grid.addWidget(self.apply_preset_to_collection_btn)
        self.clear_preset_edits_btn = self._make_icon_button(
            "delete", "Clear Preset Edits",
            "Remove saved preset/edit overrides from the current or selected filmstrip image(s).",
            callback=self._clear_preset_edits_for_scope,
        )
        preset_btn_grid.addWidget(self.clear_preset_edits_btn)
        preset_layout.addLayout(preset_btn_grid)
        nav_layout.addWidget(self._make_collapsible_section("Preset Browser", preset_content, expanded=True), 1)

        center_panel = QWidget(self)
        center_panel.setObjectName("CanvasColumn")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 0, 10, 0)
        center_layout.setSpacing(10)

        canvas_header = QFrame(self)
        self._canvas_header = canvas_header
        canvas_header.setObjectName("CanvasHeader")
        canvas_header_layout = QVBoxLayout(canvas_header)
        canvas_header_layout.setContentsMargins(6, 3, 6, 3)
        canvas_header_layout.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(4)
        self._view_mode_buttons = {}
        for key, label, tip in (
            ("edit", "Edit", "Full editing controls and the normal inspector."),
            ("cull", "Cull", "Emphasize collection review and filmstrip filtering."),
            ("batch", "Batch", "Emphasize collection-wide export and job controls."),
            ("compare", "Compare", "Emphasize before/after controls for visual checking."),
            ("focus", "Focus", "Hide panels and chrome for the largest canvas."),
        ):
            btn = QPushButton(label, self)
            btn.setObjectName("SegmentButton")
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, value=key: self._set_view_mode(value))
            self._view_mode_buttons[key] = btn
            mode_row.addWidget(btn)
        mode_row.addStretch(1)
        canvas_header_layout.addLayout(mode_row)

        # Single compact row: recipe actions | zoom | stretch | compare | copy/paste | status
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        # Recipe strip inline (left side).
        top_row.addWidget(self._build_recipe_strip())
        top_row.addSpacing(6)

        # Zoom level — tooltip explains controls.
        self.zoom_indicator = QLabel("Fit", self)
        self.zoom_indicator.setObjectName("MutedLabel")
        self.zoom_indicator.setToolTip("Current zoom level — scroll/pinch to zoom, double-click to reset")
        top_row.addWidget(self.zoom_indicator)

        # info label kept (hidden; updated by the rest of the app for status bar text).
        self.info_label = QLabel("", self)
        self.info_label.setObjectName("ImageStatus")
        self.info_label.setVisible(False)

        top_row.addStretch(1)

        # Compare controls. The "Hold Space to view the original" hint used to live on its own
        # "Before/after" label (removed when the top bar was condensed to one row) -- restored
        # here as a tooltip so the shortcut stays discoverable.
        cmp_label = QLabel("Compare", self)
        cmp_label.setObjectName("MutedLabel")
        cmp_label.setToolTip("Hold Space to view the original")
        top_row.addWidget(cmp_label)
        self.compare_hint_label = cmp_label
        self.compare_combo = QComboBox(self)
        self.compare_combo.setAccessibleName("Compare Mode")
        self.compare_combo.setToolTip("Hold Space to view the original")
        self.compare_combo.addItems(["off", "before", "split", "side_by_side"])
        self.compare_combo.setMaximumWidth(110)
        self.compare_combo.currentTextChanged.connect(self._on_compare_mode_changed)
        top_row.addWidget(self.compare_combo)

        self.split_slider = QSlider(Qt.Horizontal, self)
        self.split_slider.setRange(0, 100)
        self.split_slider.setMaximumWidth(120)
        self.split_slider.setValue(int(self._split_position * 100))
        self.split_slider.setEnabled(self._compare_mode == "split")
        self.split_slider.valueChanged.connect(self._on_split_slider_changed)
        top_row.addWidget(self.split_slider)

        top_row.addSpacing(6)

        self.copy_settings_btn = self._make_icon_button(
            "copy", "Copy Settings",
            "Copy global and selective adjustments.",
            "<i>Ctrl+Alt+C</i>",
            callback=self._copy_settings,
        )
        top_row.addWidget(self.copy_settings_btn)

        self.paste_settings_btn = self._make_icon_button(
            "paste", "Paste Settings",
            "Paste settings onto this image.",
            "Pastes to every selected filmstrip image at once if more than one is selected.",
            "<i>Ctrl+Alt+V</i>",
            callback=self._paste_settings,
        )
        self.paste_settings_btn.setEnabled(False)
        self._paste_settings_btn = self.paste_settings_btn
        top_row.addWidget(self.paste_settings_btn)

        self.settings_indicator = QLabel("", self)
        self.settings_indicator.setObjectName("MutedLabel")
        self.settings_indicator.setMaximumWidth(80)
        top_row.addWidget(self.settings_indicator)

        canvas_header_layout.addLayout(top_row)

        # Preset preview bar — hidden until a preset is being previewed.
        self.preset_preview_bar = QFrame(self)
        self.preset_preview_bar.setVisible(False)
        preview_bar_layout = QHBoxLayout(self.preset_preview_bar)
        preview_bar_layout.setContentsMargins(0, 2, 0, 2)
        self.preset_preview_label = QLabel("", self)
        preview_bar_layout.addWidget(self.preset_preview_label, 1)
        preset_keep_btn = QPushButton("Keep", self)
        preset_keep_btn.clicked.connect(self._keep_preset_preview)
        preview_bar_layout.addWidget(preset_keep_btn)
        preset_discard_btn = QPushButton("Discard", self)
        preset_discard_btn.clicked.connect(self._discard_preset_preview)
        preview_bar_layout.addWidget(preset_discard_btn)
        canvas_header_layout.addWidget(self.preset_preview_bar)
        self._preset_preview_snapshot = None

        center_layout.addWidget(canvas_header)

        canvas_frame = QFrame(self)
        canvas_frame.setObjectName("CanvasSurface")
        canvas_layout = QStackedLayout(canvas_frame)
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.setSpacing(0)
        canvas_layout.setStackingMode(QStackedLayout.StackAll)
        self._canvas_stack = canvas_layout
        self.image_label = ImagePreviewLabel(self)
        self.image_label.setObjectName("ImagePreview")
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.image_label.set_edit_state(False, self._paint_active_mask_at, self._begin_mask_stroke, self._commit_mask_stroke)
        self.image_label.set_crop_callbacks(
            self._on_crop_changed, self._on_crop_committed, self._on_crop_drag_start
        )
        self.image_label.set_wb_pick_state(False, self._on_wb_picked)
        self.image_label.set_hires_request_callback(self._on_hires_requested)
        self.image_label.zoomChanged.connect(self._on_zoom_changed)
        canvas_layout.addWidget(self.image_label)
        self._empty_state = self._build_empty_state()
        canvas_layout.addWidget(self._empty_state)
        center_layout.addWidget(canvas_frame, 1)

        # Filmstrip for quick image navigation.
        filmstrip_frame = QFrame(self)
        self._filmstrip_frame = filmstrip_frame
        filmstrip_frame.setObjectName("FilmstripStrip")
        # 162px is the real minimum: margins(12) + collection header row + filter row + the
        # 98px-tall FilmstripThumbnail. The previous 110 clipped thumbnails by nearly half.
        filmstrip_frame.setMaximumHeight(162)
        filmstrip_frame.setMinimumHeight(0)
        filmstrip_layout = QVBoxLayout(filmstrip_frame)
        filmstrip_layout.setContentsMargins(8, 6, 8, 6)
        filmstrip_layout.setSpacing(0)

        filmstrip_header_row = QHBoxLayout()
        filmstrip_header_row.setContentsMargins(0, 0, 0, 4)
        filmstrip_header_row.setSpacing(8)
        filmstrip_caption = QLabel("Collection", self)
        filmstrip_caption.setObjectName("MutedLabel")
        filmstrip_header_row.addWidget(filmstrip_caption)
        self.collection_combo = QComboBox(self)
        self.collection_combo.setAccessibleName("Active Collection")
        self.collection_combo.setMinimumWidth(240)
        self.collection_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.collection_combo.setToolTip("Select an imported collection to show in the filmstrip")
        self.collection_combo.currentIndexChanged.connect(self._on_collection_combo_changed)
        filmstrip_header_row.addWidget(self.collection_combo)
        filmstrip_header_row.addStretch(1)

        manage_menu = QMenu(self)
        self.rename_collection_btn = QAction("Rename Collection", self)
        self.rename_collection_btn.setToolTip("Rename the selected collection")
        self.rename_collection_btn.setEnabled(False)
        self.rename_collection_btn.triggered.connect(self._rename_active_collection)
        manage_menu.addAction(self.rename_collection_btn)
        self.delete_collection_btn = QAction("Delete Collection", self)
        self.delete_collection_btn.setToolTip("Remove this collection from the app (image files are not deleted)")
        self.delete_collection_btn.setEnabled(False)
        self.delete_collection_btn.triggered.connect(self._delete_active_collection)
        manage_menu.addAction(self.delete_collection_btn)
        manage_menu.addSeparator()
        self.apply_settings_collection_btn = QAction("Apply Settings", self)
        self.apply_settings_collection_btn.setToolTip("Apply the copied settings to every image in this collection (non-destructive)")
        self.apply_settings_collection_btn.setEnabled(False)
        self.apply_settings_collection_btn.triggered.connect(self._apply_settings_to_collection)
        manage_menu.addAction(self.apply_settings_collection_btn)
        self.review_cull_btn = QAction("Review / Cull", self)
        self.review_cull_btn.setToolTip(
            "Group burst/near-duplicate shots and flag soft (blurry) frames so you can quickly "
            "set aside the rejects (moved to a culled list, never deleted from disk)"
        )
        self.review_cull_btn.setEnabled(False)
        self.review_cull_btn.triggered.connect(self._review_and_cull)
        manage_menu.addAction(self.review_cull_btn)
        self.view_culled_btn = QAction("View Culled", self)
        self.view_culled_btn.setToolTip("View images set aside by Review/Cull and restore any of them back into the collection")
        self.view_culled_btn.setEnabled(False)
        self.view_culled_btn.triggered.connect(self._view_culled_images)
        manage_menu.addAction(self.view_culled_btn)
        manage_btn = QPushButton("Manage", self)
        manage_btn.setObjectName("CompactButton")
        manage_btn.setMenu(manage_menu)
        manage_btn.setToolTip("Collection management and review actions")
        filmstrip_header_row.addWidget(manage_btn)

        self._filmstrip_filter_buttons = {}
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(3)
        for key, label, tip in (
            ("all", "All", "Show every image in the active collection."),
            ("edited", "Edited", "Show images with saved per-image edits or preset settings."),
            ("unedited", "Unedited", "Show images without saved edits."),
            ("raw", "RAW", "Show RAW source images."),
            ("failed", "Failed", "Show images that failed in the latest monitored export/retry."),
        ):
            btn = QPushButton(label, self)
            btn.setObjectName("CompactButton")
            btn.setCheckable(True)
            btn.setFixedHeight(22)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, value=key: self._set_filmstrip_filter(value))
            self._filmstrip_filter_buttons[key] = btn
            filter_row.addWidget(btn)
        self.filmstrip_count_label = QLabel("", self)
        self.filmstrip_count_label.setObjectName("MutedLabel")
        filter_row.addWidget(self.filmstrip_count_label)
        export_menu = QMenu(self)
        self.export_selected_collection_btn = QAction("Export Selected...", self)
        self.export_selected_collection_btn.setToolTip("Batch export the selected filmstrip images")
        self.export_selected_collection_btn.setEnabled(False)
        self.export_selected_collection_btn.triggered.connect(self._export_selected_collection)
        export_menu.addAction(self.export_selected_collection_btn)
        self.export_collection_btn = QAction("Export All...", self)
        self.export_collection_btn.setToolTip("Batch export every image in this collection using a preset")
        self.export_collection_btn.setEnabled(False)
        self.export_collection_btn.triggered.connect(self._export_collection)
        export_menu.addAction(self.export_collection_btn)
        self.quick_export_collection_btn = QAction("Quick Export All", self)
        self.quick_export_collection_btn.setToolTip(
            "Export the whole collection now with the last-used destination/format/quality -- "
            "no dialog. Falls back to Export... if nothing's been exported yet this session."
        )
        self.quick_export_collection_btn.setEnabled(False)
        self.quick_export_collection_btn.triggered.connect(self._quick_export_collection)
        export_menu.addAction(self.quick_export_collection_btn)
        export_btn = QPushButton("Export", self)
        export_btn.setObjectName("PrimaryButton")
        export_btn.setMenu(export_menu)
        export_btn.setToolTip("Export selected images or the whole collection")
        filmstrip_header_row.addWidget(export_btn)
        filmstrip_layout.addLayout(filmstrip_header_row)
        filmstrip_layout.addSpacing(2)
        filmstrip_layout.addLayout(filter_row)
        filmstrip_layout.addSpacing(4)

        self._filmstrip_scroll = QScrollArea(self)
        self._filmstrip_scroll.setWidgetResizable(True)
        self._filmstrip_scroll.setFrameShape(QFrame.NoFrame)
        self._filmstrip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._filmstrip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._filmstrip_host = QWidget(self)
        self._filmstrip_layout = QHBoxLayout(self._filmstrip_host)
        self._filmstrip_layout.setContentsMargins(0, 0, 0, 0)
        self._filmstrip_layout.setSpacing(6)
        self._filmstrip_scroll.setWidget(self._filmstrip_host)
        filmstrip_layout.addWidget(self._filmstrip_scroll, 1)

        self._filmstrip_items = {}  # path -> thumbnail widget
        self._filmstrip_images = []  # sorted list of image paths
        self._filmstrip_placeholder = None
        self._current_filmstrip_path = None
        center_layout.addWidget(filmstrip_frame, 0)

        status_panel = QFrame(self)
        self._status_panel = status_panel
        status_panel.setObjectName("StatusStrip")
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(12, 8, 12, 10)
        status_layout.setSpacing(6)
        self.perf_label = QLabel(
            "Perf: detect=-- ms | segment=-- ms | render=-- ms | preview=idle | expr=off | refine=off",
            self,
        )
        self.perf_label.setObjectName("MutedLabel")
        self.perf_label.setWordWrap(True)
        status_layout.addWidget(self.perf_label)
        center_layout.addWidget(status_panel)

        inspector_panel = QWidget(self)
        self._inspector_panel = inspector_panel
        inspector_panel.setObjectName("InspectorPanel")
        inspector_panel.setMinimumWidth(300)
        inspector_panel.setMaximumWidth(390)
        inspector_layout = QVBoxLayout(inspector_panel)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(5)

        workspace_content = QWidget(self)
        workspace_layout = QVBoxLayout(workspace_content)
        workspace_layout.setContentsMargins(8, 6, 8, 8)
        workspace_layout.setSpacing(5)

        quick_layer_row = QHBoxLayout()
        quick_layer_row.setContentsMargins(0, 0, 0, 0)
        quick_layer_row.setSpacing(4)
        self._quick_layer_buttons = {}
        # Region quick-picks for the Retouch tab -- portrait regions only. "global" is excluded:
        # global develop now lives in the Adjust tab, not as a retouch region here.
        for layer in (lyr for lyr in self.BASIC_LAYERS if lyr != "global"):
            button = QPushButton(self)
            button.setText(layer.title())
            button.setObjectName("SegmentButton")
            button.setCheckable(True)
            button.setChecked(layer == self._active_layer)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda _checked=False, layer_name=layer: self._activate_layer(layer_name))
            self._quick_layer_buttons[layer] = button
            quick_layer_row.addWidget(button)
        workspace_layout.addLayout(quick_layer_row)

        org_row = QHBoxLayout()
        org_row.setContentsMargins(0, 0, 0, 0)
        # Essentials-only filtering is superseded by the Basic/All view toggle. Keep the widget
        # (hidden, unwired) so existing load/restore references stay valid and harmless.
        self.essentials_only_check = QCheckBox("Essentials Only", self)
        self.essentials_only_check.setChecked(False)
        self.essentials_only_check.setVisible(False)
        org_row.addStretch(1)
        self.reset_layer_btn = QPushButton("Reset Layer", self)
        self.reset_layer_btn.setObjectName("CompactButton")
        self.reset_layer_btn.clicked.connect(self._reset_active_layer)
        org_row.addWidget(self.reset_layer_btn)
        workspace_layout.addLayout(org_row)
        target_section = self._make_collapsible_section("Edit Target", workspace_content, expanded=True)

        # Global adjustments live in their own "Basic" section; the layer tabs hold
        # only the portrait feature layers (subjects, background, face, ...).
        self._tab_layers = [layer for layer in ALL_LAYERS if layer != "global"]

        global_content = QWidget(self)
        global_layout = QVBoxLayout(global_content)
        global_layout.setContentsMargins(8, 6, 8, 8)
        global_layout.setSpacing(5)
        # Six auto-action buttons don't fit in one row at the sidebar's width -- a 3-column
        # grid (2 rows) plus a compact button style keeps every label fully readable instead
        # of truncating mid-word.
        # Grouped by behavioral contract, not just topic -- that's what actually differs for
        # the user: "Suggest" buttons are instant and just set sliders/framing (undo works
        # normally); "AI Fix" buttons are slow, bake into the source pixels, and grow their
        # own Revert/Re-apply toggle once run. Mixing the two in one row hid that difference.
        def make_auto_grid(cols=3):
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            for col in range(cols):
                grid.setColumnStretch(col, 1)
            return grid

        def add_auto_button(grid, text, slot, tooltip, row, col, col_span=1):
            button = QPushButton(text, self)
            button.setObjectName("CompactButton")
            button.setToolTip(tooltip)
            button.clicked.connect(slot)
            grid.addWidget(button, row, col, 1, col_span)
            return button

        global_layout.addWidget(self._make_group_header("Suggest"))
        suggest_grid = make_auto_grid()
        self.auto_tone_btn = add_auto_button(
            suggest_grid, "Auto Tone", self._apply_auto_tone,
            "Set Exposure/Blacks/Whites together from this image's histogram", 0, 0,
        )
        self.auto_tone_face_btn = add_auto_button(
            suggest_grid, "Tone (Face)", self._apply_auto_tone_face,
            "Set Exposure/Blacks/Whites using the detected face's brightness as the target "
            "(good for backlit/shadowed subjects); falls back to whole-image if no face is found",
            0, 1,
        )
        self.auto_subject_btn = add_auto_button(
            suggest_grid, "Auto Subject", self._apply_auto_subject,
            "One click: expose for the subject and gently separate it from the background "
            "(uses the subject mask; needs the model-based Subject backend)",
            0, 2,
        )
        self.auto_crop_btn = add_auto_button(
            suggest_grid, "Auto Crop", self._apply_auto_crop,
            "One click: crop centered on the subject (fit to the selected Aspect preset) and "
            "straighten the horizon if a confident tilt is detected. Conservative -- leaves "
            "framing alone when the frame already looks good.",
            1, 0, col_span=3,
        )
        global_layout.addLayout(suggest_grid)

        global_layout.addWidget(self._make_group_header("AI Fix"))
        ai_fix_grid = make_auto_grid(cols=2)
        self.fix_eyes_btn = add_auto_button(
            ai_fix_grid, "Fix Eyes", self._run_fix_eyes,
            "One click: if this image is part of a detected burst and someone blinked, replace "
            "their closed eyes with the open-eyes version from a nearby frame in the same burst "
            "(matched by position, blended seamlessly). No new model -- needs the collection's "
            "import-time quality data (run Review / Cull first) and other frames of the same "
            "burst to still be present in the collection.",
            0, 0,
        )
        self.deep_denoise_btn = add_auto_button(
            ai_fix_grid, "Deep Denoise", self._run_deep_denoise,
            "Heavy AI denoise (NAFNet, real-noise trained), previewed at full preview resolution "
            "(sharp, no downscale) -- it takes a while; progress shows on the button, then "
            "toggling off/on is instant. The full-resolution pass is applied to the exported file. "
            "Needs the optional deep-denoise model.",
            0, 1,
        )
        self.deep_denoise_region_btn = add_auto_button(
            ai_fix_grid, "Denoise Mask", self._run_deep_denoise_region,
            "Preview-only test: denoise just the active mask layer's bounding box, then blend "
            "it back through that mask. Use Face/Skin/Background/etc. as the active layer first.",
            1, 0,
        )
        self.denoise_point_btn = QPushButton("Denoise Point", self)
        self.denoise_point_btn.setObjectName("CompactButton")
        self.denoise_point_btn.setCheckable(True)
        self.denoise_point_btn.setToolTip(
            "Preview-only test: click a point on the image to denoise a small square centered "
            "there, blended back through a soft feather. Fast way to judge detail loss on a "
            "specific spot without waiting for the whole image (Escape to exit)."
        )
        self.denoise_point_btn.toggled.connect(self._on_denoise_point_toggled)
        ai_fix_grid.addWidget(self.denoise_point_btn, 1, 1)
        global_layout.addLayout(ai_fix_grid)

        denoise_preview_row = QHBoxLayout()
        denoise_preview_row.setSpacing(5)

        def _make_denoise_thumb_col(caption):
            col = QVBoxLayout()
            col.setSpacing(2)
            label = QLabel(caption, self)
            label.setAlignment(Qt.AlignCenter)
            col.addWidget(label)
            thumb = QLabel("--", self)
            thumb.setFixedSize(96, 96)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setScaledContents(False)
            thumb.setStyleSheet("border: 1px solid #4a5666; background: #0b1018; color: #bac2ce;")
            col.addWidget(thumb)
            return col, thumb

        before_col, self.denoise_point_before_label = _make_denoise_thumb_col("Before")
        denoise_preview_row.addLayout(before_col)
        after_col, self.denoise_point_after_label = _make_denoise_thumb_col("After")
        denoise_preview_row.addLayout(after_col)
        denoise_preview_row.addStretch(1)
        global_layout.addLayout(denoise_preview_row)

        self.denoise_point_preview_hint = QLabel(
            "Denoise Point: click a spot on the image above to compare the raw denoised square "
            "here, before the edge feather blends it back in.", self
        )
        self.denoise_point_preview_hint.setWordWrap(True)
        self.denoise_point_preview_hint.setObjectName("MutedLabel")
        global_layout.addWidget(self.denoise_point_preview_hint)

        global_layout.addStretch(1)
        # The former monolithic "Global" section is split: the one-click suggestions/AI fixes stay
        # here as "Auto", and the sliders become flat, independently collapsible sections (one per
        # LAYER_SLIDER_GROUPS["global"] group) so the Adjust tab reads as a clean develop stack.
        # Each _build_slider_block still registers into self._sliders["global"], so nothing is lost.
        global_auto_section = self._make_collapsible_section("Auto", global_content, expanded=False)

        def _global_group_section(title, keys, expanded):
            spec_by_key = {k: (k, label, mn, mx, default) for k, label, mn, mx, default in ALL_LAYERS["global"]}
            auto_keys = self.AUTO_SUGGEST_SLIDERS.get("global", set())
            content = QWidget(self)
            layout = QVBoxLayout(content)
            layout.setContentsMargins(6, 6, 8, 8)
            layout.setSpacing(5)
            for key in keys:
                if key in spec_by_key:
                    layout.addWidget(self._build_slider_block("global", *spec_by_key[key], auto_suggest_keys=auto_keys))
            return self._make_collapsible_section(title, content, expanded=expanded)

        _global_groups = {title: keys for title, keys in LAYER_SLIDER_GROUPS["global"]}
        light_tone_section = _global_group_section("Light & Tone", _global_groups.get("Light", []), expanded=True)
        color_grade_section = _global_group_section("Color", _global_groups.get("Color", []), expanded=False)
        detail_section = _global_group_section("Detail", _global_groups.get("Detail", []), expanded=False)
        effects_section = _global_group_section("Effects", _global_groups.get("Effects", []), expanded=False)

        histogram_content = QWidget(self)
        histogram_layout = QVBoxLayout(histogram_content)
        histogram_layout.setContentsMargins(8, 6, 8, 8)
        histogram_layout.setSpacing(5)
        self.histogram_widget = HistogramWidget(self)
        histogram_layout.addWidget(self.histogram_widget)
        histogram_section = self._make_collapsible_section("Histogram", histogram_content, expanded=False)

        layers_content = QWidget(self)
        layers_layout = QVBoxLayout(layers_content)
        layers_layout.setContentsMargins(8, 6, 8, 8)
        layers_layout.setSpacing(5)
        # Lives here, not in the left sidebar -- this is what the Face/Skin/Eyes/Lips tabs
        # below actually target when more than one face is detected, so it belongs right
        # next to them, not disconnected on the other side of the window.
        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_label = QLabel("Face Target", self)
        face_label.setObjectName("ControlLabel")
        face_row.addWidget(face_label)
        self.face_combo = QComboBox(self)
        self.face_combo.setAccessibleName("Face Target")
        self.face_combo.setAccessibleDescription("Selects which detected face/person the Face, Skin, Eyes, Lips, Hair, and Person tabs edit")
        self.face_combo.addItem("Auto")
        self.face_combo.currentIndexChanged.connect(self._on_face_changed)
        face_row.addWidget(self.face_combo, 1)
        layers_layout.addLayout(face_row)
        # Persistent reminder of which face/person the layer tabs below currently target --
        # without this, switching Face Target then getting distracted by the filmstrip makes
        # it easy to edit the wrong person without noticing.
        self.face_scope_label = QLabel("Editing: Auto (whole scene)", self)
        self.face_scope_label.setObjectName("AccentHint")
        layers_layout.addWidget(self.face_scope_label)
        # Group-portrait fan-out: the per-face tax (Face/Skin/Eyes/Lips/Hair/Person edits
        # repeated once per person) is the dominant cost in a group photo -- one click here
        # copies the active face's adjustments onto everyone else's own mask, instead of
        # requiring N manual repeats. Hidden until there's more than one face to fan out to.
        self.apply_to_all_faces_btn = QPushButton("Apply to All Faces", self)
        self.apply_to_all_faces_btn.setObjectName("CompactButton")
        self.apply_to_all_faces_btn.setToolTip(
            "Copy this face's Face/Skin/Eyes/Lips/Hair/Person adjustments to every other "
            "detected face in this image (each keeps its own mask -- only the adjustment "
            "values are copied). One Undo step."
        )
        self.apply_to_all_faces_btn.setVisible(False)
        self.apply_to_all_faces_btn.clicked.connect(self._apply_active_face_to_all)
        layers_layout.addWidget(self.apply_to_all_faces_btn)

        # The Basic/All view toggle and the curated Basic panel (simple_panel) live at the
        # inspector top (built in the inspector assembly), shown/hidden by that one toggle. Here
        # we build only the full per-layer tabs -- the "All" view's Retouch content.
        self.simple_panel = self._build_simple_panel()

        self.layer_tabs = QTabWidget(self)
        self.layer_tabs.setDocumentMode(True)
        # The sidebar isn't wide enough to show all layer tabs at once -- scroll arrows
        # instead of letting tabs silently clip mid-label past the panel edge.
        self.layer_tabs.setUsesScrollButtons(True)
        self.layer_tabs.tabBar().setExpanding(False)
        self.layer_tabs.tabBar().setElideMode(Qt.ElideNone)
        for layer in self._tab_layers:
            self.layer_tabs.addTab(self._build_layer_tab(layer, ALL_LAYERS[layer]), LAYER_NAMES[layer])
        self.layer_tabs.currentChanged.connect(self._on_layer_changed)
        layers_layout.addWidget(self.layer_tabs)
        layers_section = self._make_collapsible_section("Layers", layers_content, expanded=True)

        mask_content = QWidget(self)
        mask_layout = QVBoxLayout(mask_content)
        mask_layout.setContentsMargins(8, 6, 8, 8)
        mask_layout.setSpacing(5)
        mask_row = QHBoxLayout()
        self.mask_view_btn = QPushButton("Mask View", self)
        self.mask_view_btn.setObjectName("SegmentButton")
        self.mask_view_btn.setCheckable(True)
        self.mask_view_btn.toggled.connect(self._on_mask_view_toggled)
        mask_row.addWidget(self.mask_view_btn)
        self.mask_edit_btn = QPushButton("Edit Mask", self)
        self.mask_edit_btn.setObjectName("SegmentButton")
        self.mask_edit_btn.setCheckable(True)
        self.mask_edit_btn.setToolTip(
            "Paint or erase this layer's mask directly with a brush -- use this to correct "
            "where an automatic mask over- or under-covers (Escape to exit)."
        )
        self.mask_edit_btn.toggled.connect(self._on_mask_edit_toggled)
        mask_row.addWidget(self.mask_edit_btn)
        self.click_mask_btn = QPushButton("AI Select", self)
        self.click_mask_btn.setObjectName("SegmentButton")
        self.click_mask_btn.setCheckable(True)
        self.click_mask_btn.setToolTip(
            "Click the center of an object to replace this layer's mask with a precise "
            "SAM-generated mask. Undo (Ctrl+Z) if it's not right (Escape to exit)."
        )
        self.click_mask_btn.toggled.connect(self._on_click_mask_toggled)
        mask_row.addWidget(self.click_mask_btn)
        mask_layout.addLayout(mask_row)

        debug_row = QHBoxLayout()
        debug_label = QLabel("View", self)
        debug_label.setObjectName("ControlLabel")
        debug_row.addWidget(debug_label)
        self.mask_debug_combo = QComboBox(self)
        self.mask_debug_combo.setAccessibleName("Mask View Mode")
        self.mask_debug_combo.addItems(["tint", "heatmap", "isolated"])
        self.mask_debug_combo.setToolTip(
            "How Mask View renders coverage: tint overlays a color, heatmap shows intensity as "
            "a gradient, isolated shows only the masked pixels against black."
        )
        self.mask_debug_combo.currentTextChanged.connect(self._on_mask_debug_mode_changed)
        debug_row.addWidget(self.mask_debug_combo, 1)
        self.guides_btn = QPushButton("Guides", self)
        self.guides_btn.setObjectName("SegmentButton")
        self.guides_btn.setCheckable(True)
        self.guides_btn.toggled.connect(self._on_guides_toggled)
        debug_row.addWidget(self.guides_btn)
        mask_layout.addLayout(debug_row)

        self.mask_debug_label = QLabel("Mask: --", self)
        self.mask_debug_label.setObjectName("MutedLabel")
        self.mask_debug_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_debug_label)

        mask_layout.addWidget(self._make_group_header("Mask Review"))
        self.mask_review_grid = QGridLayout()
        self.mask_review_grid.setHorizontalSpacing(4)
        self.mask_review_grid.setVerticalSpacing(3)
        for col, text in enumerate(("Mask", "Layer", "Status")):
            header_label = QLabel(text, self)
            header_label.setObjectName("ControlLabel")
            self.mask_review_grid.addWidget(header_label, 0, col)
        fix_header = QLabel("Fix", self)
        fix_header.setObjectName("ControlLabel")
        self.mask_review_grid.addWidget(fix_header, 0, 3, 1, 4)
        self._mask_review_rows = {}
        for row_index, layer in enumerate(self._mask_review_layers, start=1):
            thumb = QLabel(self)
            thumb.setFixedSize(42, 28)
            thumb.setScaledContents(False)
            thumb.setStyleSheet("border: 1px solid #334155; background: #0f172a;")
            self.mask_review_grid.addWidget(thumb, row_index, 0)

            layer_btn = QPushButton(LAYER_NAMES.get(layer, layer.title()), self)
            layer_btn.setObjectName("SegmentButton")
            layer_btn.setCheckable(True)
            layer_btn.setToolTip("Select this layer and show its mask.")
            layer_btn.clicked.connect(lambda _checked=False, layer=layer: self._select_mask_review_layer(layer))
            self.mask_review_grid.addWidget(layer_btn, row_index, 1)

            status = QLabel("Pending", self)
            status.setMinimumWidth(62)
            status.setAlignment(Qt.AlignCenter)
            status.setToolTip("Open an image to review this mask.")
            self.mask_review_grid.addWidget(status, row_index, 2)

            recalc_btn = QPushButton("Recalc", self)
            recalc_btn.setObjectName("CompactButton")
            recalc_btn.setToolTip("Force a fresh model recompute for this layer only.")
            recalc_btn.clicked.connect(lambda _checked=False, layer=layer: self._review_recalculate_mask(layer))
            self.mask_review_grid.addWidget(recalc_btn, row_index, 3)

            ai_btn = QPushButton("AI", self)
            ai_btn.setObjectName("CompactButton")
            ai_btn.setToolTip("Use AI Select: click an object in the image to replace this layer's mask.")
            ai_btn.clicked.connect(lambda _checked=False, layer=layer: self._review_ai_select_mask(layer))
            self.mask_review_grid.addWidget(ai_btn, row_index, 4)

            reset_btn = QPushButton("Reset", self)
            reset_btn.setObjectName("CompactButton")
            reset_btn.setToolTip("Restore this layer to its current automatic baseline.")
            reset_btn.clicked.connect(lambda _checked=False, layer=layer: self._review_reset_mask(layer))
            self.mask_review_grid.addWidget(reset_btn, row_index, 5)

            edit_btn = QPushButton("Edit", self)
            edit_btn.setObjectName("CompactButton")
            edit_btn.setToolTip("Select this layer and enter brush mask editing.")
            edit_btn.clicked.connect(lambda _checked=False, layer=layer: self._review_edit_mask(layer))
            self.mask_review_grid.addWidget(edit_btn, row_index, 6)

            self._mask_review_rows[layer] = {
                "thumb": thumb,
                "layer": layer_btn,
                "status": status,
                "recalc": recalc_btn,
                "ai": ai_btn,
                "reset": reset_btn,
                "edit": edit_btn,
            }
        mask_layout.addLayout(self.mask_review_grid)

        mask_layout.addWidget(self._make_group_header("Mask Settings"))
        for key, label, mn, mx in (
            ("strength", "Strength", 0, 200),
            ("feather", "Feather", 0, 40),
            ("expand", "Expand", -40, 40),
        ):
            mask_layout.addLayout(self._build_mask_adjustment_row(key, label, mn, mx))

        reset_settings_row = QHBoxLayout()
        reset_settings_row.addStretch(1)
        self.reset_mask_settings_btn = QPushButton("Reset Settings", self)
        self.reset_mask_settings_btn.setObjectName("CompactButton")
        self.reset_mask_settings_btn.clicked.connect(self._reset_active_mask_settings)
        reset_settings_row.addWidget(self.reset_mask_settings_btn)
        mask_layout.addLayout(reset_settings_row)

        brush_row = QHBoxLayout()
        brush_label = QLabel("Brush", self)
        brush_label.setObjectName("ControlLabel")
        brush_row.addWidget(brush_label)
        self.brush_slider = QSlider(Qt.Horizontal, self)
        self.brush_slider.setRange(2, 80)
        self.brush_slider.setValue(self._mask_brush_size)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        brush_row.addWidget(self.brush_slider, 1)
        self.brush_value_label = QLabel(str(self._mask_brush_size), self)
        self.brush_value_label.setObjectName("ValueBadge")
        self.brush_value_label.setMinimumWidth(42)
        brush_row.addWidget(self.brush_value_label)
        mask_layout.addLayout(brush_row)

        hardness_row = QHBoxLayout()
        hardness_label = QLabel("Hardness", self)
        hardness_label.setObjectName("ControlLabel")
        hardness_row.addWidget(hardness_label)
        self.hardness_slider = QSlider(Qt.Horizontal, self)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(self._mask_brush_hardness)
        self.hardness_slider.valueChanged.connect(self._on_brush_hardness_changed)
        hardness_row.addWidget(self.hardness_slider, 1)
        self.hardness_value_label = QLabel(f"{self._mask_brush_hardness}%", self)
        self.hardness_value_label.setObjectName("ValueBadge")
        self.hardness_value_label.setMinimumWidth(42)
        hardness_row.addWidget(self.hardness_value_label)
        mask_layout.addLayout(hardness_row)

        mask_mode_row = QHBoxLayout()
        mask_mode_label = QLabel("Mode", self)
        mask_mode_label.setObjectName("ControlLabel")
        mask_mode_row.addWidget(mask_mode_label)
        self.mask_mode_combo = QComboBox(self)
        self.mask_mode_combo.addItems(["paint", "erase"])
        self.mask_mode_combo.setToolTip(
            "While Edit Mask is active: paint adds coverage where you brush, erase removes it."
        )
        self.mask_mode_combo.currentTextChanged.connect(self._on_mask_mode_changed)
        mask_mode_row.addWidget(self.mask_mode_combo, 1)
        self.reset_mask_btn = QPushButton("Reset Mask", self)
        self.reset_mask_btn.setObjectName("CompactButton")
        self.reset_mask_btn.setToolTip("Discard manual paint/erase edits, back to the automatic mask for this layer.")
        self.reset_mask_btn.clicked.connect(self._reset_active_mask)
        mask_mode_row.addWidget(self.reset_mask_btn)
        self.recalculate_mask_btn = QPushButton("Recalculate", self)
        self.recalculate_mask_btn.setObjectName("CompactButton")
        self.recalculate_mask_btn.setToolTip(
            "Clear this layer's mask and force a genuinely fresh recomputation from the models -- "
            "unlike Reset Mask, this bypasses any cached result instead of reverting to it. Only "
            "the active layer changes; other layers are untouched. Can take a while if Mask DINO "
            "is enabled for Person."
        )
        self.recalculate_mask_btn.clicked.connect(self._recalculate_active_mask)
        mask_mode_row.addWidget(self.recalculate_mask_btn)
        self.feather_mask_btn = QPushButton("Feather", self)
        self.feather_mask_btn.setObjectName("CompactButton")
        self.feather_mask_btn.setToolTip("Soften the current mask's edge by a fixed amount, one-time (use the Feather slider above for a live, adjustable version).")
        self.feather_mask_btn.clicked.connect(self._feather_active_mask)
        mask_mode_row.addWidget(self.feather_mask_btn)
        mask_layout.addLayout(mask_mode_row)

        history_row = QHBoxLayout()
        self.undo_mask_btn = QPushButton("Undo", self)
        self.undo_mask_btn.setObjectName("CompactButton")
        self.undo_mask_btn.setToolTip("Undo the last action (same as Ctrl+Z) -- mask edits share one timeline with every other edit")
        self.undo_mask_btn.clicked.connect(self._undo_document_state)
        history_row.addWidget(self.undo_mask_btn)
        self.redo_mask_btn = QPushButton("Redo", self)
        self.redo_mask_btn.setObjectName("CompactButton")
        self.redo_mask_btn.setToolTip("Redo the last undone action (same as Ctrl+Shift+Z)")
        self.redo_mask_btn.clicked.connect(self._redo_document_state)
        history_row.addWidget(self.redo_mask_btn)
        mask_layout.addLayout(history_row)

        diagnostics_row = QHBoxLayout()
        self.mask_diagnostics_btn = QPushButton("Diagnose Masks", self)
        self.mask_diagnostics_btn.setObjectName("CompactButton")
        self.mask_diagnostics_btn.setToolTip(
            "Recompute every detected face's masks and show coverage for each layer in a table "
            "-- use this to see exactly which face/layer combination is coming out empty, "
            "instead of guessing from how Mask View looks on screen."
        )
        self.mask_diagnostics_btn.clicked.connect(self._run_mask_diagnostics)
        diagnostics_row.addWidget(self.mask_diagnostics_btn)
        mask_layout.addLayout(diagnostics_row)
        masks_section = self._make_collapsible_section("Masks", mask_content, expanded=False)

        geometry_content = QWidget(self)
        geometry_layout = QVBoxLayout(geometry_content)
        geometry_layout.setContentsMargins(8, 6, 8, 8)
        geometry_layout.setSpacing(5)

        self.crop_edit_btn = QPushButton("Crop", self)
        self.crop_edit_btn.setObjectName("SegmentButton")
        self.crop_edit_btn.setCheckable(True)
        self.crop_edit_btn.toggled.connect(self._on_crop_edit_toggled)
        geometry_layout.addWidget(self.crop_edit_btn)

        aspect_row = QHBoxLayout()
        aspect_label = QLabel("Aspect", self)
        aspect_label.setObjectName("ControlLabel")
        aspect_row.addWidget(aspect_label)
        self.aspect_combo = QComboBox(self)
        self._aspect_presets = [
            ("Free", None),
            ("Original", "original"),
            ("1:1", 1.0),
            ("4:5", 4.0 / 5.0),
            ("5:4", 5.0 / 4.0),
            ("3:2", 3.0 / 2.0),
            ("2:3", 2.0 / 3.0),
            ("16:9", 16.0 / 9.0),
        ]
        for label, _value in self._aspect_presets:
            self.aspect_combo.addItem(label)
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect_preset_changed)
        aspect_row.addWidget(self.aspect_combo, 1)
        self.crop_lock_check = QCheckBox("Lock", self)
        self.crop_lock_check.setToolTip(
            "Keep the selected aspect ratio while dragging crop handles, instead of free-form resize"
        )
        self.crop_lock_check.toggled.connect(self._on_crop_lock_toggled)
        aspect_row.addWidget(self.crop_lock_check)
        geometry_layout.addLayout(aspect_row)

        straighten_row = QHBoxLayout()
        straighten_label = QLabel("Straighten", self)
        straighten_label.setObjectName("ControlLabel")
        straighten_row.addWidget(straighten_label)
        self.straighten_slider = QSlider(Qt.Horizontal, self)
        self.straighten_slider.setRange(-45, 45)
        self.straighten_slider.setValue(0)
        self.straighten_slider.valueChanged.connect(self._on_straighten_changed)
        straighten_row.addWidget(self.straighten_slider, 1)
        self.straighten_value_label = QLabel("0°", self)
        self.straighten_value_label.setObjectName("ValueBadge")
        self.straighten_value_label.setMinimumWidth(42)
        straighten_row.addWidget(self.straighten_value_label)
        geometry_layout.addLayout(straighten_row)

        flip_row = QHBoxLayout()
        self.flip_h_btn = QPushButton("Flip H", self)
        self.flip_h_btn.setObjectName("CompactButton")
        self.flip_h_btn.clicked.connect(lambda: self._on_flip("flip_h"))
        flip_row.addWidget(self.flip_h_btn)
        self.flip_v_btn = QPushButton("Flip V", self)
        self.flip_v_btn.setObjectName("CompactButton")
        self.flip_v_btn.clicked.connect(lambda: self._on_flip("flip_v"))
        flip_row.addWidget(self.flip_v_btn)
        self.reset_framing_btn = QPushButton("Reset", self)
        self.reset_framing_btn.setObjectName("CompactButton")
        self.reset_framing_btn.clicked.connect(self._reset_framing)
        flip_row.addWidget(self.reset_framing_btn)
        geometry_layout.addLayout(flip_row)
        geometry_section = self._make_collapsible_section("Geometry", geometry_content, expanded=False)

        wb_content = QWidget(self)
        wb_layout = QVBoxLayout(wb_content)
        wb_layout.setContentsMargins(8, 6, 8, 8)
        wb_layout.setSpacing(5)
        self.wb_pick_btn = QPushButton("Pick Neutral", self)
        self.wb_pick_btn.setObjectName("SegmentButton")
        self.wb_pick_btn.setCheckable(True)
        self.wb_pick_btn.setToolTip("Click a neutral area in the image")
        self.wb_pick_btn.toggled.connect(self._on_wb_pick_toggled)
        wb_layout.addWidget(self.wb_pick_btn)
        wb_auto_row = QHBoxLayout()
        self.wb_gray_btn = QPushButton("Auto Gray", self)
        self.wb_gray_btn.setObjectName("CompactButton")
        self.wb_gray_btn.clicked.connect(self._wb_auto_gray_world)
        wb_auto_row.addWidget(self.wb_gray_btn)
        self.wb_white_btn = QPushButton("Auto White", self)
        self.wb_white_btn.setObjectName("CompactButton")
        self.wb_white_btn.clicked.connect(self._wb_auto_white_patch)
        wb_auto_row.addWidget(self.wb_white_btn)
        self.wb_ai_btn = QPushButton("Auto AI", self)
        self.wb_ai_btn.setObjectName("CompactButton")
        self.wb_ai_btn.setToolTip(
            "Learned auto white balance: predicts the scene illuminant (not fooled by a "
            "dominant color like a colored wall). Falls back to Gray World if no model is installed."
        )
        self.wb_ai_btn.clicked.connect(self._wb_auto_ai)
        wb_auto_row.addWidget(self.wb_ai_btn)
        self.wb_reset_btn = QPushButton("Reset", self)
        self.wb_reset_btn.setObjectName("CompactButton")
        self.wb_reset_btn.clicked.connect(self._reset_wb)
        wb_auto_row.addWidget(self.wb_reset_btn)
        wb_layout.addLayout(wb_auto_row)

        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset", self)
        preset_label.setObjectName("ControlLabel")
        preset_row.addWidget(preset_label)
        self.wb_preset_combo = QComboBox(self)
        self.wb_preset_combo.addItem("Custom")
        for name, _temp, _tint in wb_ops.PRESETS:
            self.wb_preset_combo.addItem(name)
        self.wb_preset_combo.activated.connect(self._on_wb_preset_chosen)
        preset_row.addWidget(self.wb_preset_combo, 1)
        wb_layout.addLayout(preset_row)

        temp_row = QHBoxLayout()
        temp_label = QLabel("Temp", self)
        temp_label.setObjectName("ControlLabel")
        temp_row.addWidget(temp_label)
        self.wb_temp_slider = QSlider(Qt.Horizontal, self)
        self.wb_temp_slider.setRange(wb_ops.MIN_K, wb_ops.MAX_K)
        self.wb_temp_slider.setValue(wb_ops.NEUTRAL_K)
        self.wb_temp_slider.setToolTip(
            "Corrective white balance: color temperature in Kelvin (neutralizes a cast via "
            "per-channel gains in linear light). For a stylistic warm/cool look, use Global > Warmth."
        )
        self.wb_temp_slider.sliderPressed.connect(self._begin_document_change)
        self.wb_temp_slider.valueChanged.connect(self._on_wb_temp_changed)
        self.wb_temp_slider.sliderReleased.connect(self._push_document_history)
        temp_row.addWidget(self.wb_temp_slider, 1)
        self.wb_temp_value_label = QLabel(f"{wb_ops.NEUTRAL_K}K", self)
        self.wb_temp_value_label.setObjectName("ValueBadge")
        self.wb_temp_value_label.setMinimumWidth(54)
        temp_row.addWidget(self.wb_temp_value_label)
        wb_layout.addLayout(temp_row)

        tint_row = QHBoxLayout()
        tint_label = QLabel("Tint", self)
        tint_label.setObjectName("ControlLabel")
        tint_row.addWidget(tint_label)
        self.wb_tint_slider = QSlider(Qt.Horizontal, self)
        self.wb_tint_slider.setRange(wb_ops.TINT_MIN, wb_ops.TINT_MAX)
        self.wb_tint_slider.setValue(0)
        self.wb_tint_slider.setToolTip(
            "Corrective white balance: green/magenta tint. For a stylistic tint, use Global > Tint (G/M)."
        )
        self.wb_tint_slider.sliderPressed.connect(self._begin_document_change)
        self.wb_tint_slider.valueChanged.connect(self._on_wb_tint_changed)
        self.wb_tint_slider.sliderReleased.connect(self._push_document_history)
        tint_row.addWidget(self.wb_tint_slider, 1)
        self.wb_tint_value_label = QLabel("0", self)
        self.wb_tint_value_label.setObjectName("ValueBadge")
        self.wb_tint_value_label.setMinimumWidth(42)
        tint_row.addWidget(self.wb_tint_value_label)
        wb_layout.addLayout(tint_row)

        self.wb_status_label = QLabel("White balance: neutral", self)
        self.wb_status_label.setObjectName("MutedLabel")
        wb_layout.addWidget(self.wb_status_label)
        wb_section = self._make_collapsible_section("White Balance", wb_content, expanded=False)

        curve_content = QWidget(self)
        curve_layout = QVBoxLayout(curve_content)
        curve_layout.setContentsMargins(8, 6, 8, 8)
        curve_layout.setSpacing(5)
        self.tone_curve_widget = ToneCurveWidget(self)
        self.tone_curve_widget.set_callbacks(
            self._on_tone_curve_changed, self._on_tone_curve_commit, self._on_tone_curve_start
        )
        curve_layout.addWidget(self.tone_curve_widget)
        curve_reset_row = QHBoxLayout()
        curve_reset_row.addStretch(1)
        self.tone_curve_reset_btn = QPushButton("Reset Curve", self)
        self.tone_curve_reset_btn.setObjectName("CompactButton")
        self.tone_curve_reset_btn.clicked.connect(self._reset_tone_curve)
        curve_reset_row.addWidget(self.tone_curve_reset_btn)
        curve_layout.addLayout(curve_reset_row)
        curve_section = self._make_collapsible_section("Tone Curve", curve_content, expanded=False)

        hsl_content = QWidget(self)
        hsl_layout = QVBoxLayout(hsl_content)
        hsl_layout.setContentsMargins(8, 6, 8, 8)
        hsl_layout.setSpacing(5)
        band_row = QHBoxLayout()
        band_label = QLabel("Color", self)
        band_label.setObjectName("ControlLabel")
        band_row.addWidget(band_label)
        self.hsl_band_combo = QComboBox(self)
        for name in cm_ops.BAND_NAMES:
            self.hsl_band_combo.addItem(name.capitalize())
        self.hsl_band_combo.currentIndexChanged.connect(self._on_hsl_band_changed)
        band_row.addWidget(self.hsl_band_combo, 1)
        hsl_layout.addLayout(band_row)

        self.hsl_sliders = {}
        self.hsl_value_labels = {}
        for key, label in (("hue", "Hue"), ("sat", "Saturation"), ("lum", "Luminance")):
            row = QHBoxLayout()
            label_widget = QLabel(label, self)
            label_widget.setObjectName("ControlLabel")
            row.addWidget(label_widget)
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.sliderPressed.connect(self._begin_document_change)
            slider.valueChanged.connect(lambda value, k=key: self._on_hsl_slider_changed(k, value))
            slider.sliderReleased.connect(self._push_document_history)
            row.addWidget(slider, 1)
            value_label = QLabel("0", self)
            value_label.setObjectName("ValueBadge")
            value_label.setMinimumWidth(42)
            row.addWidget(value_label)
            hsl_layout.addLayout(row)
            self.hsl_sliders[key] = slider
            self.hsl_value_labels[key] = value_label

        hsl_reset_row = QHBoxLayout()
        hsl_reset_row.addStretch(1)
        self.hsl_reset_band_btn = QPushButton("Reset Color", self)
        self.hsl_reset_band_btn.setObjectName("CompactButton")
        self.hsl_reset_band_btn.clicked.connect(self._reset_hsl_band)
        hsl_reset_row.addWidget(self.hsl_reset_band_btn)
        self.hsl_reset_all_btn = QPushButton("Reset All", self)
        self.hsl_reset_all_btn.setObjectName("CompactButton")
        self.hsl_reset_all_btn.clicked.connect(self._reset_hsl_all)
        hsl_reset_row.addWidget(self.hsl_reset_all_btn)
        hsl_layout.addLayout(hsl_reset_row)
        hsl_section = self._make_collapsible_section("Color Mixer (HSL)", hsl_content, expanded=False)

        # RAW Decode: controls that affect how a camera RAW is developed by LibRaw, so changing
        # one re-decodes the source (handled in _on_raw_decode_combo_changed). Shown only for
        # RAW sources (_sync_raw_decode_controls toggles visibility).
        raw_content = QWidget(self)
        raw_layout = QVBoxLayout(raw_content)
        raw_layout.setContentsMargins(8, 6, 8, 8)
        raw_layout.setSpacing(5)

        def add_raw_combo(label, key, options, tooltip):
            row = QHBoxLayout()
            label_widget = QLabel(label, self)
            label_widget.setObjectName("ControlLabel")
            row.addWidget(label_widget)
            combo = QComboBox(self)
            for opt_label, opt_value in options:
                combo.addItem(opt_label, opt_value)
            combo.setToolTip(tooltip)
            combo.currentIndexChanged.connect(
                lambda _idx, k=key, c=combo: self._on_raw_decode_combo_changed(k, c)
            )
            row.addWidget(combo, 1)
            raw_layout.addLayout(row)
            return combo

        self.raw_wb_combo = add_raw_combo(
            "White Balance", "raw_white_balance",
            [("Camera (as shot)", "camera"), ("Auto", "auto")],
            "How the RAW is white-balanced at decode. Camera uses the as-shot multipliers "
            "(recommended); Auto is LibRaw's gray-world estimate.",
        )
        self.raw_colorspace_combo = add_raw_combo(
            "Color Space", "raw_colorspace",
            [("sRGB", "srgb"), ("Adobe RGB", "adobe"), ("ProPhoto", "prophoto"), ("XYZ", "xyz"), ("Raw", "raw")],
            "Output color space the RAW is decoded into.",
        )
        self.raw_highlight_combo = add_raw_combo(
            "Highlights", "raw_highlight_mode",
            [("Clip", "clip"), ("Blend", "blend"), ("Rebuild", "rebuild")],
            "Clipped-highlight handling. Clip is hard white; Blend softens the channel-clip "
            "transition; Rebuild reconstructs blown channels from neighbors (best for skies "
            "and specular falloff).",
        )
        self.raw_demosaic_combo = add_raw_combo(
            "Demosaic", "raw_demosaic",
            [("Auto (LibRaw)", "auto"), ("AHD", "ahd"), ("DCB", "dcb"), ("VNG", "vng"), ("AAHD", "aahd"), ("DHT", "dht")],
            "CFA-to-RGB reconstruction. Auto uses LibRaw's default (AHD). DCB/AAHD/DHT can "
            "resolve more fine detail where supported by your LibRaw build (falls back to the "
            "default otherwise).",
        )

        # Scene-linear processing is a render-path choice (not a decode option), so it only
        # re-renders -- handled separately from the combos above, which re-decode.
        self.raw_scene_linear_check = QCheckBox("Scene-linear (linear-light) processing", self)
        self.raw_scene_linear_check.setToolTip(
            "Process in linear light: exposure, white balance, blur, sharpen and bloom run in "
            "physically-correct linear RGB, while tone/color controls convert to display space "
            "and back. More natural highlights and exposure roll-off on RAW. Re-renders the "
            "image (no re-decode)."
        )
        self.raw_scene_linear_check.toggled.connect(self._on_scene_linear_toggled)
        raw_layout.addWidget(self.raw_scene_linear_check)

        self.raw_scene_linear_denoise_check = QCheckBox("    Noise-model-aware luma denoise (VST)", self)
        self.raw_scene_linear_denoise_check.setToolTip(
            "In scene-linear mode, denoise luminance via a variance-stabilizing transform so "
            "signal-dependent sensor noise (worst in shadows) is handled uniformly across the "
            "tonal range, instead of a fixed strength that over- or under-smooths by brightness. "
            "Chroma noise is unaffected. Requires Scene-linear; re-renders the image."
        )
        self.raw_scene_linear_denoise_check.toggled.connect(self._on_scene_linear_denoise_toggled)
        raw_layout.addWidget(self.raw_scene_linear_denoise_check)

        # Learned (DnCNN) luma denoiser. It's the sRGB-mode engine -- Scene-linear uses the VST
        # denoise instead and bypasses it -- so it's disabled while Scene-linear is on.
        self.raw_learned_denoise_check = QCheckBox("Use learned denoiser (DnCNN, sRGB)", self)
        self.raw_learned_denoise_check.setToolTip(
            "Use the learned DnCNN model for luminance noise reduction (the sRGB-mode denoiser). "
            "Scene-linear mode uses the variance-stabilizing (VST) denoise instead, so this is "
            "disabled while Scene-linear is on. Turn off Scene-linear to use the learned model."
        )
        self.raw_learned_denoise_check.toggled.connect(self._on_learned_denoise_toggled)
        raw_layout.addWidget(self.raw_learned_denoise_check)

        raw_note = QLabel("Decode combos re-decode the source; Scene-linear options only re-render.", self)
        raw_note.setObjectName("MutedLabel")
        raw_note.setWordWrap(True)
        raw_layout.addWidget(raw_note)
        raw_section = self._make_collapsible_section("RAW Decode", raw_content, expanded=False)
        self._raw_decode_section = raw_section
        self._raw_decode_content = raw_content
        # The section is always visible (discoverable at startup); its controls are RAW-only, so
        # the content is greyed (disabled) for non-RAW images rather than the whole section hidden.
        raw_content.setEnabled(False)

        # Sidebar UX: three purpose tabs. Adjust = global develop (decode -> white balance ->
        # tone -> color -> detail); Retouch = per-region portrait work (Edit Target, layers,
        # masks); Crop = framing. Histogram is pinned above the tabs as a persistent scope.
        inspector_title = QLabel("Adjustments", self)
        inspector_title.setObjectName("InspectorTitle")
        inspector_layout.addWidget(inspector_title)

        # One disclosure control replaces the old Simple/Advanced + Essentials-Only mechanisms:
        # Basic = the curated quick panel; All = the full Adjust/Retouch/Crop tabs.
        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 0, 0, 0)
        view_label = QLabel("View", self)
        view_label.setObjectName("ControlLabel")
        view_row.addWidget(view_label)
        self.simple_mode_btn = QPushButton("Basic", self)
        self.simple_mode_btn.setObjectName("SegmentButton")
        self.simple_mode_btn.setCheckable(True)
        self.simple_mode_btn.setToolTip("Basic: a curated set of the most-used controls in one panel.")
        self.simple_mode_btn.toggled.connect(self._on_simple_mode_toggled)
        view_row.addWidget(self.simple_mode_btn)
        self.advanced_mode_btn = QPushButton("All", self)
        self.advanced_mode_btn.setObjectName("SegmentButton")
        self.advanced_mode_btn.setCheckable(True)
        self.advanced_mode_btn.setToolTip("All: the full Adjust / Retouch / Crop tabs and every slider.")
        self.advanced_mode_btn.toggled.connect(self._on_advanced_mode_toggled)
        view_row.addWidget(self.advanced_mode_btn)
        view_row.addStretch(1)
        inspector_layout.addLayout(view_row)

        self.inspector_search = QLineEdit(self)
        self.inspector_search.setPlaceholderText("Find adjustment...")
        self.inspector_search.setClearButtonEnabled(True)
        self.inspector_search.setToolTip("Filter adjustment sections and sliders by name.")
        self.inspector_search.textChanged.connect(self._apply_inspector_search)
        inspector_layout.addWidget(self.inspector_search)

        inspector_layout.addWidget(histogram_section)

        def _inspector_tab(sections):
            page = QWidget(self)
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(8)
            for section in sections:
                page_layout.addWidget(section)
            page_layout.addStretch(1)
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            scroll.viewport().setStyleSheet("background: #15171d;")
            return scroll

        self._inspector_tabs = QTabWidget(self)
        self._inspector_tabs.setObjectName("InspectorTabs")
        self._inspector_tabs.setDocumentMode(True)
        self._inspector_tabs.tabBar().setExpanding(True)  # 3 primary tabs fill the panel width
        self._inspector_tabs.tabBar().setDrawBase(False)
        self._inspector_tabs.addTab(
            _inspector_tab([
                raw_section, wb_section, global_auto_section,
                light_tone_section, curve_section, color_grade_section, hsl_section,
                detail_section, effects_section,
            ]),
            "Adjust",
        )
        self._inspector_tabs.addTab(_inspector_tab([target_section, layers_section, masks_section]), "Retouch")
        self._inspector_tabs.addTab(_inspector_tab([geometry_section]), "Crop")
        inspector_layout.addWidget(self._inspector_tabs, 1)

        # Basic view: the curated quick panel, toggled against the tabs by the View control.
        simple_scroll = QScrollArea(self)
        simple_scroll.setWidgetResizable(True)
        simple_scroll.setFrameShape(QFrame.NoFrame)
        simple_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        simple_scroll.setWidget(self.simple_panel)
        simple_scroll.viewport().setStyleSheet("background: #15171d;")
        self._simple_scroll = simple_scroll
        inspector_layout.addWidget(simple_scroll, 1)

        # Apply the saved view (blockSignals: initial setup, not a user toggle). "simple" = Basic.
        is_basic = self._inspector_mode == "simple"
        self.simple_mode_btn.blockSignals(True)
        self.advanced_mode_btn.blockSignals(True)
        self.simple_mode_btn.setChecked(is_basic)
        self.advanced_mode_btn.setChecked(not is_basic)
        self.simple_mode_btn.blockSignals(False)
        self.advanced_mode_btn.blockSignals(False)
        self._simple_scroll.setVisible(is_basic)
        self._inspector_tabs.setVisible(not is_basic)

        # Wrap the inspector panel in a QScrollArea so the splitter can't stretch it beyond its
        # max width -- splitters ignore QWidget.maximumWidth() on direct children but do respect
        # the scroll wrapper's own sizing. This restores the clamping that the old outer scroll
        # provided before Phase 1.
        inspector_outer = QScrollArea(self)
        inspector_outer.setObjectName("InspectorOuter")
        inspector_outer.setWidgetResizable(True)
        inspector_outer.setFrameShape(QFrame.NoFrame)
        inspector_outer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Vertical must stay AsNeeded (not Off): the panel's minimum height (title + view toggle
        # + histogram + tab bar) can exceed a constrained splitter allocation on a small/short
        # screen. With vertical scrolling hard-disabled there'd be no way to reach the clipped
        # controls below -- the same silent-content-loss bug the filmstrip had.
        inspector_outer.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        inspector_outer.setWidget(inspector_panel)
        inspector_outer.setMinimumWidth(inspector_panel.minimumWidth())
        inspector_outer.setMaximumWidth(inspector_panel.maximumWidth())
        inspector_outer.viewport().setStyleSheet("background: #15171d;")
        # _inspector_scroll now points at the outer wrapper so show/hide still works.
        self._inspector_scroll = inspector_outer

        splitter.addWidget(nav_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(inspector_outer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 840, 330])
        self._splitter = splitter

        self._apply_window_style()
        self._refresh_mask_controls()
        self._refresh_preset_browser()
        self._apply_view_mode()

    # Local fallback styling for #SegmentButton/#CompactButton's :checked state. The global
    # window-level stylesheet already defines these rules (search "QPushButton#SegmentButton"),
    # and they are correct -- but in this app's very deep, heavily-nested widget tree (multiple
    # levels of QScrollArea/QTabWidget), the inherited :checked background/text-color reliably
    # fails to paint for some checked buttons (confirmed via direct rendering, not just visually:
    # only border-color from the rule applies, background/color silently don't, leaving an
    # unreadable label). Setting an equivalent stylesheet directly on the affected widgets fixes
    # it (a per-widget stylesheet is always applied regardless of inheritance-chain depth), so
    # this is a targeted, defensive reinforcement -- not a replacement -- of the existing rules.
    _SEGMENT_BUTTON_CHECKED_CSS = """
        QPushButton#SegmentButton {
            background: #222936; border: 1px solid #3b4554; border-radius: 5px;
            color: #f0f4f8; font-size: 8pt; padding: 3px 5px; min-height: 23px;
        }
        QPushButton#SegmentButton:hover {
            background: #303947; border-color: #5a6678;
        }
        QPushButton#SegmentButton:checked {
            background: #d4a853; border-color: #e0bd72; color: #16130b; font-weight: 700;
        }
        QPushButton#SegmentButton:disabled {
            background: #1b2029; border-color: #303846; color: #9aa4b2;
        }
    """
    _COMPACT_BUTTON_CHECKED_CSS = """
        QPushButton#CompactButton {
            background: #252d3a; border-color: #3d4654; color: #f1eee7;
            font-size: 8pt; padding: 2px 4px; min-height: 20px;
        }
        QPushButton#CompactButton:hover {
            background: #303a49; border-color: #566274;
        }
        QPushButton#CompactButton:checked {
            background: #8a6a2c; border-color: #d4a853; color: #fff6e2; font-weight: 600;
        }
        QPushButton#CompactButton:checked:hover {
            background: #9a7732;
        }
    """

    def _harden_checked_button_styles(self):
        """Apply the local-stylesheet reinforcement above to every existing checkable
        SegmentButton/CompactButton, after the full UI tree is built. Called once from
        __init__; any button created later (e.g. inside a dialog) is unaffected, but dialogs
        are shallow-nested and haven't shown this bug."""
        for btn in self.findChildren(QPushButton):
            if not btn.isCheckable():
                continue
            name = btn.objectName()
            if name == "SegmentButton":
                btn.setStyleSheet(self._SEGMENT_BUTTON_CHECKED_CSS)
            elif name == "CompactButton":
                btn.setStyleSheet(self._COMPACT_BUTTON_CHECKED_CSS)

    def _build_ui(self):
        self._build_editor_shell()

    MASK_ADJUSTMENT_TOOLTIPS = {
        "strength": "How strongly this layer's mask applies, as a percentage of the detected "
        "region -- 100% is the unmodified mask; lower values blend the effect in more lightly.",
        "feather": "Softens the mask's edge by this many pixels, so the effect fades out "
        "gradually instead of stopping with a hard line.",
        "expand": "Grows the mask outward (positive) or shrinks it inward (negative), in "
        "pixels -- useful for catching a sliver of missed hair/skin, or pulling an effect back "
        "from a region's edge.",
    }

    def _build_mask_adjustment_row(self, key: str, label: str, mn: int, mx: int):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        tooltip = self.MASK_ADJUSTMENT_TOOLTIPS.get(key, "")
        label_widget = QLabel(label)
        label_widget.setObjectName("ControlLabel")
        label_widget.setToolTip(tooltip)
        row.addWidget(label_widget)
        slider = QSlider(Qt.Horizontal)
        slider.setToolTip(tooltip)
        slider.setAccessibleName(f"Mask {label}")
        slider.setRange(int(mn), int(mx))
        slider.setValue(100 if key == "strength" else 0)
        slider.sliderPressed.connect(self._begin_document_change)
        slider.sliderPressed.connect(self._on_slider_drag_started)
        slider.sliderReleased.connect(self._on_slider_drag_finished)
        slider.valueChanged.connect(lambda value, name=key: self._on_mask_adjustment_changed(name, value))
        row.addWidget(slider, 1)
        value_label = QLabel(self._format_mask_adjustment_value(key, slider.value()))
        value_label.setObjectName("ValueBadge")
        value_label.setMinimumWidth(44)
        row.addWidget(value_label)
        self._mask_adjustment_sliders[key] = slider
        self._mask_adjustment_labels[key] = value_label
        return row

    def _format_mask_adjustment_value(self, key: str, value: float) -> str:
        value = int(round(float(value)))
        if key == "strength":
            return f"{value}%"
        if key == "expand":
            return f"{value:+d}" if value else "0"
        return str(value)

    def _build_layer_tab(self, layer: str, sliders):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_slider_stack(layer, sliders))
        return scroll

    def _build_simple_panel(self) -> QWidget:
        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)
        for layer, key, display_label, tooltip in SIMPLE_CONTROLS:
            spec_by_key = {k: (k, label, mn, mx, default) for k, label, mn, mx, default in ALL_LAYERS[layer]}
            _key, _label, mn, mx, default = spec_by_key[key]
            layout.addWidget(self._build_simple_slider_block(layer, key, display_label, mn, mx, default, tooltip))
        layout.addStretch(1)
        return content

    def _build_simple_slider_block(self, layer, key, display_label, mn, mx, default, tooltip):
        """A standalone slider that drives the real (layer, key) slider rather than being read
        directly by rendering/export -- _all_params() only ever reads self._sliders, so this
        must forward every change there to have any effect (see _on_simple_slider_changed)."""
        block = QWidget()
        block.setObjectName("SliderBlock")
        block.setAttribute(Qt.WA_StyledBackground, True)
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 2, 0, 5)
        block_layout.setSpacing(3)

        top = QHBoxLayout()
        title = QLabel(display_label)
        title.setObjectName("ControlLabel")
        title.setToolTip(tooltip)
        value_label = QLabel("0")
        value_label.setObjectName("ValueBadge")
        value_label.setMinimumWidth(42)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(value_label)
        block_layout.addLayout(top)

        slider = _ResettableSlider(Qt.Horizontal, default_value=default)
        slider.setRange(int(mn), int(mx))
        slider.setValue(int(default))
        slider.setToolTip(tooltip)
        slider.setAccessibleName(f"Simple {display_label}")
        slider.sliderPressed.connect(self._begin_document_change)
        slider.sliderPressed.connect(self._on_slider_drag_started)
        slider.sliderReleased.connect(self._on_slider_drag_finished)
        slider.valueChanged.connect(
            lambda value, layer=layer, key=key, label=value_label: self._on_simple_slider_changed(layer, key, value, label)
        )
        block_layout.addWidget(slider)

        self._simple_sliders[(layer, key)] = slider
        self._simple_value_labels[(layer, key)] = value_label
        return block

    def _on_simple_slider_changed(self, layer: str, key: str, value: int, label: QLabel):
        label.setText(f"{value:+d}" if value else "0")
        real = self._sliders.get(layer, {}).get(key)
        if real is not None and real.value() != value:
            real.setValue(value)  # triggers the real slider's own handler -> renders

    def _sync_simple_controls(self):
        """Keep the Simple panel's displayed values matching the real sliders -- called after
        every settled render, since param changes can come from many places (recipes, presets,
        undo/redo, face switches, fan-out...) that set the real slider directly, often with
        blockSignals, so a signal-based mirror alone would miss most of them."""
        for (layer, key), slider in self._simple_sliders.items():
            real = self._sliders.get(layer, {}).get(key)
            if real is None:
                continue
            value = real.value()
            if slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
            label = self._simple_value_labels.get((layer, key))
            if label is not None:
                label.setText(f"{value:+d}" if value else "0")

    def _on_simple_mode_toggled(self, checked: bool):
        """Basic view: the curated quick panel instead of the full tabs."""
        if not checked:
            return
        self.advanced_mode_btn.setChecked(False)
        self._simple_scroll.setVisible(True)
        self._inspector_tabs.setVisible(False)
        self._inspector_mode = "simple"
        self._preferences["inspector_mode"] = "simple"
        self._save_preferences()
        if self._inspector_search_text:
            self._apply_inspector_search(self._inspector_search_text)

    def _on_advanced_mode_toggled(self, checked: bool):
        """All view: the full Adjust / Retouch / Crop tabs."""
        if not checked:
            return
        self.simple_mode_btn.setChecked(False)
        self._simple_scroll.setVisible(False)
        self._inspector_tabs.setVisible(True)
        self._inspector_mode = "advanced"
        self._preferences["inspector_mode"] = "advanced"
        self._save_preferences()
        if self._inspector_search_text:
            self._apply_inspector_search(self._inspector_search_text)

    def _build_slider_stack(self, layer: str, sliders, add_stretch: bool = True):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        auto_suggest_keys = self.AUTO_SUGGEST_SLIDERS.get(layer, set())
        spec_by_key = {key: (key, label, mn, mx, default) for key, label, mn, mx, default in sliders}

        groups = LAYER_SLIDER_GROUPS.get(layer)
        if groups:
            grouped_keys = set()
            for title, keys in groups:
                present = [k for k in keys if k in spec_by_key]
                if not present:
                    continue
                layout.addWidget(self._make_group_header(title))
                for key in present:
                    grouped_keys.add(key)
                    layout.addWidget(self._build_slider_block(layer, *spec_by_key[key], auto_suggest_keys=auto_suggest_keys))
            # Any slider not assigned to a group still shows up, so this never silently
            # drops a control if the *_SLIDERS list and the group config drift apart.
            leftover = [key for key, *_ in sliders if key not in grouped_keys]
            if leftover:
                layout.addWidget(self._make_group_header("More"))
                for key in leftover:
                    layout.addWidget(self._build_slider_block(layer, *spec_by_key[key], auto_suggest_keys=auto_suggest_keys))
        else:
            for spec in sliders:
                layout.addWidget(self._build_slider_block(layer, *spec, auto_suggest_keys=auto_suggest_keys))

        if add_stretch:
            layout.addStretch(1)
        return content

    def _build_slider_block(self, layer, key, label, mn, mx, default, auto_suggest_keys=frozenset()):
        block = QWidget()
        block.setObjectName("SliderBlock")
        block.setAttribute(Qt.WA_StyledBackground, True)
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 2, 0, 5)
        block_layout.setSpacing(3)

        tooltip = self.SLIDER_TOOLTIPS.get(
            (layer, key),
            f"{label}: range {int(mn)} to {int(mx)}, default {int(default)}. Double-click the slider to reset.",
        )

        top = QHBoxLayout()
        title = QLabel(label)
        title.setObjectName("ControlLabel")
        title.setToolTip(tooltip)
        value_label = QLabel("0")
        value_label.setObjectName("ValueBadge")
        value_label.setMinimumWidth(42)
        top.addWidget(title)
        top.addStretch(1)
        if layer == "global" and key == "sharpen_masking":
            self.sharpen_mask_preview_btn = QPushButton("Preview Mask", self)
            self.sharpen_mask_preview_btn.setObjectName("CompactButton")
            self.sharpen_mask_preview_btn.setCheckable(True)
            self.sharpen_mask_preview_btn.setToolTip(
                "Show the Sharpen Masking edge mask (white = sharpened, black = protected)"
            )
            self.sharpen_mask_preview_btn.setFixedHeight(20)
            self.sharpen_mask_preview_btn.toggled.connect(self._on_sharpen_mask_preview_toggled)
            top.addWidget(self.sharpen_mask_preview_btn)
        if key in auto_suggest_keys:
            auto_btn = QPushButton("Auto", self)
            auto_btn.setObjectName("CompactButton")
            auto_btn.setToolTip("Estimate from this image's noise level")
            auto_btn.setFixedHeight(20)
            auto_btn.clicked.connect(
                lambda _checked=False, layer_name=layer, slider_key=key: self._auto_correct_slider(layer_name, slider_key)
            )
            top.addWidget(auto_btn)
        top.addWidget(value_label)
        block_layout.addLayout(top)

        slider = _ResettableSlider(Qt.Horizontal, default_value=default)
        slider.setRange(int(mn), int(mx))
        slider.setValue(int(default))
        slider.setToolTip(tooltip)
        slider.setAccessibleName(f"{layer.title()} {label}")
        slider.sliderPressed.connect(self._begin_document_change)
        slider.sliderPressed.connect(self._on_slider_drag_started)
        slider.sliderReleased.connect(self._on_slider_drag_finished)
        slider.valueChanged.connect(self._make_slider_handler(layer, key, value_label))
        block_layout.addWidget(slider)

        self._sliders.setdefault(layer, {})[key] = slider
        self._slider_value_labels.setdefault(layer, {})[key] = value_label
        self._slider_blocks.setdefault(layer, {})[key] = block
        self._slider_search_meta[(layer, key)] = self._normalize_search_text(
            layer,
            key,
            label,
            LAYER_NAMES.get(layer, layer),
            tooltip,
        )
        return block

    def _make_group_header(self, text: str) -> QLabel:
        label = QLabel(text.upper(), self)
        label.setObjectName("GroupHeader")
        return label

    def _normalize_search_text(self, *parts) -> str:
        return " ".join(str(part or "").lower().replace("_", " ") for part in parts)

    def _query_matches(self, haystack: str, query: str) -> bool:
        terms = [term for term in str(query or "").lower().split() if term]
        if not terms:
            return True
        return all(term in haystack for term in terms)

    def _widget_search_text(self, widget: QWidget, title: str = "") -> str:
        parts = [title]
        children = []
        for child_type in (QLabel, QPushButton, QCheckBox, QComboBox):
            children.extend(widget.findChildren(child_type))
        for child in children:
            if hasattr(child, "text"):
                try:
                    parts.append(child.text())
                except TypeError:
                    pass
            if isinstance(child, QComboBox):
                parts.extend(child.itemText(i) for i in range(child.count()))
            parts.append(child.toolTip())
            parts.append(child.accessibleName())
            parts.append(child.accessibleDescription())
        return self._normalize_search_text(*parts)

    def _section_has_matching_slider(self, content: QWidget, query: str) -> bool:
        for (layer, key), block in self._iter_slider_blocks():
            if content is block or content.isAncestorOf(block):
                if self._query_matches(self._slider_search_meta.get((layer, key), ""), query):
                    return True
        return False

    def _iter_slider_blocks(self):
        for layer, blocks in self._slider_blocks.items():
            for key, block in blocks.items():
                yield (layer, key), block

    def _apply_inspector_search(self, text: str = ""):
        query = str(text or "").strip().lower()
        self._inspector_search_text = query
        searching = bool(query)

        if hasattr(self, "_simple_scroll") and hasattr(self, "_inspector_tabs"):
            if searching:
                self._simple_scroll.setVisible(False)
                self._inspector_tabs.setVisible(True)
            else:
                is_basic = self._inspector_mode == "simple"
                self._simple_scroll.setVisible(is_basic)
                self._inspector_tabs.setVisible(not is_basic)

        for (layer, key), block in self._iter_slider_blocks():
            block.setVisible(not searching or self._query_matches(self._slider_search_meta.get((layer, key), ""), query))

        for title, wrapper in self._section_wrappers.items():
            content = self._section_contents.get(title)
            button = self._section_buttons.get(title)
            if content is None or button is None:
                continue
            if not searching:
                wrapper.setVisible(True)
                content.setVisible(bool(self._section_state.get(title, button.isChecked())))
                button.setChecked(bool(self._section_state.get(title, button.isChecked())))
                button.setText(self._section_button_text(title, content.isVisible()))
                continue

            title_match = self._query_matches(self._normalize_search_text(title), query)
            section_match = self._query_matches(self._widget_search_text(content, title), query)
            slider_match = self._section_has_matching_slider(content, query)
            visible = section_match or slider_match
            wrapper.setVisible(visible)
            if visible:
                if title_match:
                    for (_layer, _key), block in self._iter_slider_blocks():
                        if content is block or content.isAncestorOf(block):
                            block.setVisible(True)
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
                content.setVisible(True)
                button.setText(self._section_button_text(title, True))

    def _section_button_text(self, title: str, expanded: bool) -> str:
        marker = "▾" if expanded else "▸"
        return f"{marker}  {title}"

    def _make_collapsible_section(self, title: str, content: QWidget, expanded: bool = True):
        expanded = bool(self._section_state.get(title, expanded))
        wrapper = QWidget(self)
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        content.setObjectName("SectionBody")
        content.setAttribute(Qt.WA_StyledBackground, True)

        header = QPushButton(self)
        header.setObjectName("SectionHeader")
        header.setText(self._section_button_text(title, expanded))
        header.setProperty("section_title", title)
        header.setCheckable(True)
        header.setChecked(bool(expanded))
        header.clicked.connect(lambda checked, widget=content, button=header: self._toggle_section(widget, button, checked))
        outer.addWidget(header)
        outer.addWidget(content)
        content.setVisible(bool(expanded))
        self._section_buttons[title] = header
        self._section_contents[title] = content
        self._section_wrappers[title] = wrapper
        self._section_state[title] = bool(expanded)
        return wrapper

    def _toggle_section(self, content: QWidget, button: QPushButton, expanded: bool):
        content.setVisible(bool(expanded))
        title = str(button.property("section_title") or button.text()).strip()
        button.setText(self._section_button_text(title, expanded))
        if title:
            self._section_state[title] = bool(expanded)
            self._save_browser_state()

    def _set_section_expanded(self, title: str, expanded: bool, save: bool = True):
        button = self._section_buttons.get(title)
        content = self._section_contents.get(title)
        if button is None or content is None:
            self._section_state[title] = bool(expanded)
            return
        if button.isChecked() == bool(expanded) and content.isVisible() == bool(expanded):
            self._section_state[title] = bool(expanded)
            return
        button.blockSignals(True)
        button.setChecked(bool(expanded))
        button.blockSignals(False)
        content.setVisible(bool(expanded))
        button.setText(self._section_button_text(title, expanded))
        self._section_state[title] = bool(expanded)
        if save:
            self._save_browser_state()

    def _make_slider_handler(self, layer: str, key: str, value_label: QLabel):
        def handle(value: int):
            value_label.setText(f"{value:+d}" if value else "0")
            self._schedule_render()

        return handle

    def _default_color_settings(self):
        return {
            "input_profile": "auto",
            "raw_white_balance": "camera",
            "raw_colorspace": "srgb",
            "raw_lut_enabled": False,
            "raw_lut_path": "",
            "raw_auto_brightness": True,
            "raw_highlight_mode": "clip",
            "raw_demosaic": "auto",
            "working_space": "srgb",
            "scene_linear_denoise": False,
            "use_learned_denoise": True,
            "output_transform": "srgb",
            "icc_policy": "srgb",
            "wb_temp_k": wb_ops.NEUTRAL_K,
            "wb_tint": 0,
            "tone_curve": [[0.0, 0.0], [1.0, 1.0]],
            "color_mixer": cm_ops.default_color_mixer(),
        }

    def _default_preset_meta(self):
        return {
            "name": "",
            "category": "general",
            "tags": [],
            "saved_at": "",
            "kind": "preset",
            "includes": [],
        }

    _DETAIL_TEMPLATE_KEYS = (
        "clarity", "sharpness", "sharpen_radius", "sharpen_masking", "noise_red", "color_noise_red",
    )

    def _settings_differ_from_default(self, values: dict, defaults: dict, keys) -> bool:
        if not isinstance(values, dict):
            return False
        for key in keys:
            if key in values and values.get(key) != defaults.get(key):
                return True
        return False

    def _preset_include_labels(self, preset: dict) -> list[str]:
        if not isinstance(preset, dict):
            return []
        labels = []
        defaults = self._default_color_settings()
        color_settings = preset.get("color_settings", {})
        if isinstance(color_settings, dict):
            if self._settings_differ_from_default(color_settings, defaults, self._RAW_DECODE_KEYS):
                labels.append("RAW")
            if self._settings_differ_from_default(color_settings, defaults, ("working_space", "scene_linear_denoise", "use_learned_denoise")):
                labels.append("Pipeline")
            if self._settings_differ_from_default(color_settings, defaults, ("wb_temp_k", "wb_tint")):
                labels.append("WB")
            if self._settings_differ_from_default(color_settings, defaults, ("tone_curve", "color_mixer")):
                labels.append("Tone/Color")

        global_params = preset.get("global_params", {})
        if self._params_differ_from_default(global_params, "global"):
            labels.append("Adjust")
        detail_defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS.get("global", [])}
        if self._settings_differ_from_default(global_params, detail_defaults, self._DETAIL_TEMPLATE_KEYS):
            labels.append("Detail")

        selective = preset.get("selective_params", {})
        if isinstance(selective, dict) and any(
            self._params_differ_from_default(selective.get(layer, {}), layer) for layer in MASK_ORDER
        ):
            labels.append("Layers")
        if preset.get("mask_adjustments"):
            labels.append("Masks")
        return labels or ["No adjustments"]

    def _stamp_preset_meta(self, preset: dict, name: str = "") -> dict:
        meta = self._default_preset_meta()
        if isinstance(preset.get("meta"), dict):
            meta.update(preset["meta"])
        if name:
            meta["name"] = name
        meta["kind"] = "preset"
        meta["includes"] = self._preset_include_labels(preset)
        preset["meta"] = meta
        return preset

    def _preset_summary_text(self, preset: dict) -> str:
        return ", ".join(self._preset_include_labels(preset))

    def _toolbar_icon(self, name: str, primary: bool = False) -> QIcon:
        """Small Qt-rendered line icons for compact tool buttons.

        These deliberately avoid emoji so the toolbar looks the same across platforms and Qt
        builds. The shapes are simple because they render at 18-20 px inside dense controls.
        """
        pixmap = QPixmap(28, 28)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        accent = QColor("#101318" if primary else "#d7dbe0")
        muted = QColor("#5d6670" if primary else "#8b949e")
        warn = QColor("#d4a853")
        danger = QColor("#ff6b6b")
        pen = QPen(accent, 2.1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        def line(x1, y1, x2, y2):
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        def rect(x, y, w, h):
            painter.drawRoundedRect(int(x), int(y), int(w), int(h), 3, 3)

        def circle(x, y, r):
            painter.drawEllipse(QPointF(float(x), float(y)), float(r), float(r))

        if name == "open-project":
            line(5, 11, 9, 7)
            line(9, 7, 14, 7)
            rect(5, 10, 18, 12)
        elif name == "save":
            rect(6, 5, 16, 18)
            rect(9, 7, 8, 5)
            line(10, 19, 18, 19)
        elif name == "preset-open":
            rect(6, 6, 16, 16)
            line(10, 10, 18, 10)
            line(10, 14, 16, 14)
            line(13, 18, 13, 24)
            line(10, 21, 13, 24)
            line(16, 21, 13, 24)
        elif name == "preset-save":
            rect(7, 5, 14, 18)
            line(11, 9, 17, 9)
            painter.setBrush(accent)
            circle(14, 18, 2)
            painter.setBrush(Qt.NoBrush)
        elif name == "recipes":
            rect(7, 5, 14, 18)
            line(11, 9, 17, 9)
            line(11, 13, 17, 13)
            line(11, 17, 15, 17)
        elif name == "system-check":
            circle(14, 14, 8)
            painter.setPen(QPen(warn if not primary else accent, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            line(14, 9, 14, 15)
            line(14, 19, 14, 19)
        elif name == "preferences":
            circle(14, 14, 3)
            for x1, y1, x2, y2 in ((14, 4, 14, 8), (14, 20, 14, 24), (4, 14, 8, 14), (20, 14, 24, 14),
                                   (7, 7, 10, 10), (18, 18, 21, 21), (21, 7, 18, 10), (10, 18, 7, 21)):
                line(x1, y1, x2, y2)
        elif name == "batch-export":
            rect(6, 8, 16, 13)
            line(6, 12, 14, 16)
            line(22, 12, 14, 16)
            line(14, 16, 14, 21)
        elif name == "batch-jobs":
            rect(7, 5, 14, 18)
            line(10, 10, 18, 10)
            line(10, 14, 18, 14)
            line(10, 18, 15, 18)
        elif name == "retry":
            painter.drawArc(6, 6, 16, 16, 40 * 16, 280 * 16)
            painter.drawPolygon(QPolygonF([QPointF(20, 8), QPointF(23, 8), QPointF(21, 12)]))
        elif name == "export-as":
            rect(7, 8, 14, 12)
            line(14, 4, 14, 14)
            line(10, 8, 14, 4)
            line(18, 8, 14, 4)
        elif name == "quick-export":
            painter.drawPolygon(QPolygonF([QPointF(15, 4), QPointF(8, 15), QPointF(14, 15), QPointF(12, 24), QPointF(21, 11), QPointF(15, 11)]))
        elif name == "apply":
            line(7, 15, 12, 20)
            line(12, 20, 22, 8)
        elif name == "rename":
            line(8, 20, 20, 8)
            line(17, 7, 21, 11)
            line(7, 21, 12, 20)
        elif name == "delete":
            painter.setPen(QPen(danger if not primary else accent, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            line(8, 8, 20, 20)
            line(20, 8, 8, 20)
        elif name == "refresh":
            painter.drawArc(6, 6, 16, 16, 35 * 16, 285 * 16)
            painter.drawPolygon(QPolygonF([QPointF(20, 7), QPointF(23, 8), QPointF(20, 11)]))
        elif name == "collection":
            rect(5, 8, 9, 12)
            rect(14, 8, 9, 12)
            line(9, 12, 19, 12)
        elif name == "copy":
            rect(8, 8, 12, 14)
            painter.setPen(QPen(muted, 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            rect(5, 5, 12, 14)
        elif name == "paste":
            rect(8, 9, 12, 13)
            line(11, 6, 17, 6)
            line(14, 6, 14, 15)
            line(11, 12, 14, 15)
            line(17, 12, 14, 15)
        elif name == "enhance":
            circle(14, 14, 4)
            line(14, 4, 14, 8)
            line(14, 20, 14, 24)
            line(4, 14, 8, 14)
            line(20, 14, 24, 14)
        elif name == "natural":
            line(14, 22, 14, 10)
            painter.drawEllipse(8, 7, 9, 7)
            painter.drawEllipse(13, 6, 9, 8)
        elif name == "studio":
            circle(14, 12, 5)
            line(10, 18, 18, 18)
            line(8, 22, 20, 22)
        elif name == "outdoor":
            circle(14, 14, 5)
            line(14, 4, 14, 7)
            line(14, 21, 14, 24)
            line(4, 14, 7, 14)
            line(21, 14, 24, 14)
        elif name == "group":
            circle(10, 11, 3)
            circle(18, 11, 3)
            circle(14, 17, 3)
            line(7, 22, 21, 22)
        else:
            circle(14, 14, 7)

        painter.end()
        return QIcon(pixmap)

    def _make_icon_button(self, icon: str, title: str, *detail_lines: str, callback=None, primary: bool = False) -> QPushButton:
        """Compact icon-only button used by the topbar and left sidebar. The label moves into
        a rich-text tooltip (bold title + one line per supporting detail) instead of being
        cramped into the button itself -- a plain single-line tooltip wasn't scannable once
        several of these sit in a row."""
        button = QPushButton("", self)
        button.setFixedWidth(34)
        button.setIcon(self._toolbar_icon(icon, primary=primary))
        button.setIconSize(QSize(22, 22))
        lines = "".join(f"<br>{line}" for line in detail_lines)
        button.setToolTip(f"<b>{title}</b>{lines}")
        if primary:
            button.setObjectName("PrimaryButton")
        if callback is not None:
            button.clicked.connect(callback)
        return button

    def _build_recipe_strip(self) -> QWidget:
        """Always-visible row above the canvas so trying a look is one click, not a detour
        through the buried Recipes modal -- each button applies live via the same Keep/Discard
        preview flow the preset browser uses (_apply_preset_with_preview), so a recipe that
        doesn't suit this photo is one click to back out of instead of relying on Ctrl+Z."""
        strip = QWidget(self)
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        recipes = self._guided_recipes()
        # Icon-only -- a text label per recipe ("Group Background Pop"...) ate too much of the
        # topbar's width. The full name/description still surface as a tooltip; this is the
        # same tooltip-carries-the-explanation pattern as the panel-toggle glyph buttons.
        recipe_icons = {
            "Natural Portrait": "natural",
            "Studio Clean": "studio",
            "Outdoor Warm": "outdoor",
            "Group Background Pop": "group",
        }
        back_out_hint = "<i>Keep or Discard appears after, so it's one click to back out.</i>"

        if recipes:
            enhance_btn = self._make_icon_button(
                "enhance", "Enhance",
                f"One-click starting point -- applies \"{recipes[0]['name']}.\"",
                recipes[0]["description"],
                back_out_hint,
                callback=lambda _checked=False, r=recipes[0]: self._apply_recipe_with_preview(r),
                primary=True,
            )
        else:
            enhance_btn = self._make_icon_button("enhance", "Enhance", primary=True)
        layout.addWidget(enhance_btn)

        for recipe in recipes:
            name = recipe.get("name", "Recipe")
            btn = self._make_icon_button(
                recipe_icons.get(name, "preset-open"), name,
                recipe.get("description", ""),
                back_out_hint,
                callback=lambda _checked=False, r=recipe: self._apply_recipe_with_preview(r),
            )
            layout.addWidget(btn)
        browse_btn = self._make_icon_button(
            "recipes", "Recipes...",
            "Browse all recipes with full descriptions before applying.",
            callback=self.open_recipe_dialog,
        )
        layout.addWidget(browse_btn)
        layout.addStretch(1)
        return strip

    def _guided_recipes(self):
        return [
            {
                "name": "Natural Portrait",
                "description": "Balanced portrait cleanup with restrained skin work and subtle eye emphasis.",
                "highlights": [
                    "Gentle exposure and vibrance lift",
                    "Moderate skin smoothing and blemish cleanup",
                    "Subtle eye whitening and sharpening",
                ],
                "preset": {
                    "global_params": {"exposure": 10, "highlights": -8, "shadows": 10, "vibrance": 12, "clarity": 6},
                    "selective_params": {
                        "face": {"exposure": 8, "smooth": 10},
                        "skin": {"smooth": 24, "blemish": 20, "warmth": 6},
                        "eyes": {"brightness": 8, "whites": 10, "sharpen": 12},
                        "lips": {"saturation": 6},
                        "hair": {"shine": 8},
                    },
                },
            },
            {
                "name": "Studio Clean",
                "description": "Polished studio finish with stronger face cleanup and tighter contrast control.",
                "highlights": [
                    "Highlight control and cleaner skin finish",
                    "More structure in face and hair",
                    "Good base for formal portraits",
                ],
                "preset": {
                    "global_params": {"exposure": 6, "highlights": -18, "shadows": 8, "clarity": 10, "sharpness": 10},
                    "selective_params": {
                        "face": {"exposure": 10, "highlights": -10, "smooth": 16, "refine": 18, "refine_fidelity": 72},
                        "skin": {"smooth": 34, "blemish": 28, "clarity": -8},
                        "eyes": {"brightness": 10, "whites": 14, "sharpen": 16, "iris_pop": 8},
                        "hair": {"shine": 12, "clarity": 10},
                    },
                },
            },
            {
                "name": "Outdoor Warm",
                "description": "Warm outdoor look with skin-friendly color balance and slightly richer background tone.",
                "highlights": [
                    "Warmer global balance",
                    "Skin warmth without heavy smoothing",
                    "Background saturation and dehaze lift",
                ],
                "preset": {
                    "global_params": {"temperature": 10, "tint": 4, "vibrance": 16, "saturation": 4},
                    "selective_params": {
                        "subjects": {"exposure": 8, "warmth": 10, "saturation": 8},
                        "background": {"saturation": 10, "warmth": 8, "dehaze": 10},
                        "skin": {"smooth": 18, "warmth": 10},
                        "hair": {"warmth": 8, "shine": 8},
                    },
                },
            },
            {
                "name": "Group Background Pop",
                "description": "Built for group photos: subjects lifted slightly, background softened and separated.",
                "highlights": [
                    "Subject lift for all people",
                    "Background blur and dehaze",
                    "Good starting point for event and family shots",
                ],
                "preset": {
                    "global_params": {"exposure": 4, "shadows": 8},
                    "selective_params": {
                        "subjects": {"exposure": 12, "clarity": 6, "saturation": 6},
                        "background": {"blur": 28, "exposure": -6, "dehaze": 14, "saturation": -4},
                        "face": {"exposure": 6},
                        "eyes": {"brightness": 6, "whites": 8},
                    },
                },
            },
        ]

    def _normalize_preset_meta(self, meta, path=""):
        normalized = self._default_preset_meta()
        if isinstance(meta, dict):
            normalized.update(meta)
        normalized["name"] = str(normalized.get("name") or "").strip() or Path(path).stem or "Untitled"
        normalized["category"] = str(normalized.get("category") or "general").strip().lower() or "general"
        tags = normalized.get("tags", [])
        if isinstance(tags, str):
            tags = [part.strip() for part in tags.split(",") if part.strip()]
        normalized["tags"] = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        normalized["saved_at"] = str(normalized.get("saved_at") or "")
        includes = normalized.get("includes", [])
        if isinstance(includes, str):
            includes = [part.strip() for part in includes.split(",") if part.strip()]
        normalized["includes"] = [str(item).strip() for item in includes if str(item).strip()]
        # "shoot_template" is the legacy kind value from before the Preset/Template rename;
        # normalize it to "preset" so older .pepreset files read consistently going forward.
        kind = str(normalized.get("kind") or "preset")
        normalized["kind"] = "preset" if kind == "shoot_template" else kind
        return normalized

    @staticmethod
    def _humanize_reason(reason: str) -> str:
        text = str(reason or "").strip()
        low = text.lower()
        # Genuine missing-package errors carry the ModuleNotFoundError text or the
        # "<pkg> unavailable: ..." prefix set when the import itself fails.
        if "no module named 'onnxruntime'" in low or low.startswith("onnxruntime unavailable"):
            return "ONNX Runtime is not installed — advanced portrait masks use fallback mode."
        if "no module named 'mediapipe'" in low or "mediapipe tasks unavailable" in low or "mediapipe unavailable" in low:
            return "MediaPipe is not installed — using fallback mode."
        if "no module named" in low:
            return "A required Python package is missing — using fallback mode."
        # A model that fails to *load* (bad/incompatible ONNX, CoreML error) is NOT a missing
        # runtime -- surface the real error rather than mislabeling it as "not installed".
        if "init failed" in low or "runtime error" in low or "onnxruntimeerror" in low:
            return f"Model failed to load: {text}"
        if any(token in low for token in ("not found", "missing", "no model", "does not exist")):
            return "Model file is not installed — feature uses fallback or is disabled."
        return text or "Unavailable."

    def _readiness_items(self):
        """Structured per-component readiness for the System Check panel.

        Returns (items, has_issues, models_dir). Each item is a dict with
        name / status (ready|fallback|off) / message / detail (raw reason).
        """
        models_dir = Path(__file__).resolve().parents[2] / "models"

        parser = getattr(self.segmenter, "_model", None)
        detector = getattr(getattr(self.segmenter, "_heuristic", None), "_face_detector", None)
        background_removal = getattr(self.segmenter, "_background_removal", None)
        matting = getattr(self.segmenter, "_matting", None)
        subject_segmenter = getattr(self.segmenter, "_subjects", None)
        facial_hair_segmenter = getattr(self.segmenter, "_facial_hair", None)
        refiner = get_face_refiner()
        refiner._ensure_session()

        def reason_of(obj):
            return str(getattr(obj, "reason_unavailable", "") or "") if obj is not None else ""

        def available_of(obj):
            return bool(getattr(obj, "available", False))

        # name, component-reason, degraded-status when unavailable, ready description
        specs = [
            ("Face parsing", reason_of(parser), "fallback",
             "Segments facial regions so Skin / Eyes / Lips / Hair edits stay local."),
            ("Face detector", reason_of(detector), "fallback",
             "Finds faces to target for selective edits."),
        ]

        items = []
        for name, reason, degraded, description in specs:
            if reason:
                items.append({"name": name, "status": degraded,
                              "message": self._humanize_reason(reason), "detail": reason})
            else:
                items.append({"name": name, "status": "ready", "message": description, "detail": ""})

        # Subject selection is "ready" if any learned subject/foreground backend can run
        # (RMBG/BiRefNet preferred, then MODNet, then MediaPipe); only the face-derived
        # heuristic counts as fallback. Name the active backend so System Check matches export.
        if available_of(background_removal) or available_of(matting) or available_of(subject_segmenter):
            if available_of(background_removal):
                active = "RMBG-2.0 foreground matting"
            elif available_of(matting):
                active = "MODNet matting"
            else:
                active = "MediaPipe selfie segmenter"
            items.append({"name": "Subject selection", "status": "ready",
                          "message": f"Separates subject from background for Subjects / Background layers ({active}).",
                          "detail": ""})
        else:
            reason = reason_of(subject_segmenter)
            items.append({"name": "Subject selection", "status": "fallback",
                          "message": self._humanize_reason(reason or "Using the face-derived heuristic mask."),
                          "detail": reason})

        if available_of(background_removal):
            device = str(getattr(background_removal, "execution_provider", "cpu") or "cpu")
            items.append({"name": "Subject/background matting (RMBG-2.0)", "status": "ready",
                          "message": f"Accuracy-first foreground alpha for Subjects / Background ({device}).",
                          "detail": ""})
        else:
            items.append({"name": "Subject/background matting (RMBG-2.0)", "status": "off",
                          "message": self._humanize_reason(reason_of(background_removal) or "Model folder is not installed."),
                          "detail": reason_of(background_removal)})

        # MODNet is an optional quality upgrade -- "off" (not "fallback") when absent, since
        # the selfie segmenter still produces usable masks without it.
        if available_of(matting):
            items.append({"name": "Subject matting (MODNet)", "status": "ready",
                          "message": "High-quality alpha matting recovers hair detail and full-body edges.",
                          "detail": ""})
        else:
            items.append({"name": "Subject matting (MODNet)", "status": "off",
                          "message": self._humanize_reason(reason_of(matting) or "Model file is not installed."),
                          "detail": reason_of(matting)})

        # SAM gives the Person layer learned per-person instance masks -- "off" (not "fallback")
        # when absent, since the watershed split still produces a Person mask without it.
        person_instances = getattr(self.segmenter, "_person_instances", None)
        if available_of(person_instances):
            device = str(getattr(person_instances, "execution_provider", "cpu") or "cpu")
            items.append({"name": "Person identity masks (Mask DINO)", "status": "ready",
                          "message": f"Accuracy-first person instance masks assign each face to a detected person ({device}).",
                          "detail": ""})
        else:
            items.append({"name": "Person identity masks (Mask DINO)", "status": "off",
                          "message": self._humanize_reason(reason_of(person_instances) or "Model files are not installed; using SAM or watershed."),
                          "detail": reason_of(person_instances)})

        # SAM gives the Person layer prompted per-person masks -- "off" (not "fallback") when
        # absent, since Mask DINO or the watershed split can still produce a Person mask without it.
        instance = getattr(self.segmenter, "_instance", None)
        sam_face_disabled = os.getenv("PORTRAIT_DISABLE_SAM_FACE", "").strip() not in ("", "0", "false", "False", "no", "No")
        if available_of(instance):
            items.append({"name": "Person instance masks (SAM)", "status": "ready",
                          "message": "Prompted SAM masks refine/fallback per-person selection when Mask DINO is unavailable.",
                          "detail": ""})
        else:
            items.append({"name": "Person instance masks (SAM)", "status": "off",
                          "message": self._humanize_reason(reason_of(instance) or "Model files are not installed; using the watershed split."),
                          "detail": reason_of(instance)})
        if available_of(instance) and not sam_face_disabled:
            items.append({"name": "Face boundary refinement (SAM)", "status": "ready",
                          "message": "Uses SAM to tighten the active Face boundary while preserving semantic Skin / Eyes / Lips / Hair masks.",
                          "detail": ""})
        else:
            reason = "disabled via PORTRAIT_DISABLE_SAM_FACE" if sam_face_disabled else reason_of(instance)
            items.append({"name": "Face boundary refinement (SAM)", "status": "off",
                          "message": self._humanize_reason(reason or "Model files are not installed; using face parsing or heuristic masks."),
                          "detail": reason})

        # Auto WB (AI) is an optional upgrade -- "off" when absent, since the classical
        # Gray World / White Patch / picker still handle white balance without it.
        wb_estimator = getattr(self, "_wb_estimator", None)
        if available_of(wb_estimator):
            items.append({"name": "Auto WB (AI)", "status": "ready",
                          "message": "Learned auto white balance (robust to dominant colors / casts).",
                          "detail": ""})
        else:
            items.append({"name": "Auto WB (AI)", "status": "off",
                          "message": self._humanize_reason(reason_of(wb_estimator) or "Model file is not installed."),
                          "detail": reason_of(wb_estimator)})

        # ML denoise is an optional upgrade -- "off" when absent, since the bilateral
        # filter still handles noise reduction without it.
        denoiser = getattr(self, "_denoiser", None)
        if available_of(denoiser):
            items.append({"name": "ML denoise", "status": "ready",
                          "message": "Learned noise reduction (preserves edges/detail better than bilateral).",
                          "detail": ""})
        else:
            items.append({"name": "ML denoise", "status": "off",
                          "message": self._humanize_reason(reason_of(denoiser) or "Model file is not installed."),
                          "detail": reason_of(denoiser)})

        # Deep Denoise (NAFNet, real-noise) is an opt-in heavy one-shot -- "off" when absent.
        deep = getattr(self, "_deep_denoiser", None)
        if available_of(deep):
            items.append({"name": "Deep Denoise", "status": "ready",
                          "message": "Heavy real-noise AI denoise (Deep Denoise button): live proxy preview in seconds, full-res applied at export.",
                          "detail": ""})
        else:
            items.append({"name": "Deep Denoise", "status": "off",
                          "message": self._humanize_reason(reason_of(deep) or "Model file is not installed."),
                          "detail": reason_of(deep)})

        # Facial hair is an optional model that refines skin smoothing.
        fh_reason = reason_of(facial_hair_segmenter)
        if fh_reason:
            items.append({"name": "Facial hair", "status": "off",
                          "message": self._humanize_reason(fh_reason), "detail": fh_reason})
        else:
            items.append({"name": "Facial hair", "status": "ready",
                          "message": "Optional model that excludes facial hair from skin smoothing.", "detail": ""})

        # Refiner exposes .available in addition to a reason string.
        refiner_reason = reason_of(refiner)
        refiner_available = bool(getattr(refiner, "available", False)) and not refiner_reason
        if refiner_available:
            items.append({"name": "Face refiner", "status": "ready",
                          "message": "CodeFormer face restoration is available.", "detail": ""})
        else:
            items.append({"name": "Face refiner", "status": "off",
                          "message": self._humanize_reason(refiner_reason or "Model file is not installed."),
                          "detail": refiner_reason})

        has_issues = any(item["status"] != "ready" for item in items)
        return items, has_issues, models_dir

    def _browser_state_path(self):
        return self._preset_library_dir() / self.BROWSER_STATE_FILE

    def _write_browser_state(self, payload):
        try:
            with open(self._browser_state_path(), "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:
            return

    def _load_browser_state(self):
        payload = self._browser_state_payload()
        if not payload:
            return
        recent = payload.get("recent", [])
        if isinstance(recent, list):
            self._recent_preset_paths = [str(Path(p)) for p in recent if p]
        category = str(payload.get("category", "all") or "all")
        self._preset_category_filter = category
        section_state = payload.get("section_state", {})
        try:
            layout_version = int(payload.get("inspector_layout_version", 0) or 0)
        except (TypeError, ValueError):
            layout_version = 0
        if isinstance(section_state, dict) and layout_version == self.INSPECTOR_LAYOUT_VERSION:
            self._section_state.update({str(key): bool(value) for key, value in section_state.items()})
        for title, expanded in self._section_state.items():
            self._set_section_expanded(title, expanded, save=False)
        if hasattr(self, "preset_category_combo"):
            idx = self.preset_category_combo.findText(category)
            if idx >= 0:
                self.preset_category_combo.setCurrentIndex(idx)

    def _save_browser_state(self):
        payload = self._browser_state_payload()
        payload.update(
            {
                "recent": list(self._recent_preset_paths[:8]),
                "category": str(self._preset_category_filter),
                "inspector_layout_version": self.INSPECTOR_LAYOUT_VERSION,
                "section_state": dict(self._section_state),
                "readiness_seen": True,
            }
        )
        self._write_browser_state(payload)

    def _collections_dir(self):
        path = Path(os.getcwd()) / "collections"
        path.mkdir(exist_ok=True)
        return path

    def _collections_state_path(self):
        return self._collections_dir() / "collections.json"

    def _thumbnail_cache_dir(self):
        path = self._collections_dir() / "thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _thumbnail_cache_path(self, img_path: str):
        try:
            stat = os.stat(img_path)
            sig = f"{img_path}:{stat.st_size}:{int(stat.st_mtime)}"
        except OSError:
            sig = img_path
        key = hashlib.sha1(sig.encode("utf-8")).hexdigest()
        return self._thumbnail_cache_dir() / f"{key}.jpg"

    def _rendered_preview_dir(self):
        path = self._collections_dir() / "rendered_previews"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _rendered_preview_path(self, key: str):
        return self._rendered_preview_dir() / f"{key}.jpg"

    def _load_collections_state(self):
        path = self._collections_state_path()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception:
                payload = {}
            collections = payload.get("collections", {})
            if isinstance(collections, dict):
                cleaned = {}
                for name, data in collections.items():
                    if not isinstance(data, dict):
                        continue
                    images = [p for p in data.get("images", []) if isinstance(p, str)]
                    overrides = data.get("image_overrides", {})
                    quality = data.get("image_quality", {})
                    culled = data.get("culled_images", [])
                    rendered_previews = data.get("rendered_previews", {})
                    cleaned[str(name)] = {
                        "images": images,
                        "created_at": data.get("created_at", ""),
                        "source_folder": data.get("source_folder", ""),
                        "image_overrides": overrides if isinstance(overrides, dict) else {},
                        "image_quality": quality if isinstance(quality, dict) else {},
                        "culled_images": [p for p in culled if isinstance(p, str)] if isinstance(culled, list) else [],
                        "rendered_previews": rendered_previews if isinstance(rendered_previews, dict) else {},
                    }
                self._collections = cleaned
            active = payload.get("active_collection")
            if active and active in self._collections:
                self._active_collection = active
        self._refresh_collection_combo()
        self._apply_active_collection()
        if self._active_collection:
            self._check_collection_missing_files(self._active_collection)

    def _save_collections_state(self):
        payload = {
            "active_collection": self._active_collection,
            "collections": self._collections,
        }
        try:
            with open(self._collections_state_path(), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as ex:
            sys.stderr.write(f"Failed to save collections state: {ex}\n")
            if hasattr(self, "statusBar"):
                self.statusBar().showMessage(
                    f"Could not save collection changes: {ex} -- your recent edits may not persist", 8000
                )

    def _preferences_path(self):
        return self._collections_dir() / "preferences.json"

    def _load_preferences(self) -> dict:
        path = self._preferences_path()
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save_preferences(self):
        try:
            with open(self._preferences_path(), "w", encoding="utf-8") as fh:
                json.dump(self._preferences, fh, indent=2)
        except Exception as ex:
            sys.stderr.write(f"Failed to save preferences: {ex}\n")
            self.statusBar().showMessage(f"Could not save preferences: {ex}", 8000)

    def show_preferences(self):
        dialog = PreferencesDialog(self, dict(self._preferences))
        if dialog.exec() == QDialog.Accepted:
            self._preferences = dialog.values()
            self._runtime_settings["acceleration_mode"] = self._preferences.get("acceleration_mode", "auto")
            self._save_preferences()
            self.statusBar().showMessage("Preferences saved")

    def _collection_stats_text(self, name: str) -> str:
        """One-line at-a-glance stats for a collection: total / edited / culled counts."""
        collection = self._collections.get(name) or {}
        total = len(collection.get("images", []))
        edited = len(collection.get("image_overrides", {}))
        culled = len(collection.get("culled_images", []))
        parts = [f"{total} image{'s' if total != 1 else ''}"]
        if edited:
            parts.append(f"{edited} edited")
        if culled:
            parts.append(f"{culled} culled")
        return " · ".join(parts)

    def _refresh_collection_combo(self):
        if not hasattr(self, "collection_combo"):
            return
        self.collection_combo.blockSignals(True)
        self.collection_combo.clear()
        self.collection_combo.addItem("No collection", None)
        for name in sorted(self._collections.keys(), key=str.lower):
            count = len(self._collections[name].get("images", []))
            self.collection_combo.addItem(f"{name} ({count})", name)
            self.collection_combo.setItemData(
                self.collection_combo.count() - 1, f"{name}\n{self._collection_stats_text(name)}", Qt.ToolTipRole
            )
        if self._active_collection and self._active_collection in self._collections:
            self.collection_combo.setToolTip(
                f"{self._active_collection}\n{self._collection_stats_text(self._active_collection)}"
            )
        else:
            self.collection_combo.setToolTip("Select an imported collection to show in the filmstrip")
        if self._active_collection and self._active_collection in self._collections:
            idx = self.collection_combo.findData(self._active_collection)
            if idx >= 0:
                self.collection_combo.setCurrentIndex(idx)
        else:
            self.collection_combo.setCurrentIndex(0)
        self.collection_combo.blockSignals(False)
        has_active = bool(self._active_collection and self._active_collection in self._collections)
        if hasattr(self, "delete_collection_btn"):
            self.delete_collection_btn.setEnabled(has_active)
        if hasattr(self, "rename_collection_btn"):
            self.rename_collection_btn.setEnabled(has_active)
        if hasattr(self, "apply_settings_collection_btn"):
            self.apply_settings_collection_btn.setEnabled(has_active)
        if hasattr(self, "export_collection_btn"):
            self.export_collection_btn.setEnabled(has_active)
        if hasattr(self, "export_selected_collection_btn"):
            self.export_selected_collection_btn.setEnabled(has_active)
        if hasattr(self, "quick_export_collection_btn"):
            self.quick_export_collection_btn.setEnabled(has_active)
        if hasattr(self, "review_cull_btn"):
            self.review_cull_btn.setEnabled(has_active)
        if hasattr(self, "view_culled_btn"):
            self.view_culled_btn.setEnabled(has_active)
        self._update_filmstrip_filter_buttons()

    def _on_collection_combo_changed(self, index: int):
        name = self.collection_combo.itemData(index)
        self._active_collection = name
        self._save_collections_state()
        self._apply_active_collection()
        has_active = bool(name)
        if hasattr(self, "delete_collection_btn"):
            self.delete_collection_btn.setEnabled(has_active)
        if hasattr(self, "rename_collection_btn"):
            self.rename_collection_btn.setEnabled(has_active)
        if hasattr(self, "apply_settings_collection_btn"):
            self.apply_settings_collection_btn.setEnabled(has_active)
        if hasattr(self, "export_collection_btn"):
            self.export_collection_btn.setEnabled(has_active)
        if hasattr(self, "export_selected_collection_btn"):
            self.export_selected_collection_btn.setEnabled(has_active)
        if hasattr(self, "quick_export_collection_btn"):
            self.quick_export_collection_btn.setEnabled(has_active)
        if hasattr(self, "review_cull_btn"):
            self.review_cull_btn.setEnabled(has_active)
        if hasattr(self, "view_culled_btn"):
            self.view_culled_btn.setEnabled(has_active)
        self._update_filmstrip_filter_buttons()
        if name:
            self._check_collection_missing_files(name)

    def _set_filmstrip_filter(self, value: str):
        value = str(value or "all")
        if value not in {"all", "edited", "unedited", "raw", "failed"}:
            value = "all"
        if value == "failed" and not self._filmstrip_failed_paths:
            self.statusBar().showMessage("No failed export images are currently tracked")
        self._filmstrip_filter = value
        self._update_filmstrip_filter_buttons()
        self._populate_filmstrip()

    def _update_filmstrip_filter_buttons(self):
        buttons = getattr(self, "_filmstrip_filter_buttons", {})
        if not buttons:
            return
        has_collection = bool(self._active_collection and self._active_collection in self._collections)
        for key, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(key == self._filmstrip_filter)
            button.blockSignals(False)
            button.setEnabled(has_collection and (key != "failed" or bool(self._filmstrip_failed_paths)))

    def _filmstrip_filter_matches(self, path: str, overrides: dict, culled_paths: set[str]) -> bool:
        mode = str(getattr(self, "_filmstrip_filter", "all") or "all")
        if mode == "edited":
            return path in overrides
        if mode == "unedited":
            return path not in overrides
        if mode == "raw":
            return is_raw_path(path)
        if mode == "failed":
            return os.path.abspath(str(path)) in self._filmstrip_failed_paths
        return True

    def _update_filmstrip_count_label(self, shown: int | None = None, total: int | None = None):
        label = getattr(self, "filmstrip_count_label", None)
        if label is None:
            return
        selected = len(getattr(self, "_filmstrip_selected_paths", set()))
        if shown is None:
            shown = len(getattr(self, "_filmstrip_items", {}))
        if total is None:
            total = len(getattr(self, "_filmstrip_images", []))
        parts = [f"{shown}/{total}"]
        if selected:
            parts.append(f"{selected} selected")
        mode = str(getattr(self, "_filmstrip_filter", "all") or "all")
        if mode != "all":
            parts.append(mode)
        label.setText(" | ".join(parts))

    def _check_collection_missing_files(self, name):
        """On an explicit collection switch, scan once for images that moved/were deleted/are
        on a disconnected drive since import -- previously this only surfaced one file at a
        time, as a bare 'file not found' error when you happened to click that thumbnail."""
        collection = self._collections.get(name)
        if not collection:
            return
        missing = [p for p in collection.get("images", []) if not os.path.isfile(p)]
        if not missing:
            return
        reply = QMessageBox.question(
            self,
            "Missing Images",
            f"{len(missing)} image(s) in \"{name}\" can no longer be found on disk (moved, "
            "renamed, deleted, or on a disconnected drive). Remove them from the collection "
            "now? Their saved edits will be discarded too -- the original files, if they still "
            "exist somewhere, are not touched.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        imgs = collection.get("images", [])
        overrides = collection.get("image_overrides", {})
        for p in missing:
            if p in imgs:
                imgs.remove(p)
            overrides.pop(p, None)
        self._save_collections_state()
        self._apply_active_collection()
        self.statusBar().showMessage(f"Removed {len(missing)} missing image reference(s) from \"{name}\"")

    def _apply_active_collection(self):
        if self._active_collection and self._active_collection in self._collections:
            self._imported_images = list(self._collections[self._active_collection].get("images", []))
        else:
            self._imported_images = []
        self._populate_filmstrip()

    def _rename_active_collection(self):
        old_name = self._active_collection
        if not old_name or old_name not in self._collections:
            return
        new_name, ok = QInputDialog.getText(self, "Rename Collection", "New name:", text=old_name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return
        if new_name in self._collections:
            QMessageBox.warning(self, "Rename Collection", f"A collection named \"{new_name}\" already exists.")
            return
        self._collections[new_name] = self._collections.pop(old_name)
        self._active_collection = new_name
        self._save_collections_state()
        self._refresh_collection_combo()
        self.statusBar().showMessage(f"Renamed collection to \"{new_name}\"")

    def _delete_active_collection(self):
        name = self._active_collection
        if not name or name not in self._collections:
            return
        reply = QMessageBox.question(
            self,
            "Delete Collection",
            f"Remove the collection \"{name}\"? This only removes it from the app; image files on disk are not affected.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._collections.pop(name, None)
        self._active_collection = None
        self._save_collections_state()
        self._refresh_collection_combo()
        self._apply_active_collection()
        self.statusBar().showMessage(f"Deleted collection \"{name}\"")

    def _get_collection_image_override(self, path: str):
        """Return the saved non-destructive settings override for an image, if any."""
        if not self._active_collection:
            return None
        collection = self._collections.get(self._active_collection)
        if not collection:
            return None
        return collection.get("image_overrides", {}).get(path)

    def _apply_settings_to_collection(self):
        """Apply the copied settings to every image in the active collection (non-destructive)."""
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        images = collection.get("images", [])
        if not images:
            QMessageBox.information(self, "Empty Collection", "This collection has no images.")
            return
        if self._settings_clipboard is None:
            if self.preview_array is None:
                QMessageBox.information(
                    self,
                    "No Settings",
                    "Open an image and use Copy Settings (Ctrl+Alt+C) first, then apply to a collection.",
                )
                return
            self._copy_settings()
        if self._settings_clipboard is None:
            return
        reply = QMessageBox.question(
            self,
            "Apply Settings to Collection",
            f"Apply the copied settings to all {len(images)} image(s) in \"{self._active_collection}\"?\n\n"
            "This updates each image's saved settings non-destructively — opening any image in this "
            "collection will show these settings applied. No files are exported.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        overrides = collection.setdefault("image_overrides", {})
        for path in images:
            overrides[path] = copy.deepcopy(self._settings_clipboard)
            thumb = self._filmstrip_items.get(path)
            if thumb is not None:
                thumb.set_has_override(True)
                state, name = self._preset_state_for_override(overrides[path])
                thumb.set_preset_state(state, name)
        self._save_collections_state()
        if self.file_path in images:
            self._begin_document_change()
            raw_decode_changed = self._apply_settings_payload(self._settings_clipboard)
            self._push_document_history()
            if raw_decode_changed:
                self._reload_source_for_decode_settings()
            else:
                self._schedule_render()
        self.statusBar().showMessage(
            f"Applied settings to {len(images)} image(s) in \"{self._active_collection}\" (non-destructive)"
        )

    def _preset_to_clipboard_payload(self, preset: dict, preset_name: str = "", preset_path: str | Path | None = None) -> dict:
        """Reshape a flat preset (one recipe, no per-face concept) into the settings-clipboard
        shape collection overrides expect, targeting the primary (first) face -- matching the
        "face 0 is primary" convention batch export already uses for preset-based jobs."""
        payload = {
            "global_params": dict(preset.get("global_params", {})),
            "color_settings": dict(preset.get("color_settings", {})),
            "layer_options": copy.deepcopy(preset.get("layer_options", {})),
            "layer_order": list(preset.get("layer_order", [])),
            "mask_adjustments": copy.deepcopy(preset.get("mask_adjustments", {})),
            "selective_by_face": [{"face_index": 0, "params": copy.deepcopy(preset.get("selective_params", {}))}],
        }
        if preset_name or preset_path:
            payload["_preset"] = self._preset_payload_meta(payload, preset_name or Path(str(preset_path)).stem, preset_path, preset)
        return payload

    def _apply_browser_preset_to_collection(self):
        """Apply a preset straight from the browser to every image in the active collection --
        previously this required a copy-then-apply detour through the settings clipboard."""
        entry = self._selected_browser_entry()
        if entry is None:
            QMessageBox.information(self, "Preset Browser", "Select a preset from the library first.")
            return
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        images = collection.get("images", [])
        if not images:
            QMessageBox.information(self, "Empty Collection", "This collection has no images.")
            return
        try:
            preset = entry.get("preset") or self._load_preset_file(entry["path"])
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))
            return
        self._apply_preset_payload_to_paths(preset, entry["path"].name, entry["path"], images, scope_label="collection")

    def _apply_preset_payload_to_paths(self, preset: dict, name: str, path: str | Path | None, paths: list[str], scope_label: str = "selection"):
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        collection_paths = set(collection.get("images", []))
        paths = [p for p in paths if p in collection_paths]
        if not paths:
            QMessageBox.information(self, "Preset", "No matching collection images for this apply scope.")
            return
        reply = QMessageBox.question(
            self,
            "Apply Preset",
            f"Apply preset \"{name}\" to {len(paths)} {scope_label} image(s) in "
            f"\"{self._active_collection}\"?\n\nThis updates saved RAW decode, "
            "white balance, Detail, and portrait settings non-destructively. No files are exported.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        payload = self._preset_to_clipboard_payload(preset, preset_name=name, preset_path=path)
        overrides = collection.setdefault("image_overrides", {})
        for image_path in paths:
            overrides[image_path] = copy.deepcopy(payload)
            thumb = self._filmstrip_items.get(image_path)
            if thumb is not None:
                thumb.set_has_override(True)
                state, preset_name = self._preset_state_for_override(overrides[image_path])
                thumb.set_preset_state(state, preset_name)
        self._save_collections_state()
        if self.file_path in paths:
            self._begin_document_change()
            raw_decode_changed = self._apply_settings_payload(payload)
            self._push_document_history()
            if raw_decode_changed:
                self._reload_source_for_decode_settings()
            else:
                self._schedule_render()
        if path is not None:
            self._remember_recent_preset(str(path))
        self.statusBar().showMessage(
            f"Applied preset \"{name}\" to {len(paths)} image(s) in \"{self._active_collection}\""
        )

    def _export_collection(self):
        """Batch export every image in the active collection using each image's own saved
        per-image settings (collection overrides) -- no preset required. Images with no saved
        override export using plain default settings (i.e. an unedited format conversion)."""
        images, overrides = self._active_collection_images_or_warn()
        if images is None:
            return

        last_options = self._last_batch_options()
        dialog = CollectionExportDialog(
            self,
            output_dir=last_options.get("output_dir", "") or self._preferences.get("default_export_folder", ""),
            suffix=last_options.get("suffix", "_enhanced"),
            output_format=last_options.get("format", "jpeg"),
            skip_completed=bool(last_options.get("skip_completed", True)),
            image_count=len(images),
            overridden_count=sum(1 for p in images if p in overrides),
            quality=last_options.get("quality", "best"),
            output_sharpening=last_options.get("output_sharpening", "standard"),
            deep_denoise=bool(last_options.get("deep_denoise", False)),
        )
        if dialog.exec() != QDialog.Accepted:
            return

        output_dir = dialog.output_dir()
        suffix = dialog.suffix()
        output_format = dialog.output_format()
        skip_completed = dialog.skip_completed()
        quality = dialog.quality()
        disable_segmentation = dialog.disable_segmentation()
        output_sharpening = dialog.output_sharpening()
        deep_denoise = dialog.deep_denoise()

        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Export Collection", "Select a valid output folder.")
            return
        if output_format not in self.BATCH_OUTPUT_FORMATS:
            QMessageBox.warning(self, "Export Collection", "Select a valid output format.")
            return
        if not self._confirm_collection_overwrite(images, output_dir, suffix, output_format, skip_completed):
            return

        self._save_last_batch_options(
            {
                **last_options,
                "output_dir": output_dir,
                "suffix": suffix,
                "format": output_format,
                "skip_completed": skip_completed,
                "quality": quality,
                "disable_segmentation": disable_segmentation,
                "output_sharpening": output_sharpening,
                "deep_denoise": deep_denoise,
            }
        )
        self._dispatch_collection_export_job(
            images, overrides, output_dir, suffix, output_format, skip_completed, quality, disable_segmentation,
            output_sharpening, deep_denoise,
        )

    def _quick_export_collection(self):
        """Zero-dialog collection export: reuses whatever destination/format/quality/suffix
        were used last time (CollectionExportDialog or this). Falls back to the full dialog if
        there's no remembered destination yet -- guessing a folder for a potentially large batch
        write is the wrong place to save a click."""
        images, overrides = self._active_collection_images_or_warn()
        if images is None:
            return
        last_options = self._last_batch_options()
        output_dir = last_options.get("output_dir", "") or self._preferences.get("default_export_folder", "")
        if not output_dir or not os.path.isdir(output_dir):
            self._export_collection()
            return
        suffix = last_options.get("suffix", "_enhanced")
        output_format = last_options.get("format", "jpeg")
        if output_format not in self.BATCH_OUTPUT_FORMATS:
            output_format = "jpeg"
        skip_completed = bool(last_options.get("skip_completed", True))
        quality = last_options.get("quality", "best")
        disable_segmentation = bool(last_options.get("disable_segmentation", False))
        output_sharpening = last_options.get("output_sharpening", "standard")
        deep_denoise = bool(last_options.get("deep_denoise", False))
        if not self._confirm_collection_overwrite(images, output_dir, suffix, output_format, skip_completed):
            return
        self._dispatch_collection_export_job(
            images, overrides, output_dir, suffix, output_format, skip_completed, quality, disable_segmentation,
            output_sharpening, deep_denoise,
        )

    def _export_selected_collection(self):
        images, overrides = self._active_collection_images_or_warn()
        if images is None:
            return
        selected = [p for p in self._selected_filmstrip_paths() if p in set(images)]
        if not selected:
            QMessageBox.information(self, "Export Selected", "Select one or more filmstrip images first.")
            return

        last_options = self._last_batch_options()
        dialog = CollectionExportDialog(
            self,
            output_dir=last_options.get("output_dir", "") or self._preferences.get("default_export_folder", ""),
            suffix=last_options.get("suffix", "_enhanced"),
            output_format=last_options.get("format", "jpeg"),
            skip_completed=bool(last_options.get("skip_completed", True)),
            image_count=len(selected),
            overridden_count=sum(1 for p in selected if p in overrides),
            quality=last_options.get("quality", "best"),
            output_sharpening=last_options.get("output_sharpening", "standard"),
            deep_denoise=bool(last_options.get("deep_denoise", False)),
        )
        if dialog.exec() != QDialog.Accepted:
            return

        output_dir = dialog.output_dir()
        suffix = dialog.suffix()
        output_format = dialog.output_format()
        skip_completed = dialog.skip_completed()
        quality = dialog.quality()
        disable_segmentation = dialog.disable_segmentation()
        output_sharpening = dialog.output_sharpening()
        deep_denoise = dialog.deep_denoise()

        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Export Selected", "Select a valid output folder.")
            return
        if output_format not in self.BATCH_OUTPUT_FORMATS:
            QMessageBox.warning(self, "Export Selected", "Select a valid output format.")
            return
        if not self._confirm_collection_overwrite(selected, output_dir, suffix, output_format, skip_completed):
            return

        self._save_last_batch_options(
            {
                **last_options,
                "output_dir": output_dir,
                "suffix": suffix,
                "format": output_format,
                "skip_completed": skip_completed,
                "quality": quality,
                "disable_segmentation": disable_segmentation,
                "output_sharpening": output_sharpening,
                "deep_denoise": deep_denoise,
            }
        )
        self._dispatch_collection_export_job(
            selected, overrides, output_dir, suffix, output_format, skip_completed, quality, disable_segmentation,
            output_sharpening, deep_denoise, label=f'"{self._active_collection}" selected',
        )

    def _active_collection_images_or_warn(self):
        """Shared guard for both collection-export entry points: validates a collection is
        active and non-empty, flushes the open image's live edits into its override (so
        exporting right after editing without switching away still picks up the latest
        changes), and returns (images, overrides) or (None, None) if export can't proceed."""
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return None, None
        collection = self._collections[self._active_collection]
        images = collection.get("images", [])
        if not images:
            QMessageBox.information(self, "Empty Collection", "This collection has no images.")
            return None, None
        self._save_current_image_override()
        return images, collection.get("image_overrides", {})

    def _confirm_collection_overwrite(self, images, output_dir, suffix, output_format, skip_completed) -> bool:
        if skip_completed:
            return True
        output_ext = self.BATCH_OUTPUT_FORMATS[output_format]
        existing = sum(
            1 for p in images
            if os.path.exists(self._batch_output_path(p, output_dir, suffix=suffix, output_ext=output_ext))
        )
        if not existing:
            return True
        reply = QMessageBox.question(
            self,
            "Files Will Be Overwritten",
            f"{existing} of {len(images)} output file(s) already exist in this folder "
            "and will be overwritten (\"Skip already completed\" is off). Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _dispatch_collection_export_job(
        self, images, overrides, output_dir, suffix, output_format, skip_completed, quality, disable_segmentation,
        output_sharpening="standard", deep_denoise=False, label=None,
    ):
        image_overrides = {os.path.abspath(p): overrides[p] for p in images if p in overrides}
        job_path = self._write_batch_job_file(
            {
                "mode": "collection_export",
                "output_dir": os.path.abspath(output_dir),
                "suffix": suffix,
                "output_format": output_format,
                "skip_completed": skip_completed,
                "disable_segmentation": disable_segmentation,
                "output_sharpening": output_sharpening,
                "deep_denoise": bool(deep_denoise),
                "runtime_settings": dict(self._runtime_settings),
                "source_paths": [os.path.abspath(p) for p in images],
                "image_overrides": image_overrides,
            },
            output_dir,
        )
        job_id = Path(job_path).stem
        self._launch_background_batch_job(job_path, output_dir)
        self._set_filmstrip_export_status(images, "queued")
        self._start_export_monitor(
            job_id=job_id,
            output_dir=os.path.abspath(output_dir),
            source_count=len(images),
            label=label or f'"{self._active_collection}"',
            source_paths=images,
        )
        self.statusBar().showMessage(
            f"Exporting \"{self._active_collection}\" ({len(images)} image(s), {quality} quality)…"
        )

    def _remember_recent_preset(self, path: str):
        normalized = str(Path(path).resolve())
        recent = [normalized]
        recent.extend(item for item in self._recent_preset_paths if item != normalized)
        self._recent_preset_paths = recent[:8]
        self._save_browser_state()
        self._refresh_recent_presets()

    def _browser_state_payload(self):
        path = self._browser_state_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _batch_profiles(self):
        payload = self._browser_state_payload()
        profiles = payload.get("batch_profiles", {})
        return profiles if isinstance(profiles, dict) else {}

    def _save_batch_profiles(self, profiles, selected_profile=""):
        payload = self._browser_state_payload()
        payload["batch_profiles"] = dict(profiles or {})
        payload["last_batch_profile"] = str(selected_profile or "")
        self._write_browser_state(payload)

    def _last_batch_options(self):
        payload = self._browser_state_payload()
        last = payload.get("last_batch_options", {})
        return last if isinstance(last, dict) else {}

    def _save_last_batch_options(self, options):
        payload = self._browser_state_payload()
        payload["last_batch_options"] = dict(options or {})
        self._write_browser_state(payload)

    def _all_params(self):
        return {
            layer: {key: slider.value() for key, slider in sliders.items()}
            for layer, sliders in self._sliders.items()
        }

    def _copy_face_profiles(self, payload):
        copied = {}
        for key, profile in (payload or {}).items():
            copied[int(key)] = {
                "selective_params": {layer: dict(values) for layer, values in profile.get("selective_params", {}).items()},
                "layer_options": {layer: dict(cfg) for layer, cfg in profile.get("layer_options", {}).items()},
                "layer_order": list(profile.get("layer_order", list(MASK_ORDER))),
                "mask_adjustments": self._copy_mask_adjustments(profile.get("mask_adjustments", {})),
                "preview_masks": self._ref_masks(profile.get("preview_masks")),
                "full_masks": self._ref_masks(profile.get("full_masks")),
                "auto_preview_masks": self._ref_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._ref_masks(profile.get("auto_full_masks")),
                "mask_face_index": profile.get("mask_face_index"),
                "preview_guides": self._copy_guides(profile.get("preview_guides")),
                "full_guides": self._copy_guides(profile.get("full_guides")),
            }
        return copied

    def _history_face_profile_signature(self, payload):
        signature = {}
        for key, profile in sorted((payload or {}).items(), key=lambda item: int(item[0])):
            signature[str(key)] = {
                "selective_params": {layer: dict(values) for layer, values in profile.get("selective_params", {}).items()},
                "layer_options": {layer: dict(cfg) for layer, cfg in profile.get("layer_options", {}).items()},
                "layer_order": list(profile.get("layer_order", list(MASK_ORDER))),
                "mask_adjustments": self._copy_mask_adjustments(profile.get("mask_adjustments", {})),
            }
        return signature

    def _document_state_signature(self, snapshot):
        payload = {
            "global_params": snapshot.get("global_params", {}),
            "color_settings": snapshot.get("color_settings", {}),
            "active_face_index": snapshot.get("active_face_index", 0),
            "face_profiles": self._history_face_profile_signature(snapshot.get("face_profiles", {})),
            "compare_mode": snapshot.get("compare_mode", "off"),
            "split_position": snapshot.get("split_position", 0.5),
            "framing": normalize_framing(snapshot.get("framing")),
            "show_mask": snapshot.get("show_mask", False),
            "mask_debug_mode": snapshot.get("mask_debug_mode", "tint"),
            "show_expression_guides": snapshot.get("show_expression_guides", False),
            "active_layer": snapshot.get("active_layer", "global"),
            "essentials_only": snapshot.get("essentials_only", True),
            "runtime_settings": snapshot.get("runtime_settings", {}),
            "mask_adjustments": self._copy_mask_adjustments(snapshot.get("mask_adjustments", {})),
            "active_preset_meta": snapshot.get("active_preset_meta"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _capture_document_state(self):
        self._store_active_face_profile()
        params = self._all_params()
        return {
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "active_face_index": int(self._active_face_index),
            "face_profiles": self._copy_face_profiles(self._face_profiles),
            "compare_mode": self._compare_mode,
            "split_position": float(self._split_position),
            "framing": normalize_framing(self._framing),
            "show_mask": bool(self._show_mask),
            "mask_debug_mode": self._mask_debug_mode,
            "show_expression_guides": bool(self._show_expression_guides),
            "active_layer": self._active_layer,
            "essentials_only": bool(self._essentials_only),
            "runtime_settings": dict(self._runtime_settings),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "active_preset_meta": copy.deepcopy(self._active_preset_meta),
        }

    def _build_settings_payload(self):
        """Snapshot global + selective adjustments, layer settings, and color management
        for the current image, in the same shape used by the settings clipboard and by
        per-image collection overrides."""
        self._store_active_face_profile()
        params = self._all_params()
        selective_by_face = []
        for face_idx in sorted(self._face_profiles.keys()):
            profile = self._face_profiles.get(face_idx)
            if profile:
                selective = {layer: dict(profile.get("selective_params", {}).get(layer, {})) for layer in MASK_ORDER}
                selective_by_face.append({"face_index": face_idx, "params": selective})
        payload = {
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "selective_by_face": selective_by_face,
            "framing": normalize_framing(self._framing),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "active_face_index": int(self._active_face_index),
        }
        if isinstance(self._active_preset_meta, dict):
            payload["_preset"] = copy.deepcopy(self._active_preset_meta)
        return payload

    def _sync_color_setting_controls(self):
        if hasattr(self, "_sync_raw_decode_controls"):
            self._sync_raw_decode_controls()
        if hasattr(self, "_sync_wb_controls"):
            self._sync_wb_controls()
        if hasattr(self, "_sync_tone_curve_controls"):
            self._sync_tone_curve_controls()
        if hasattr(self, "_sync_hsl_controls"):
            self._sync_hsl_controls()

    def _raw_decode_settings_changed(self, previous_color_settings: dict) -> bool:
        if not self.file_path or not is_raw_path(self.file_path):
            return False
        previous = previous_color_settings if isinstance(previous_color_settings, dict) else {}
        defaults = self._default_color_settings()
        for key in self._RAW_DECODE_KEYS:
            old_value = previous.get(key, defaults.get(key))
            new_value = self._color_settings.get(key, defaults.get(key))
            if old_value != new_value:
                return True
        return False

    @staticmethod
    def _read_preset_meta(payload: dict | None):
        """Read the "applied from a preset" meta block, preferring the current '_preset' key
        but falling back to the legacy '_preset' key -- so collection overrides and clipboard
        payloads saved before the Preset/Preset rename keep loading correctly."""
        if not isinstance(payload, dict):
            return None
        meta = payload.get("_preset")
        if meta is None:
            meta = payload.get("_template")
        return meta

    def _settings_payload_signature(self, payload: dict) -> str:
        normalized = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        normalized.pop("_preset", None)
        normalized.pop("_template", None)  # legacy key; strip it too if present
        if "framing" in normalized:
            normalized["framing"] = normalize_framing(normalized.get("framing"))
        if "mask_adjustments" in normalized:
            normalized["mask_adjustments"] = self._copy_mask_adjustments(normalized.get("mask_adjustments"))
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    def _preset_payload_meta(self, payload: dict, name: str, path: str | Path | None = None, preset: dict | None = None) -> dict:
        payload = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        payload.pop("_preset", None)
        payload.pop("_template", None)
        return {
            "name": str(name or "Preset"),
            "path": str(path or ""),
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "includes": self._preset_include_labels(preset or {}) if isinstance(preset, dict) else [],
            "signature": self._settings_payload_signature(payload),
        }

    def _attach_preset_meta(self, payload: dict, meta: dict | None) -> dict:
        result = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        result.pop("_template", None)  # never leave a stale legacy key alongside the new one
        if isinstance(meta, dict) and meta.get("signature"):
            result["_preset"] = copy.deepcopy(meta)
        return result

    def _preset_state_for_override(self, payload: dict):
        if not isinstance(payload, dict):
            return "", ""
        meta = self._read_preset_meta(payload)
        if not isinstance(meta, dict) or not meta.get("signature"):
            return "", ""
        state = "applied" if self._settings_payload_signature(payload) == meta.get("signature") else "modified"
        return state, str(meta.get("name") or "")

    def _copy_settings(self):
        """Copy global + selective adjustments, layer settings, and color management to clipboard."""
        if self.preview_array is None:
            self.statusBar().showMessage("No image loaded — nothing to copy")
            return
        self._settings_clipboard = self._build_settings_payload()
        self._paste_settings_action.setEnabled(True)
        if hasattr(self, '_paste_settings_btn'):
            self._paste_settings_btn.setEnabled(True)
        if hasattr(self, 'settings_indicator'):
            self.settings_indicator.setText("copied ✓")
        self.statusBar().showMessage("Settings copied ✓")

    def _save_current_image_override(self):
        """Persist the current image's own edits into its collection's per-image override,
        so switching to another image and back doesn't lose them. No-op outside a collection."""
        if not self.file_path or not self._active_collection or self.preview_array is None:
            return
        if self._document_history_index <= 0:
            return
        collection = self._collections.get(self._active_collection)
        if not collection or self.file_path not in collection.get("images", []):
            return
        overrides = collection.setdefault("image_overrides", {})
        payload = self._build_settings_payload()
        existing = overrides.get(self.file_path)
        if (
            isinstance(existing, dict)
            and self._settings_payload_signature(existing) == self._settings_payload_signature(payload)
            and self._read_preset_meta(existing) == self._read_preset_meta(payload)
        ):
            return
        overrides[self.file_path] = payload
        self._save_collections_state()
        thumb = self._filmstrip_items.get(self.file_path)
        if thumb is not None:
            thumb.set_has_override(True)
            state, name = self._preset_state_for_override(payload)
            thumb.set_preset_state(state, name)

    def _reset_editing_state_to_defaults(self):
        """Reset every slider, layer option, and color setting to its default before loading
        a fresh image, so an image with no saved override starts clean instead of inheriting
        whatever the previously open image was left at."""
        for layer, sliders in self._sliders.items():
            defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS.get(layer, [])}
            for key, slider in sliders.items():
                value = defaults.get(key, 0)
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                label = self._slider_value_labels.get(layer, {}).get(key)
                if label is not None:
                    label.setText(f"{int(value):+d}" if value else "0")
        self._layer_order = list(MASK_ORDER)
        self._layer_options = {layer: {"enabled": True, "opacity": 100.0, "blend_mode": "normal"} for layer in MASK_ORDER}
        self._color_settings = self._default_color_settings()
        self._active_preset_meta = None

    def _apply_settings_payload(self, payload: dict):
        """Apply a settings-clipboard-shaped payload (global/color/layers/selective) to the current image."""
        if not payload:
            return False
        previous_color_settings = dict(self._color_settings)
        # Apply global params.
        global_params = payload.get("global_params", {})
        for key, value in global_params.items():
            if "global" in self._sliders and key in self._sliders["global"]:
                slider = self._sliders["global"][key]
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                if "global" in self._slider_value_labels and key in self._slider_value_labels["global"]:
                    label = self._slider_value_labels["global"][key]
                    label.setText(f"{int(value):+d}" if value else "0")
        # Apply color settings.
        color_settings = payload.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged = self._default_color_settings()
            merged.update(color_settings)
            self._color_settings = merged
            self._sync_color_setting_controls()
        # Apply layer options and order.
        layer_options = payload.get("layer_options", {})
        if isinstance(layer_options, dict):
            for layer in MASK_ORDER:
                cfg = dict(self._layer_options.get(layer, {}))
                cfg.update(layer_options.get(layer, {}))
                self._layer_options[layer] = cfg
        layer_order = [layer for layer in payload.get("layer_order", []) if layer in MASK_ORDER]
        for layer in MASK_ORDER:
            if layer not in layer_order:
                layer_order.append(layer)
        self._layer_order = layer_order
        # Sync mask adjustment controls.
        if hasattr(self, '_sync_mask_adjustment_controls'):
            self._sync_mask_adjustment_controls()
        # Apply selective params to detected faces.
        selective_by_face = payload.get("selective_by_face", [])
        for face_idx, profile_data in enumerate(selective_by_face):
            if face_idx >= len(self._detected_faces):
                break
            key = int(face_idx) if self._detected_faces else -1
            if key not in self._face_profiles:
                self._face_profiles[key] = {
                    "selective_params": {layer: {} for layer in MASK_ORDER},
                    "layer_options": {layer: {} for layer in MASK_ORDER},
                    "layer_order": list(MASK_ORDER),
                    "mask_adjustments": {},
                    "preview_masks": {},
                    "full_masks": {},
                    "auto_preview_masks": {},
                    "auto_full_masks": {},
                    "preview_guides": {},
                    "full_guides": {},
                }
            for layer in MASK_ORDER:
                self._face_profiles[key]["selective_params"][layer] = dict(
                    profile_data.get("params", {}).get(layer, {})
                )
        # Reload the active face profile to update the UI.
        self._restore_face_profile(self._active_face_index)
        # Apply crop/straighten/flip framing, if present.
        if "framing" in payload:
            self._framing = normalize_framing(payload.get("framing"))
            self._sync_framing_controls()
        # Apply mask adjustments (feather/strength/expand-contract per layer), if present.
        # Applied after the face-profile reload above so it isn't overwritten by that face's
        # own (possibly placeholder) mask_adjustments.
        if "mask_adjustments" in payload:
            self._mask_adjustments = self._copy_mask_adjustments(payload.get("mask_adjustments"))
            if hasattr(self, '_sync_mask_adjustment_controls'):
                self._sync_mask_adjustment_controls()
        self._active_preset_meta = copy.deepcopy(self._read_preset_meta(payload)) if isinstance(self._read_preset_meta(payload), dict) else None
        return self._raw_decode_settings_changed(previous_color_settings)

    def _apply_pre_segmentation_settings_payload(self, payload: dict):
        """Apply saved settings that do not need masks/faces before the first image paint."""
        if not payload:
            return
        global_params = payload.get("global_params", {})
        for key, value in global_params.items():
            if "global" in self._sliders and key in self._sliders["global"]:
                slider = self._sliders["global"][key]
                slider.blockSignals(True)
                slider.setValue(int(value))
                slider.blockSignals(False)
                if "global" in self._slider_value_labels and key in self._slider_value_labels["global"]:
                    label = self._slider_value_labels["global"][key]
                    label.setText(f"{int(value):+d}" if value else "0")

        color_settings = payload.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged = self._default_color_settings()
            merged.update(color_settings)
            self._color_settings = merged
            self._sync_color_setting_controls()

        layer_options = payload.get("layer_options", {})
        if isinstance(layer_options, dict):
            for layer in MASK_ORDER:
                cfg = dict(self._layer_options.get(layer, {}))
                cfg.update(layer_options.get(layer, {}))
                self._layer_options[layer] = cfg
        layer_order = [layer for layer in payload.get("layer_order", []) if layer in MASK_ORDER]
        if layer_order:
            for layer in MASK_ORDER:
                if layer not in layer_order:
                    layer_order.append(layer)
            self._layer_order = layer_order

        if "mask_adjustments" in payload:
            self._mask_adjustments = self._copy_mask_adjustments(payload.get("mask_adjustments"))
            if hasattr(self, "_sync_mask_adjustment_controls"):
                self._sync_mask_adjustment_controls()

        if "active_face_index" in payload:
            self._active_face_index = max(0, int(payload.get("active_face_index", 0)))

        if "framing" in payload:
            self._framing = normalize_framing(payload.get("framing"))
            self._sync_framing_controls()
        self._active_preset_meta = copy.deepcopy(self._read_preset_meta(payload)) if isinstance(self._read_preset_meta(payload), dict) else None

    def _paste_settings(self):
        """Paste copied settings to the current image, or to every selected filmstrip image
        at once if more than one is selected -- mirrors the existing Auto WB (AI) multi-select
        pattern, instead of requiring one paste per image."""
        if self._settings_clipboard is None:
            self.statusBar().showMessage("No settings in clipboard")
            return
        selected_paths = self._selected_filmstrip_paths()
        if len(selected_paths) > 1:
            self._paste_settings_to_selected_filmstrip_images(selected_paths)
            return
        if self.preview_array is None:
            self.statusBar().showMessage("No image loaded — nothing to paste to")
            return
        clip_faces = len(self._settings_clipboard.get("selective_by_face", []))
        target_faces = len(self._detected_faces)
        if clip_faces and target_faces and clip_faces != target_faces:
            reply = QMessageBox.question(
                self,
                "Face Count Mismatch",
                f"The copied settings were saved for {clip_faces} face(s), but this image has "
                f"{target_faces}. Only the first {min(clip_faces, target_faces)} face(s) will be "
                "updated; any others keep their current settings. Paste anyway?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._begin_document_change()
        raw_decode_changed = self._apply_settings_payload(self._settings_clipboard)
        self._push_document_history()
        if raw_decode_changed:
            self._reload_source_for_decode_settings()
        else:
            self._schedule_render()
        pasted_name = os.path.basename(self.file_path) if self.file_path else "image"
        if hasattr(self, 'settings_indicator'):
            # Briefly confirm *this* paste landed, then fall back to "copied" -- the
            # clipboard is still loaded and can be pasted to more images.
            self.settings_indicator.setText("pasted ✓")
            QTimer.singleShot(1500, lambda: self.settings_indicator.setText("copied ✓") if self._settings_clipboard is not None else None)
        self.statusBar().showMessage(f"Settings pasted to {pasted_name} ✓")

    def _paste_settings_to_selected_filmstrip_images(self, paths):
        if not self._active_collection or self._active_collection not in self._collections:
            self.statusBar().showMessage("Paste Settings: select images from a collection first")
            return
        collection = self._collections[self._active_collection]
        collection_paths = set(collection.get("images", []))
        paths = [p for p in paths if p in collection_paths]
        if len(paths) <= 1:
            return
        overrides = collection.setdefault("image_overrides", {})
        for path in paths:
            if path == self.file_path and self.preview_array is not None:
                self._begin_document_change()
                raw_decode_changed = self._apply_settings_payload(self._settings_clipboard)
                self._push_document_history()
                if raw_decode_changed:
                    self._reload_source_for_decode_settings()
                else:
                    self._schedule_render()
                overrides[path] = self._build_settings_payload()
            else:
                overrides[path] = copy.deepcopy(self._settings_clipboard)
            thumb = self._filmstrip_items.get(path)
            if thumb is not None:
                thumb.set_has_override(True)
                state, name = self._preset_state_for_override(overrides[path])
                thumb.set_preset_state(state, name)
        self._save_collections_state()
        self.statusBar().showMessage(f"Pasted settings to {len(paths)} selected image(s)")

    def _clear_document_history(self):
        self._document_history = []
        self._document_history_index = -1
        self._update_document_history_actions()

    def _push_document_history(self):
        if self.preview_array is None or self._restoring_document_state:
            return
        snapshot = self._capture_document_state()
        snapshot["_history_sig"] = self._document_state_signature(snapshot)
        if self._document_history_index < len(self._document_history) - 1:
            self._document_history = self._document_history[: self._document_history_index + 1]
        self._document_history.append(snapshot)
        if len(self._document_history) > self._max_document_history:
            overflow = len(self._document_history) - self._max_document_history
            self._document_history = self._document_history[overflow:]
        self._document_history_index = len(self._document_history) - 1
        self._update_document_history_actions()

    def _begin_document_change(self):
        if self._restoring_document_state or self.preview_array is None:
            if not self._restoring_document_state:
                self._hydrate_active_image_if_needed("edit", preserve_current_settings=True)
            return
        current = self._capture_document_state()
        current_sig = self._document_state_signature(current)
        if self._document_history_index >= 0:
            last = self._document_history[self._document_history_index]
            if current_sig == last.get("_history_sig"):
                return
        current["_history_sig"] = current_sig
        if self._document_history_index < len(self._document_history) - 1:
            self._document_history = self._document_history[: self._document_history_index + 1]
        self._document_history.append(current)
        if len(self._document_history) > self._max_document_history:
            overflow = len(self._document_history) - self._max_document_history
            self._document_history = self._document_history[overflow:]
        self._document_history_index = len(self._document_history) - 1
        self._update_document_history_actions()

    def _restore_document_state(self, snapshot):
        if snapshot is None:
            return
        previous_color_settings = dict(self._color_settings)
        self._restoring_document_state = True
        try:
            global_params = snapshot.get("global_params", {})
            defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS["global"]}
            for key, slider in self._sliders.get("global", {}).items():
                slider.blockSignals(True)
                value = int(global_params.get(key, defaults[key]))
                slider.setValue(value)
                slider.blockSignals(False)
                self._slider_value_labels["global"][key].setText(f"{value:+d}" if value else "0")

            color_settings = snapshot.get("color_settings", {})
            if isinstance(color_settings, dict):
                merged = self._default_color_settings()
                merged.update(color_settings)
                self._color_settings = merged

            runtime_settings = snapshot.get("runtime_settings", {})
            if isinstance(runtime_settings, dict):
                self._runtime_settings.update(runtime_settings)

            self._mask_adjustments = self._copy_mask_adjustments(snapshot.get("mask_adjustments", {}))
            self._active_preset_meta = copy.deepcopy(snapshot.get("active_preset_meta")) if isinstance(snapshot.get("active_preset_meta"), dict) else None
            self._face_profiles = self._copy_face_profiles(snapshot.get("face_profiles", {}))
            if self._detected_faces:
                self._active_face_index = int(np.clip(int(snapshot.get("active_face_index", 0)), 0, len(self._detected_faces) - 1))
            else:
                self._active_face_index = 0

            self._compare_mode = str(snapshot.get("compare_mode", "off") or "off")
            self._split_position = max(0.0, min(1.0, float(snapshot.get("split_position", 0.5))))
            self._framing = normalize_framing(snapshot.get("framing"))
            self._show_mask = bool(snapshot.get("show_mask", False))
            self._mask_debug_mode = str(snapshot.get("mask_debug_mode", "tint") or "tint")
            self._show_expression_guides = bool(snapshot.get("show_expression_guides", False))
            self._essentials_only = bool(snapshot.get("essentials_only", True))
            active_layer = str(snapshot.get("active_layer", "global") or "global")

            self.face_combo.blockSignals(True)
            if self.face_combo.count():
                self.face_combo.setCurrentIndex(self._active_face_index)
            self.face_combo.blockSignals(False)
            self._update_face_scope_indicator()
            self.compare_combo.blockSignals(True)
            self.compare_combo.setCurrentText(self._compare_mode)
            self.compare_combo.blockSignals(False)
            self.mask_view_btn.blockSignals(True)
            self.mask_view_btn.setChecked(self._show_mask)
            self.mask_view_btn.blockSignals(False)
            self.mask_debug_combo.blockSignals(True)
            self.mask_debug_combo.setCurrentText(self._mask_debug_mode)
            self.mask_debug_combo.blockSignals(False)
            self.guides_btn.blockSignals(True)
            self.guides_btn.setChecked(self._show_expression_guides)
            self.guides_btn.blockSignals(False)
            self.essentials_only_check.blockSignals(True)
            self.essentials_only_check.setChecked(self._essentials_only)
            self.essentials_only_check.blockSignals(False)
            self.split_slider.blockSignals(True)
            self.split_slider.setValue(int(round(self._split_position * 100)))
            self.split_slider.blockSignals(False)
            self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
            self.split_slider.setEnabled(self._compare_mode == "split")
            self._sync_framing_controls()
            self._sync_color_setting_controls()

            if not self._restore_face_profile(self._active_face_index):
                self.preview_masks = self._copy_masks(self._auto_preview_masks)
                self.full_masks = self._copy_masks(self._auto_full_masks)
            self._apply_essentials_filter()
            if active_layer in ALL_LAYERS:
                self._activate_layer(active_layer)
            self._refresh_mask_controls()
            self._update_preview_label()
        finally:
            self._restoring_document_state = False
        if self._raw_decode_settings_changed(previous_color_settings):
            return self._reload_source_for_decode_settings()
        self._schedule_render()
        return False

    def _update_document_history_actions(self):
        can_undo = self._document_history_index > 0
        can_redo = self._document_history_index >= 0 and self._document_history_index < len(self._document_history) - 1
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(can_undo)
        if hasattr(self, "redo_action"):
            self.redo_action.setEnabled(can_redo)
        # Masks panel's Undo/Redo are the same timeline as Ctrl+Z now, just a convenient
        # local affordance while actively painting -- keep their enabled state in sync too.
        if hasattr(self, "undo_mask_btn"):
            self.undo_mask_btn.setEnabled(can_undo)
        if hasattr(self, "redo_mask_btn"):
            self.redo_mask_btn.setEnabled(can_redo)

    def _undo_document_state(self):
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit)):
            return
        if self._document_history_index <= 0:
            return
        self._document_history_index -= 1
        self._restore_document_state(self._document_history[self._document_history_index])
        self._update_document_history_actions()

    def _redo_document_state(self):
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit)):
            return
        if self._document_history_index < 0 or self._document_history_index >= len(self._document_history) - 1:
            return
        self._document_history_index += 1
        self._restore_document_state(self._document_history[self._document_history_index])
        self._update_document_history_actions()

    def _current_profile_key(self):
        if not self._detected_faces:
            return -1
        return int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))

    def _copy_guides(self, guides):
        return _copy_guides_data(guides)

    def _store_active_face_profile(self):
        if self.preview_masks is None:
            return
        key = self._current_profile_key()
        params = self._all_params()
        self._face_profiles[key] = {
            "selective_params": {layer: dict(params.get(layer, {})) for layer in MASK_ORDER},
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "preview_masks": self._ref_masks(self.preview_masks),
            "full_masks": self._ref_masks(self.full_masks),
            "auto_preview_masks": self._ref_masks(self._auto_preview_masks),
            "auto_full_masks": self._ref_masks(self._auto_full_masks),
            "mask_face_index": int(self._mask_face_index) if self._mask_face_index is not None else key,
            "preview_guides": self._copy_guides(self.preview_guides),
            "full_guides": self._copy_guides(self.full_guides),
        }

    def _restore_face_profile(self, face_index: int):
        key = -1 if not self._detected_faces else int(face_index)
        profile = self._face_profiles.get(key)
        if profile is None:
            return False
        selective_params = profile.get("selective_params", {})
        for layer in MASK_ORDER:
            defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS[layer]}
            for key_name, slider in self._sliders.get(layer, {}).items():
                slider.blockSignals(True)
                value = int(selective_params.get(layer, {}).get(key_name, defaults[key_name]))
                slider.setValue(value)
                slider.blockSignals(False)
                self._slider_value_labels[layer][key_name].setText(f"{value:+d}" if value else "0")
        self._layer_options = {layer: dict(cfg) for layer, cfg in profile.get("layer_options", self._layer_options).items()}
        layer_order = [layer for layer in profile.get("layer_order", list(MASK_ORDER)) if layer in MASK_ORDER]
        for layer in MASK_ORDER:
            if layer not in layer_order:
                layer_order.append(layer)
        self._layer_order = layer_order
        self._mask_adjustments = self._copy_mask_adjustments(profile.get("mask_adjustments", {}))
        # Params/layer options/mask-adjustments above are always safe to restore. The masks are
        # only restorable if the profile actually has them -- a profile injected by
        # _apply_settings_payload from a saved collection override carries empty mask dicts
        # (masks aren't persisted), and restoring those empties would blank every layer's Mask
        # View. Report not-restored in that case so the caller re-runs segmentation to fill
        # them, while keeping the per-face slider params we just applied.
        if not _masks_usable(profile.get("preview_masks")):
            return False
        profile_mask_face_index = profile.get("mask_face_index")
        if profile_mask_face_index is None and len(self._detected_faces) > 1:
            return False
        if profile_mask_face_index is not None and int(profile_mask_face_index) != int(face_index):
            return False
        self.preview_masks = self._copy_masks(profile.get("preview_masks"))
        self.full_masks = self._copy_masks(profile.get("full_masks"))
        self._auto_preview_masks = self._copy_masks(profile.get("auto_preview_masks"))
        self._auto_full_masks = self._copy_masks(profile.get("auto_full_masks"))
        self._mask_face_index = int(profile_mask_face_index) if profile_mask_face_index is not None else int(face_index)
        self.preview_guides = self._copy_guides(profile.get("preview_guides"))
        self.full_guides = self._copy_guides(profile.get("full_guides"))
        return True

    def _encode_array(self, array):
        if array is None:
            return None
        payload = zlib.compress(np.asarray(array).tobytes())
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
                "selective_params": profile.get("selective_params", {}),
                "layer_options": profile.get("layer_options", {}),
                "layer_order": profile.get("layer_order", list(MASK_ORDER)),
                "mask_adjustments": self._copy_mask_adjustments(profile.get("mask_adjustments", {})),
                "preview_masks": self._encode_masks(profile.get("preview_masks")),
                "full_masks": self._encode_masks(profile.get("full_masks")),
                "auto_preview_masks": self._encode_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._encode_masks(profile.get("auto_full_masks")),
                "mask_face_index": profile.get("mask_face_index"),
                "preview_guides": profile.get("preview_guides"),
                "full_guides": profile.get("full_guides"),
            }
        return out

    def _deserialize_face_profiles(self, payload):
        out = {}
        for key, profile in (payload or {}).items():
            out[int(key)] = {
                "selective_params": profile.get("selective_params", {}),
                "layer_options": profile.get("layer_options", {}),
                "layer_order": profile.get("layer_order", list(MASK_ORDER)),
                "mask_adjustments": self._copy_mask_adjustments(profile.get("mask_adjustments", {})),
                "preview_masks": self._decode_masks(profile.get("preview_masks")),
                "full_masks": self._decode_masks(profile.get("full_masks")),
                "auto_preview_masks": self._decode_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._decode_masks(profile.get("auto_full_masks")),
                "mask_face_index": profile.get("mask_face_index"),
                "preview_guides": self._copy_guides(profile.get("preview_guides")),
                "full_guides": self._copy_guides(profile.get("full_guides")),
            }
        return out

    def _serialize_project_state(self):
        self._store_active_face_profile()
        params = self._all_params()
        return {
            "version": 1,
            "image_path": os.path.abspath(self.file_path) if self.file_path else "",
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "active_face_index": int(self._active_face_index),
            "detected_faces": [list(face) for face in self._detected_faces],
            "face_profiles": self._serialize_face_profiles(),
            "compare_mode": self._compare_mode,
            "split_position": float(self._split_position),
            "framing": normalize_framing(self._framing),
            "show_mask": bool(self._show_mask),
            "mask_debug_mode": self._mask_debug_mode,
            "show_expression_guides": bool(self._show_expression_guides),
            "active_layer": self._active_layer,
            "essentials_only": bool(self._essentials_only),
            "runtime_settings": dict(self._runtime_settings),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
        }

    def _apply_project_state(self, project):
        params = project.get("global_params", {})
        defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS["global"]}
        for key, slider in self._sliders.get("global", {}).items():
            slider.blockSignals(True)
            value = int(params.get(key, defaults[key]))
            slider.setValue(value)
            slider.blockSignals(False)
            self._slider_value_labels["global"][key].setText(f"{value:+d}" if value else "0")

        color_settings = project.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged = self._default_color_settings()
            merged.update(color_settings)
            self._color_settings = merged

        runtime_settings = project.get("runtime_settings", {})
        if isinstance(runtime_settings, dict):
            self._runtime_settings.update(runtime_settings)

        self._mask_adjustments = self._copy_mask_adjustments(project.get("mask_adjustments", {}))
        self._detected_faces = [tuple(int(v) for v in face) for face in project.get("detected_faces", [])]
        self._active_face_index = int(project.get("active_face_index", 0))
        if self._detected_faces:
            self._active_face_index = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        else:
            self._active_face_index = 0
        self._face_profiles = self._deserialize_face_profiles(project.get("face_profiles", {}))

        self._compare_mode = str(project.get("compare_mode", "off") or "off")
        self._split_position = max(0.0, min(1.0, float(project.get("split_position", 0.5))))
        self._framing = normalize_framing(project.get("framing"))
        self._show_mask = bool(project.get("show_mask", False))
        self._mask_debug_mode = str(project.get("mask_debug_mode", "tint") or "tint")
        self._show_expression_guides = bool(project.get("show_expression_guides", False))
        self._essentials_only = bool(project.get("essentials_only", True))
        active_layer = str(project.get("active_layer", "global") or "global")

        self.face_combo.blockSignals(True)
        self.face_combo.clear()
        if not self._detected_faces:
            self.face_combo.addItem("Auto")
            self.face_combo.setCurrentIndex(0)
        else:
            for idx, (x, y, w, h) in enumerate(self._detected_faces):
                self.face_combo.addItem(f"Face {idx + 1} ({w}x{h} @ {x},{y})")
            self.face_combo.setCurrentIndex(self._active_face_index)
        self.face_combo.blockSignals(False)
        self._update_face_scope_indicator()

        self.compare_combo.blockSignals(True)
        self.compare_combo.setCurrentText(self._compare_mode)
        self.compare_combo.blockSignals(False)
        self.mask_view_btn.blockSignals(True)
        self.mask_view_btn.setChecked(self._show_mask)
        self.mask_view_btn.blockSignals(False)
        self.mask_debug_combo.blockSignals(True)
        self.mask_debug_combo.setCurrentText(self._mask_debug_mode)
        self.mask_debug_combo.blockSignals(False)
        self.guides_btn.blockSignals(True)
        self.guides_btn.setChecked(self._show_expression_guides)
        self.guides_btn.blockSignals(False)
        self.essentials_only_check.blockSignals(True)
        self.essentials_only_check.setChecked(self._essentials_only)
        self.essentials_only_check.blockSignals(False)
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(int(round(self._split_position * 100)))
        self.split_slider.blockSignals(False)

        if active_layer in ALL_LAYERS:
            self._activate_layer(active_layer)
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.split_slider.setEnabled(self._compare_mode == "split")
        self._sync_framing_controls()
        self._sync_wb_controls()
        self._sync_raw_decode_controls()
        self._sync_tone_curve_controls()
        self._sync_hsl_controls()

        if not self._restore_face_profile(self._active_face_index):
            self._run_segmentation()
            return
        self._apply_essentials_filter()
        self._refresh_mask_controls()
        self._update_preview_label()
        self._clear_document_history()
        self._push_document_history()
        self._schedule_render()

    def _serialize_preset_state(self):
        params = self._all_params()
        preset = {
            "version": 1,
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "selective_params": {layer: dict(params.get(layer, {})) for layer in MASK_ORDER},
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "meta": self._default_preset_meta(),
        }
        return self._stamp_preset_meta(preset)

    def _apply_preset_state(self, preset: dict, reload_raw_decode: bool = False):
        self._begin_document_change()
        previous_color_settings = dict(self._color_settings)
        global_params = preset.get("global_params", {})
        selective_params = preset.get("selective_params", {})
        for layer, sliders in ALL_LAYERS.items():
            defaults = {key: default for key, _label, _mn, _mx, default in sliders}
            layer_values = global_params if layer == "global" else selective_params.get(layer, {})
            for key, slider in self._sliders.get(layer, {}).items():
                slider.blockSignals(True)
                value = int(layer_values.get(key, defaults[key]))
                slider.setValue(value)
                slider.blockSignals(False)
                self._slider_value_labels[layer][key].setText(f"{value:+d}" if value else "0")

        color_settings = preset.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged = self._default_color_settings()
            merged.update(color_settings)
            self._color_settings = merged
            self._sync_color_setting_controls()

        layer_options = preset.get("layer_options", {})
        if isinstance(layer_options, dict):
            for layer in MASK_ORDER:
                cfg = dict(self._layer_options.get(layer, {}))
                cfg.update(layer_options.get(layer, {}))
                self._layer_options[layer] = cfg

        layer_order = [layer for layer in preset.get("layer_order", []) if layer in MASK_ORDER]
        for layer in MASK_ORDER:
            if layer not in layer_order:
                layer_order.append(layer)
        self._layer_order = layer_order
        self._mask_adjustments = self._copy_mask_adjustments(preset.get("mask_adjustments", self._mask_adjustments))
        self._sync_mask_adjustment_controls()
        if reload_raw_decode and self._raw_decode_settings_changed(previous_color_settings):
            if self._reload_source_for_decode_settings():
                return True
        self._schedule_render()
        return False

    def open_preset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Preset",
            "",
            "Portrait Preset (*.pepreset *.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                preset = json.load(fh)
            self._apply_preset_state(preset, reload_raw_decode=True)
            self._remember_recent_preset(path)
            self.statusBar().showMessage(f"Preset loaded -> {os.path.basename(path)}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))

    def save_preset(self):
        default_name = "preset.pepreset"
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preset",
            default_name,
            "Portrait Preset (*.pepreset *.json)",
        )
        if not out_path:
            return
        try:
            preset = self._serialize_preset_state()
            preset["meta"]["name"] = Path(out_path).stem
            preset["meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
            self._stamp_preset_meta(preset)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self._save_preset_thumbnail(out_path)
            self._remember_recent_preset(out_path)
            self.statusBar().showMessage(
                f"Preset saved -> {os.path.basename(out_path)} ({self._preset_summary_text(preset)})"
            )
        except Exception as ex:
            QMessageBox.critical(self, "Preset Save Error", self._friendly_error_message(ex))

    def _preset_library_dir(self):
        path = Path(os.getcwd()) / "presets"
        path.mkdir(exist_ok=True)
        return path

    def _preset_thumbnail_path(self, preset_path):
        if not preset_path:
            return None
        preset_path = Path(preset_path)
        if not preset_path.name:
            return None
        return preset_path.with_name(f"{preset_path.stem}.thumb.jpg")

    def _preset_has_thumbnail(self, preset_path) -> bool:
        thumb_path = self._preset_thumbnail_path(preset_path)
        return bool(thumb_path is not None and thumb_path.exists())

    def _configure_preset_list_layout(self, entries):
        if not hasattr(self, "preset_list"):
            return
        if not entries:
            compact = False
        else:
            missing = sum(1 for entry in entries if not self._preset_has_thumbnail(entry["path"]))
            compact = missing > len(entries) / 2
        if compact == getattr(self, "_preset_browser_compact", False):
            return
        self._preset_browser_compact = compact
        if compact:
            self.preset_list.setViewMode(QListView.ListMode)
            self.preset_list.setIconSize(QSize(42, 42))
            self.preset_list.setGridSize(QSize())
            self.preset_list.setWordWrap(False)
            self.preset_list.setSpacing(2)
            self.preset_list.setUniformItemSizes(True)
        else:
            self.preset_list.setViewMode(QListView.IconMode)
            self.preset_list.setResizeMode(QListView.Adjust)
            self.preset_list.setMovement(QListView.Static)
            self.preset_list.setIconSize(QSize(112, 76))
            self.preset_list.setGridSize(QSize(132, 116))
            self.preset_list.setWordWrap(True)
            self.preset_list.setSpacing(6)
            self.preset_list.setUniformItemSizes(False)

    def _save_preset_thumbnail(self, preset_path):
        if self.preview_image is None:
            return
        try:
            thumb = self.preview_image.convert("RGB").copy()
            thumb.thumbnail((320, 220), Image.LANCZOS)
            thumb.save(self._preset_thumbnail_path(preset_path), quality=90)
        except Exception:
            return

    def _preset_accent_color(self, meta: dict) -> QColor:
        category = str(meta.get("category", "general") or "general")
        palette = {
            "general": "#d4a853",
            "recipe": "#5b9bd5",
            "portrait": "#d98c9f",
            "event": "#65b891",
            "studio": "#b7a6e8",
        }
        if category in palette:
            return QColor(palette[category])
        digest = hashlib.sha1(category.encode("utf-8")).digest()
        return QColor(90 + digest[0] % 120, 90 + digest[1] % 120, 90 + digest[2] % 120)

    def _preset_swatch_pixmap(self, meta: dict, preset: dict | None, size: QSize, detailed: bool = False) -> QPixmap:
        width = max(32, int(size.width()))
        height = max(32, int(size.height()))
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("#10151d"))
        accent = self._preset_accent_color(meta)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#2f3947"), 1))
        painter.setBrush(QColor("#151d27"))
        painter.drawRoundedRect(0, 0, width - 1, height - 1, 7, 7)
        painter.setPen(Qt.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(6, 6, max(8, width // 6), height - 12, 4, 4)

        scope = self._preset_scope_label(preset or {})
        name = str(meta.get("name", "Preset") or "Preset")
        category = str(meta.get("category", "general") or "general")
        painter.setPen(QColor("#f1eee7"))
        if detailed:
            painter.drawText(22, 26, name[:32])
            painter.setPen(QColor("#b8c2ce"))
            painter.drawText(22, 48, category)
            painter.drawText(22, 70, scope)
        elif width >= 90:
            painter.drawText(22, 28, name[:18])
            painter.setPen(QColor("#b8c2ce"))
            painter.drawText(22, 48, scope[:18])
        else:
            initials = "".join(part[:1] for part in name.split()[:2]).upper() or "P"
            painter.setPen(QColor("#f1eee7"))
            painter.drawText(pixmap.rect().adjusted(10, 0, 0, 0), Qt.AlignCenter, initials[:2])
        painter.end()
        return pixmap

    def _set_preset_preview(self, preset_path):
        if not hasattr(self, "preset_browser_thumbnail"):
            return
        if not preset_path:
            self.preset_browser_thumbnail.setVisible(False)
            self.preset_meta_label.setVisible(False)
            return
        self.preset_browser_thumbnail.setVisible(True)
        thumb_path = self._preset_thumbnail_path(preset_path)
        if thumb_path is None or not thumb_path.exists():
            try:
                preset = self._load_preset_file(preset_path) if preset_path else {}
            except Exception:
                preset = {}
            meta = self._normalize_preset_meta(preset.get("meta") if isinstance(preset, dict) else {}, path=str(preset_path or ""))
            self.preset_browser_thumbnail.setText("")
            self.preset_browser_thumbnail.setPixmap(self._preset_swatch_pixmap(meta, preset, QSize(320, 120), detailed=True))
            return
        pixmap = QPixmap(str(thumb_path))
        if pixmap.isNull():
            try:
                preset = self._load_preset_file(preset_path) if preset_path else {}
            except Exception:
                preset = {}
            meta = self._normalize_preset_meta(preset.get("meta") if isinstance(preset, dict) else {}, path=str(preset_path or ""))
            self.preset_browser_thumbnail.setText("")
            self.preset_browser_thumbnail.setPixmap(self._preset_swatch_pixmap(meta, preset, QSize(320, 120), detailed=True))
            return
        scaled = pixmap.scaled(320, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preset_browser_thumbnail.setText("")
        self.preset_browser_thumbnail.setPixmap(scaled)

    def _preset_list_icon(self, entry):
        preset_path = entry["path"]
        thumb_path = self._preset_thumbnail_path(preset_path)
        compact = bool(getattr(self, "_preset_browser_compact", False))
        pixmap = QPixmap(str(thumb_path)) if thumb_path is not None and thumb_path.exists() else QPixmap()
        if pixmap.isNull():
            size = QSize(42, 42) if compact else QSize(112, 76)
            pixmap = self._preset_swatch_pixmap(entry["meta"], entry.get("preset") or {}, size, detailed=False)
        else:
            size = QSize(42, 42) if compact else QSize(140, 96)
            pixmap = pixmap.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        return QIcon(pixmap)

    def _load_preset_file(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _filtered_preset_entries(self):
        search = self.preset_search.text().strip().lower() if hasattr(self, "preset_search") else ""
        category = self._preset_category_filter if hasattr(self, "preset_category_combo") else "all"
        entries = []
        for entry in self._preset_library_entries:
            meta = entry["meta"]
            if category != "all" and meta.get("category", "general") != category:
                continue
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
            entries.append(entry)
        return entries

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
        if not hasattr(self, "preset_list"):
            return
        categories = ["all"]
        categories.extend(sorted({entry["meta"].get("category", "general") for entry in entries}))
        current_category = self._preset_category_filter
        self.preset_category_combo.blockSignals(True)
        self.preset_category_combo.clear()
        self.preset_category_combo.addItems(categories)
        idx = self.preset_category_combo.findText(current_category)
        self.preset_category_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.preset_category_combo.blockSignals(False)
        self._preset_category_filter = self.preset_category_combo.currentText()
        visible_entries = self._filtered_preset_entries()
        self._configure_preset_list_layout(visible_entries)
        self.preset_list.clear()
        compact = bool(getattr(self, "_preset_browser_compact", False))
        for entry in visible_entries:
            meta = entry["meta"]
            is_recent = str(entry["path"].resolve()) in self._recent_preset_paths
            prefix = "★ " if is_recent else ""
            scope = self._preset_scope_label(entry.get("preset") or {})
            if compact:
                text = f"{prefix}{meta.get('name', entry['path'].stem)}  ·  {meta.get('category', 'general')}  ·  {scope}"
            else:
                text = f"{prefix}{meta.get('name', entry['path'].stem)}\n{meta.get('category', 'general')}"
            item = QListWidgetItem(
                self._preset_list_icon(entry),
                text,
            )
            tags = ", ".join(meta.get("tags", [])) or "none"
            includes = self._preset_summary_text(entry.get("preset") or {})
            item.setToolTip(
                f"{meta.get('name', entry['path'].stem)}\n"
                f"Category: {meta.get('category', 'general')}\n"
                f"Affects: {scope}\n"
                f"Includes: {includes}\n"
                f"Tags: {tags}\n"
                f"Saved: {meta.get('saved_at') or 'unknown'}"
            )
            self.preset_list.addItem(item)
        self._refresh_recent_presets()
        self._save_browser_state()
        self._update_selected_preset_meta()

    def _refresh_recent_presets(self):
        if not hasattr(self, "recent_preset_list"):
            return
        self.recent_preset_list.clear()
        for path_str in self._recent_preset_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                preset = self._load_preset_file(path)
            except Exception:
                preset = {}
            meta = self._normalize_preset_meta(preset.get("meta"), path=str(path))
            self.recent_preset_list.addItem(f"{meta.get('name', path.stem)} [{meta.get('category', 'general')}]")

    def _selected_browser_entry(self):
        if not hasattr(self, "preset_list"):
            return None
        row = self.preset_list.currentRow()
        entries = self._filtered_preset_entries()
        if row < 0 or row >= len(entries):
            return None
        return entries[row]

    def _selected_recent_path(self):
        if not hasattr(self, "recent_preset_list"):
            return None
        row = self.recent_preset_list.currentRow()
        paths = [path for path in self._recent_preset_paths if Path(path).exists()]
        if row < 0 or row >= len(paths):
            return None
        return paths[row]

    def _params_differ_from_default(self, values, layer):
        if not isinstance(values, dict):
            return False
        defaults = {key: default for key, _label, _mn, _mx, default in ALL_LAYERS.get(layer, [])}
        for key, default in defaults.items():
            if key in values and int(round(float(values[key]))) != int(default):
                return True
        return False

    def _preset_scope_label(self, preset):
        """Describe whether a preset affects Global edits, Portrait layers, or both."""
        if not isinstance(preset, dict):
            return "unknown"
        global_active = self._params_differ_from_default(preset.get("global_params", {}), "global")
        color_settings = preset.get("color_settings", {})
        if isinstance(color_settings, dict) and color_settings:
            if color_settings != self._default_color_settings():
                global_active = True

        portrait_active = False
        selective = preset.get("selective_params", {})
        if isinstance(selective, dict):
            for layer in MASK_ORDER:
                if self._params_differ_from_default(selective.get(layer, {}), layer):
                    portrait_active = True
                    break
        if not portrait_active:
            layer_options = preset.get("layer_options", {})
            if isinstance(layer_options, dict) and any(layer_options.get(layer) for layer in MASK_ORDER):
                portrait_active = True

        if global_active and portrait_active:
            return "Global + Portrait"
        if global_active:
            return "Global only"
        if portrait_active:
            return "Portrait only"
        return "No adjustments"

    def _update_selected_preset_meta(self):
        if not hasattr(self, "preset_meta_label"):
            return
        entry = self._selected_browser_entry()
        recent_path = self._selected_recent_path()
        if entry is None and recent_path:
            try:
                preset = self._load_preset_file(recent_path)
            except Exception:
                preset = {}
            meta = self._normalize_preset_meta(preset.get("meta"), path=recent_path)
            self.preset_meta_label.setVisible(True)
            self.preset_meta_label.setText(
                f"Name: {meta.get('name', Path(recent_path).stem)}\n"
                f"Category: {meta.get('category', 'general')}\n"
                f"Affects: {self._preset_scope_label(preset)}\n"
                f"Includes: {self._preset_summary_text(preset)}\n"
                f"Tags: {', '.join(meta.get('tags', [])) or 'none'}\n"
                f"Saved: {meta.get('saved_at') or 'unknown'}\n"
                f"Path: {recent_path}"
            )
            self._set_preset_preview(recent_path)
            return
        if entry is None:
            self.preset_meta_label.setVisible(False)
            self._set_preset_preview("")
            return
        self.preset_meta_label.setVisible(True)
        meta = entry["meta"]
        tags = ", ".join(meta.get("tags", [])) or "none"
        saved_at = meta.get("saved_at") or "unknown"
        scope = self._preset_scope_label(entry.get("preset") or {})
        self.preset_meta_label.setText(
            f"Name: {meta.get('name', entry['path'].stem)}\n"
            f"Category: {meta.get('category', 'general')}\n"
            f"Affects: {scope}\n"
            f"Includes: {self._preset_summary_text(entry.get('preset') or {})}\n"
            f"Tags: {tags}\n"
            f"Saved: {saved_at}\n"
            f"Path: {entry['path']}"
        )
        self._set_preset_preview(entry["path"])

    def _apply_preset_with_preview(self, preset: dict, name: str, remember_path=None):
        """Apply a preset live on the canvas without committing it yet -- shows a Keep/Discard
        bar so a preset that doesn't suit this particular photo can be backed out with one
        click instead of relying on remembering to hit Undo."""
        if self.preview_array is None:
            if self._hydrate_active_image_if_needed(
                "preset",
                on_complete=lambda preset=copy.deepcopy(preset), name=name, remember_path=remember_path: (
                    self._apply_preset_with_preview(preset, name, remember_path=remember_path)
                ),
                preserve_current_settings=True,
            ):
                self.statusBar().showMessage(f"Loading source before applying preset -> {name}")
                return
            QMessageBox.information(self, "Preset", "Open an image first.")
            return
        previous_color_settings = dict(self._color_settings)
        self._preset_preview_snapshot = self._capture_document_state()
        self._pending_preset_preview_meta = None
        self._apply_preset_state(preset, reload_raw_decode=True)
        if remember_path is not None:
            payload = self._preset_to_clipboard_payload(preset, preset_name=name, preset_path=remember_path)
            self._pending_preset_preview_meta = copy.deepcopy(self._read_preset_meta(payload))
        raw_decode_changed = self._raw_decode_settings_changed(previous_color_settings)
        self._preset_preview_requires_raw_reload = False
        if remember_path is not None:
            self._remember_recent_preset(str(remember_path))
        self.preset_preview_label.setText(f"Previewing preset: {name}")
        self.preset_preview_bar.setVisible(True)
        reload_hint = "; RAW source is re-decoding" if raw_decode_changed else ""
        self.statusBar().showMessage(f"Preset preview -> {name} (Keep or Discard above{reload_hint})")

    def _keep_preset_preview(self):
        self.preset_preview_bar.setVisible(False)
        requires_raw_reload = bool(getattr(self, "_preset_preview_requires_raw_reload", False))
        if isinstance(self._pending_preset_preview_meta, dict):
            self._active_preset_meta = copy.deepcopy(self._pending_preset_preview_meta)
        self._pending_preset_preview_meta = None
        self._preset_preview_snapshot = None
        self._push_document_history()
        self._save_current_image_override()
        self._preset_preview_requires_raw_reload = False
        if requires_raw_reload:
            self._reload_source_for_decode_settings()
            self.statusBar().showMessage("Preset kept; re-decoding RAW with preset settings")
        else:
            self.statusBar().showMessage("Preset kept")

    def _discard_preset_preview(self):
        self.preset_preview_bar.setVisible(False)
        if self._preset_preview_snapshot is not None:
            self._restore_document_state(self._preset_preview_snapshot)
            self._preset_preview_snapshot = None
        self._pending_preset_preview_meta = None
        self._preset_preview_requires_raw_reload = False
        self.statusBar().showMessage("Preset discarded")

    def _apply_selected_browser_preset(self):
        entry = self._selected_browser_entry()
        if entry is None:
            QMessageBox.information(self, "Preset Browser", "Select a preset from the library first.")
            return
        try:
            preset = entry.get("preset") or self._load_preset_file(entry["path"])
            scope = self.preset_apply_scope_combo.currentData() if hasattr(self, "preset_apply_scope_combo") else "current"
            if scope == "collection":
                if not self._active_collection or self._active_collection not in self._collections:
                    QMessageBox.information(self, "No Collection", "Select a collection first.")
                    return
                images = list(self._collections[self._active_collection].get("images", []))
                self._apply_preset_payload_to_paths(preset, entry["path"].name, entry["path"], images, scope_label="collection")
            elif scope == "selected":
                selected = self._selected_filmstrip_paths()
                if len(selected) <= 1:
                    QMessageBox.information(self, "Preset Browser", "Select two or more filmstrip images first, or choose Current image.")
                    return
                self._apply_preset_payload_to_paths(preset, entry["path"].name, entry["path"], selected, scope_label="selected")
            else:
                self._apply_preset_with_preview(preset, entry["path"].name, remember_path=entry["path"])
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))

    def _apply_selected_recent_preset(self):
        path = self._selected_recent_path()
        if not path:
            QMessageBox.information(self, "Recent Presets", "Select a recent preset first.")
            return
        try:
            preset = self._load_preset_file(path)
            self._apply_preset_with_preview(preset, Path(path).name, remember_path=path)
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))

    def _clear_preset_edits_for_scope(self):
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        overrides = collection.setdefault("image_overrides", {})
        scope = self.preset_apply_scope_combo.currentData() if hasattr(self, "preset_apply_scope_combo") else "current"
        if scope == "collection":
            paths = list(collection.get("images", []))
            scope_label = "collection"
        elif scope == "selected":
            paths = self._selected_filmstrip_paths()
            scope_label = "selected"
        else:
            paths = [self.file_path] if self.file_path else []
            scope_label = "current"
        paths = [p for p in paths if p in overrides and isinstance(overrides.get(p), dict) and self._read_preset_meta(overrides[p])]
        if not paths:
            QMessageBox.information(self, "Clear Preset Edits", "No preset-applied image overrides found for this scope.")
            return
        reply = QMessageBox.question(
            self,
            "Clear Preset Edits",
            f"Clear saved preset-applied edits from {len(paths)} {scope_label} image(s)?\n\n"
            "Image files are not changed.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        active_cleared = self.file_path in paths
        for path in paths:
            overrides.pop(path, None)
            thumb = self._filmstrip_items.get(path)
            if thumb is not None:
                thumb.set_has_override(False)
                thumb.set_preset_state("", "")
        self._save_collections_state()
        if active_cleared and self.file_path:
            path = self.file_path
            self._active_preset_meta = None
            self._load_image_path(path)
        self.statusBar().showMessage(f"Cleared preset edits from {len(paths)} image(s)")

    def _save_preset_to_library(self):
        preset_dir = self._preset_library_dir()
        name = f"{Path(self.file_path).stem}_preset" if self.file_path else "preset"
        out_path = preset_dir / f"{name}.pepreset"
        try:
            preset = self._serialize_preset_state()
            preset["meta"]["name"] = Path(out_path).stem
            preset["meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
            self._stamp_preset_meta(preset)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self._save_preset_thumbnail(out_path)
            self._remember_recent_preset(str(out_path))
            self._refresh_preset_browser()
            self.statusBar().showMessage(f"Preset saved -> {out_path.name} ({self._preset_summary_text(preset)})")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Save Error", self._friendly_error_message(ex))

    def _rename_selected_browser_preset(self):
        entry = self._selected_browser_entry()
        if entry is None:
            QMessageBox.information(self, "Preset Browser", "Select a preset from the library first.")
            return
        new_name, ok = QInputDialog.getText(self, "Rename Preset", "Preset Name", text=entry["meta"].get("name", entry["path"].stem))
        new_name = str(new_name).strip()
        if not ok or not new_name:
            return
        out_path = entry["path"].with_name(f"{new_name}.pepreset")
        if out_path == entry["path"]:
            return
        if out_path.exists():
            QMessageBox.warning(self, "Preset Rename", "A preset with that name already exists.")
            return
        try:
            preset = entry.get("preset") or self._load_preset_file(entry["path"])
            preset.setdefault("meta", {})
            preset["meta"]["name"] = new_name
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            old_thumb = self._preset_thumbnail_path(entry["path"])
            new_thumb = self._preset_thumbnail_path(out_path)
            if old_thumb.exists():
                old_thumb.replace(new_thumb)
            entry["path"].unlink(missing_ok=True)
            self._remember_recent_preset(str(out_path))
            self._refresh_preset_browser()
            self.statusBar().showMessage(f"Preset renamed -> {out_path.name}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Rename Error", self._friendly_error_message(ex))

    def _delete_selected_browser_preset(self):
        entry = self._selected_browser_entry()
        if entry is None:
            QMessageBox.information(self, "Preset Browser", "Select a preset from the library first.")
            return
        decision = QMessageBox.question(
            self,
            "Delete Preset",
            f"Delete preset '{entry['path'].name}' from the library?",
        )
        if decision != QMessageBox.Yes:
            return
        try:
            thumb_path = self._preset_thumbnail_path(entry["path"])
            entry["path"].unlink(missing_ok=True)
            thumb_path.unlink(missing_ok=True)
            self._recent_preset_paths = [p for p in self._recent_preset_paths if Path(p) != entry["path"]]
            self._refresh_preset_browser()
            self._save_browser_state()
            self.statusBar().showMessage(f"Preset deleted -> {entry['path'].name}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Delete Error", self._friendly_error_message(ex))

    def _on_preset_category_changed(self, category: str):
        self._preset_category_filter = str(category or "all")
        self._set_section_expanded("Preset Browser", True)
        self._save_browser_state()
        self._refresh_preset_browser()

    def eventFilter(self, obj, event):
        preset_widgets = {
            getattr(self, "preset_search", None),
            getattr(self, "preset_category_combo", None),
            getattr(self, "recent_preset_list", None),
            getattr(self, "preset_list", None),
        }
        if obj in preset_widgets and event.type() in {QEvent.FocusIn, QEvent.MouseButtonPress}:
            self._set_section_expanded("Preset Browser", True)
        return super().eventFilter(obj, event)

    def _recipe_to_preset(self, recipe: dict) -> dict:
        preset = {
            "version": 1,
            "global_params": dict(recipe.get("preset", {}).get("global_params", {})),
            "color_settings": dict(self._color_settings),
            "selective_params": {layer: {} for layer in MASK_ORDER},
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "meta": {
                "name": recipe.get("name", "Recipe"),
                "category": "recipe",
                "tags": ["guided", "recipe"],
                "saved_at": "",
            },
        }
        preset["selective_params"].update(recipe.get("preset", {}).get("selective_params", {}))
        return preset

    def open_recipe_dialog(self):
        dialog = RecipeDialog(self._guided_recipes(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        recipe = dialog.selected_recipe()
        if not recipe:
            return
        self._apply_recipe_with_preview(recipe)

    def _apply_recipe_with_preview(self, recipe: dict):
        """Apply a guided recipe through the same live-preview-then-Keep/Discard flow the
        preset browser already uses (_apply_preset_with_preview) -- recipes used to call
        _apply_preset_state directly with no matching _push_document_history, which left no
        clean undo checkpoint for "just the recipe." Routing through here both fixes that and
        gives every recipe the same try-it-risk-free behavior for free."""
        name = recipe.get("name", "Recipe")
        self._apply_preset_with_preview(self._recipe_to_preset(recipe), name)

    def _reload_models_and_readiness(self):
        """Re-check callback: rebuild the segmenter so models added while the app is running
        (e.g. a freshly downloaded MODNet) are actually detected, then return fresh readiness.
        Without this, Re-check only re-reads the segmenter built at startup and a newly added
        model keeps showing as 'not installed' until a full restart."""
        previous = QApplication.overrideCursor()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.segmenter = FaceSegmenter()
            self._wb_estimator = wb_ops.WhiteBalanceEstimator()
            self._denoiser = MLDenoiser()
            self._deep_denoiser = DeepDenoiser()
        except Exception:
            pass
        finally:
            QApplication.restoreOverrideCursor()
            if previous is not None:
                QApplication.setOverrideCursor(previous)
        return self._readiness_items()

    def _update_readiness_warning(self, items, has_issues: bool):
        if not hasattr(self, "readiness_warning_btn"):
            return
        if not has_issues:
            self.readiness_warning_btn.setVisible(False)
            return
        count = sum(1 for it in items if it.get("status") != "ready")
        self.readiness_warning_btn.setText(f"! {count} readiness issue(s) -- click for details")
        self.readiness_warning_btn.setVisible(True)

    def show_system_check(self):
        items, has_issues, models_dir = self._readiness_items()
        self._update_readiness_warning(items, has_issues)
        ReadinessDialog(
            items, self, title="System Check",
            recheck_callback=self._reload_models_and_readiness, models_dir=models_dir,
        ).exec()
        items, has_issues, _models_dir = self._readiness_items()
        self._update_readiness_warning(items, has_issues)

    def _maybe_show_startup_readiness(self):
        state = self._browser_state_payload()
        items, has_issues, models_dir = self._readiness_items()
        self._update_readiness_warning(items, has_issues)
        if not has_issues and state.get("readiness_seen"):
            return
        ReadinessDialog(
            items, self, title="Startup Readiness",
            recheck_callback=self._reload_models_and_readiness, models_dir=models_dir,
        ).exec()
        state["readiness_seen"] = True
        self._write_browser_state(state)
        items, has_issues, _models_dir = self._readiness_items()
        self._update_readiness_warning(items, has_issues)

    def _on_left_panel_toggled(self, visible: bool):
        self._nav_panel.setVisible(bool(visible))

    def _on_right_panel_toggled(self, visible: bool):
        self._inspector_scroll.setVisible(bool(visible))

    def _set_view_mode(self, mode: str):
        mode = str(mode or "edit").lower()
        if mode not in {"edit", "cull", "batch", "compare", "focus"}:
            mode = "edit"
        if mode == "focus":
            self._view_mode = "focus"
            self._sync_view_mode_buttons()
            if not self._focus_action.isChecked():
                self._focus_action.setChecked(True)
            else:
                self._on_focus_mode_toggled(True)
            return
        if self._focus_action.isChecked():
            self._focus_action.setChecked(False)
        self._view_mode = mode
        self._apply_view_mode()

    def _sync_view_mode_buttons(self):
        buttons = getattr(self, "_view_mode_buttons", {})
        if not buttons:
            return
        active = str(getattr(self, "_view_mode", "edit") or "edit")
        for key, button in buttons.items():
            button.blockSignals(True)
            button.setChecked(key == active)
            button.blockSignals(False)

    def _apply_view_mode(self):
        mode = str(getattr(self, "_view_mode", "edit") or "edit")
        if mode == "focus":
            self._sync_view_mode_buttons()
            return
        if self._focus_mode:
            return
        self._sync_view_mode_buttons()

        if self._mode_restore_panels is None:
            self._mode_restore_panels = (
                self._left_panel_action.isChecked(),
                self._right_panel_action.isChecked(),
            )

        left_default, right_default = self._mode_restore_panels
        left_visible = left_default
        right_visible = right_default
        # 162 is the real minimum to show the collection header + filter row + the 98px-tall
        # thumbnails without clipping (see filmstrip_frame construction). Mode-specific values
        # stay >= that floor so no mode re-introduces the clipped-filmstrip bug.
        filmstrip_height = 162
        status_visible = True
        message = ""

        if mode == "cull":
            left_visible = False
            right_visible = False
            filmstrip_height = 200
            message = "Cull mode: filmstrip and collection review emphasized"
        elif mode == "batch":
            left_visible = True
            right_visible = False
            filmstrip_height = 184
            message = "Batch mode: collection export controls emphasized"
        elif mode == "compare":
            left_visible = False
            right_visible = False
            filmstrip_height = 162
            message = "Compare mode: before/after controls emphasized"
        else:
            self._mode_restore_panels = None
            left_visible = self._left_panel_action.isChecked()
            right_visible = self._right_panel_action.isChecked()

        self._nav_panel.setVisible(bool(left_visible))
        self._inspector_scroll.setVisible(bool(right_visible))
        self._status_panel.setVisible(status_visible)
        self._canvas_header.setVisible(True)
        if hasattr(self, "_filmstrip_frame"):
            self._filmstrip_frame.setMaximumHeight(filmstrip_height)
            self._filmstrip_frame.setVisible(mode != "compare" or bool(self._filmstrip_images))
        if mode != "edit" and message:
            self.statusBar().showMessage(message)

    def _on_focus_mode_toggled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._focus_mode:
            return
        self._focus_mode = enabled
        if enabled:
            self._view_mode = "focus"
            self._sync_view_mode_buttons()
        if enabled:
            # Remember the side-panel state so exiting focus restores it.
            self._focus_restore = (
                self._nav_panel.isVisible(),
                self._inspector_scroll.isVisible(),
            )
            if self._main_toolbar is not None:
                self._main_toolbar.setVisible(False)
            self._canvas_header.setVisible(False)
            self._status_panel.setVisible(False)
            self._nav_panel.setVisible(False)
            self._inspector_scroll.setVisible(False)
        else:
            left, right = self._focus_restore or (True, True)
            if self._main_toolbar is not None:
                self._main_toolbar.setVisible(True)
            self._canvas_header.setVisible(True)
            self._status_panel.setVisible(True)
            self._left_panel_action.setChecked(left)
            self._right_panel_action.setChecked(right)
            self._nav_panel.setVisible(left)
            self._inspector_scroll.setVisible(right)
            self._focus_restore = None
            if self._view_mode == "focus":
                self._view_mode = "edit"
            self._apply_view_mode()

    def _activate_layer(self, layer: str):
        if layer == "global":
            # Global develop lives in the Adjust tab now (split into Light & Tone / Color /
            # Detail / Effects sections), not as a layer tab -- surface it by switching there.
            self._active_layer = "global"
            self._sync_quick_layer_buttons()
            if getattr(self, "_inspector_tabs", None) is not None:
                self._inspector_tabs.setCurrentIndex(0)  # Adjust
            self._refresh_mask_controls()
            self._update_preview_label()
            return
        if layer not in self._tab_layers:
            return
        self._active_layer = layer
        self._set_section_expanded("Layers", True)
        index = self._tab_layers.index(layer)
        if self.layer_tabs.currentIndex() != index:
            self.layer_tabs.setCurrentIndex(index)
        self._sync_quick_layer_buttons()
        self._refresh_mask_controls()
        self._update_preview_label()

    def _sync_quick_layer_buttons(self):
        for layer, button in getattr(self, "_quick_layer_buttons", {}).items():
            button.blockSignals(True)
            button.setChecked(layer == self._active_layer)
            button.blockSignals(False)

    def _on_essentials_only_toggled(self, checked: bool):
        if bool(checked) != self._essentials_only:
            self._begin_document_change()
        self._essentials_only = bool(checked)
        self._apply_essentials_filter()

    def _apply_essentials_filter(self):
        # Superseded by the Basic/All view toggle (Basic = curated quick panel; All = every
        # slider). Always show all slider blocks in the full tabs, regardless of any persisted
        # essentials_only value, so loading an older project can't leave sliders hidden.
        for blocks in self._slider_blocks.values():
            for block in blocks.values():
                block.setVisible(True)

    def _reset_active_layer(self):
        self._begin_document_change()
        layer = self._active_layer
        sliders = ALL_LAYERS.get(layer, [])
        if not sliders:
            return
        defaults = {key: default for key, _label, _mn, _mx, default in sliders}
        for key, slider in self._sliders.get(layer, {}).items():
            slider.blockSignals(True)
            slider.setValue(int(defaults[key]))
            slider.blockSignals(False)
            self._slider_value_labels[layer][key].setText(f"{int(defaults[key]):+d}" if int(defaults[key]) else "0")
        if layer in MASK_ORDER:
            self._mask_adjustments[layer] = default_mask_adjustments([layer])[layer]
            self._sync_mask_adjustment_controls()
        self._schedule_render()

    def _schedule_render(self):
        # Any document-mutating edit invalidates a cached hi-res tile -- it was rendered from
        # the params/masks/framing as of the last request, which this edit just changed.
        self.image_label.set_hires_tile(None, None)
        if self.preview_array is None:
            self._hydrate_active_image_if_needed("edit", preserve_current_settings=True)
            return
        if self._slider_drag_active > 0 and self._render_timer.isActive():
            # During an active drag, let the already-running timer fire on its
            # ~60ms cadence so the preview updates live. Restarting it on every
            # valueChanged (as below) would coalesce them and starve updates until
            # the drag pauses or is released, which reads as "not responding".
            return
        self._render_timer.start()

    def _get_preset_render_params(self, preset):
        params = {"global": dict(preset.get("global_params", {}))}
        selective = preset.get("selective_params", {})
        for layer in MASK_ORDER:
            params[layer] = dict(selective.get(layer, {}))
        return params

    def _get_preset_color_settings(self, preset):
        merged = self._default_color_settings()
        color_settings = preset.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged.update(color_settings)
        return merged

    def _supported_image_paths(self, input_dir):
        paths = []
        if not input_dir or not os.path.isdir(input_dir):
            return paths
        for name in sorted(os.listdir(input_dir)):
            path = os.path.join(input_dir, name)
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

    def _batch_output_path(self, source_path, output_dir, suffix, output_ext):
        stem = Path(source_path).stem
        return os.path.join(output_dir, f"{stem}{suffix}{output_ext}")

    def _preset_hash(self, preset):
        payload = json.dumps(preset, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _batch_config_key(self, preset_hash, suffix, output_format):
        return json.dumps({"preset_hash": preset_hash, "suffix": suffix, "format": output_format}, sort_keys=True)

    def _utc_timestamp(self):
        return datetime.now(timezone.utc).isoformat()

    def _append_batch_log(self, log_path, record):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _load_batch_log_records(self, log_path):
        records = []
        if not os.path.exists(log_path):
            return records
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    def _load_batch_log_index(self, log_path):
        index = {}
        for record in self._load_batch_log_records(log_path):
            if record.get("status") != "success":
                continue
            key = (
                record.get("batch_key"),
                os.path.abspath(record.get("source_path", "")),
                os.path.abspath(record.get("output_path", "")),
            )
            index[key] = record
        return index

    def _batch_should_skip(self, completed_index, batch_key, source_path, output_path):
        key = (batch_key, os.path.abspath(source_path), os.path.abspath(output_path))
        return key in completed_index and os.path.exists(output_path)

    def _latest_failed_batch_records(self, records):
        groups = {}
        for record in records:
            if record.get("status") not in {"error", "success", "skipped"}:
                continue
            batch_key = record.get("batch_key")
            if not batch_key:
                continue
            groups.setdefault(batch_key, []).append(record)
        failed_groups = []
        for batch_key, group in groups.items():
            if any(r.get("status") == "error" for r in group):
                failed_groups.append((max(r.get("ts", "") for r in group), batch_key, group))
        if not failed_groups:
            return None
        failed_groups.sort(key=lambda item: item[0], reverse=True)
        return failed_groups[0][2]

    def _write_batch_job_file(self, payload, output_dir):
        jobs_dir = Path(output_dir) / ".batch_jobs"
        jobs_dir.mkdir(exist_ok=True)
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        job_id = payload.get("job_id") or f"job_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        payload = dict(payload)
        payload["job_id"] = job_id
        payload.setdefault("created_at", created_at)
        job_path = jobs_dir / f"{job_id}.json"
        payload.setdefault("cancel_path", f"{job_path}.cancel")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        return str(job_path)

    def _update_batch_job_file(self, job_path, updates):
        try:
            with open(job_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return
            payload.update(updates)
            with open(job_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True)
        except Exception:
            return

    def _append_batch_log_record(self, log_path, record):
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            pass

    def _launch_background_batch_job(self, job_path, output_dir):
        runner_log = os.path.join(output_dir, "batch_runner_stdout.log")
        with open(runner_log, "ab") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-m", "portrait_enhancer.batch_runner", "--job", job_path],
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                cwd=os.getcwd(),
            )
        self._update_batch_job_file(
            job_path,
            {
                "runner_pid": int(proc.pid),
                "runner_pgid": int(proc.pid),
                "runner_started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            },
        )
        return runner_log

    def _process_is_running(self, pid):
        try:
            os.kill(int(pid), 0)
            try:
                state = subprocess.check_output(["ps", "-o", "stat=", "-p", str(int(pid))], text=True).strip()
                if state.startswith("Z"):
                    return False
            except Exception:
                pass
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    def _find_batch_runner_pid_for_job(self, job_path):
        if not job_path:
            return None
        try:
            output = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
        except Exception:
            return None
        for line in output.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            pid_text, command = parts
            if "portrait_enhancer.batch_runner" in command and str(job_path) in command:
                try:
                    return int(pid_text)
                except ValueError:
                    continue
        return None

    def _batch_job_runner_pid(self, job):
        pid = job.get("runner_pid")
        if pid and self._process_is_running(pid):
            return int(pid)
        payload = self._read_batch_job_payload(job.get("job_path", ""))
        if isinstance(payload, dict):
            pid = payload.get("runner_pid")
            if pid and self._process_is_running(pid):
                return int(pid)
        return self._find_batch_runner_pid_for_job(job.get("job_path", ""))

    def _batch_cancel_marker_path(self, job):
        payload = self._read_batch_job_payload(job.get("job_path", ""))
        if isinstance(payload, dict) and payload.get("cancel_path"):
            return str(payload["cancel_path"])
        if job.get("job_path"):
            return f"{job['job_path']}.cancel"
        return ""

    def _request_batch_job_cancel(self, job, log_path, pid):
        job_id = str(job.get("job_id", "") or "unknown")
        cancel_path = self._batch_cancel_marker_path(job)
        if cancel_path:
            try:
                Path(cancel_path).parent.mkdir(parents=True, exist_ok=True)
                with open(cancel_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "job_id": job_id,
                            "reason": "user_canceled",
                            "requested_at": datetime.now(timezone.utc).isoformat(),
                            "runner_pid": pid,
                        },
                        fh,
                        sort_keys=True,
                    )
            except Exception:
                pass
        self._append_batch_log_record(
            log_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "job_mode": job.get("mode", "batch"),
                "status": "cancel_requested",
                "reason": "user_canceled",
                "runner_pid": pid,
            },
        )
        if job.get("job_path"):
            self._update_batch_job_file(
                job["job_path"],
                {
                    "cancel_requested": True,
                    "cancel_requested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "cancel_reason": "user_canceled",
                },
            )

    def _cancel_batch_job(self, job):
        job_id = str(job.get("job_id", "") or "unknown")
        pid = self._batch_job_runner_pid(job)
        log_path = job.get("log_path") or os.path.join(job.get("output_dir", ""), "batch_export_log.jsonl")
        self._request_batch_job_cancel(job, log_path, pid)
        if pid:
            for _ in range(20):
                if not self._process_is_running(pid):
                    break
                QApplication.processEvents()
                time.sleep(0.1)
            try:
                if self._process_is_running(pid):
                    os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pid = None
            except Exception:
                try:
                    if self._process_is_running(pid):
                        os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pid = None
                except Exception as ex:
                    return False, f"Could not cancel export process {pid}: {ex}"

            if pid:
                for _ in range(20):
                    if not self._process_is_running(pid):
                        break
                    QApplication.processEvents()
                    time.sleep(0.1)
                if self._process_is_running(pid):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except Exception:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except Exception:
                            pass

        self._append_batch_log_record(
            log_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "job_mode": job.get("mode", "batch"),
                "status": "canceled",
                "reason": "user_canceled",
                "runner_pid": pid,
            },
        )
        if job.get("job_path"):
            self._update_batch_job_file(
                job["job_path"],
                {
                    "canceled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "cancel_reason": "user_canceled",
                },
            )
        if getattr(self, "_export_monitors", None):
            self._export_monitors.pop(job_id, None)
        return True, f"Export job {job_id} was canceled."

    def _start_export_monitor(self, job_id, output_dir, source_count, label="", source_paths=None):
        """Poll the batch log for a just-launched job and surface a completion summary
        (succeeded / skipped / failed) when it finishes -- so export errors no longer
        vanish silently into batch_export_log.jsonl. Live progress shows in the status bar."""
        if not hasattr(self, "_export_monitors") or self._export_monitors is None:
            self._export_monitors = {}
        log_path = os.path.join(output_dir, "batch_export_log.jsonl")
        self._export_monitors[job_id] = {
            "output_dir": output_dir,
            "log_path": log_path,
            "source_count": int(source_count),
            "source_paths": [os.path.abspath(str(p)) for p in (source_paths or [])],
            "label": label or job_id,
            "started_at": time.time(),
        }
        if not hasattr(self, "_export_monitor_timer") or self._export_monitor_timer is None:
            self._export_monitor_timer = QTimer(self)
            self._export_monitor_timer.setInterval(2000)
            self._export_monitor_timer.timeout.connect(self._poll_export_monitors)
        if not self._export_monitor_timer.isActive():
            self._export_monitor_timer.start()

    def _export_job_counts(self, log_path, job_id):
        success = skipped = error = canceled = 0
        failures = []
        job_records = []
        status_by_path = {}
        for record in self._load_batch_log_records(log_path):
            if str(record.get("job_id", "")) != job_id:
                continue
            job_records.append(record)
            status = record.get("status")
            source_path = record.get("source_path")
            if source_path:
                if status in {"success", "skipped"}:
                    status_by_path[os.path.abspath(str(source_path))] = "done"
                elif status == "error":
                    status_by_path[os.path.abspath(str(source_path))] = "failed"
                elif status == "canceled":
                    status_by_path[os.path.abspath(str(source_path))] = "queued"
            if status == "success":
                success += 1
            elif status == "skipped":
                skipped += 1
            elif status == "error":
                error += 1
                failures.append(record)
            elif status == "canceled":
                canceled += 1
        return success, skipped, error, canceled, failures, self._latest_deep_denoise_progress(job_records), status_by_path

    def _poll_export_monitors(self):
        if not getattr(self, "_export_monitors", None):
            if getattr(self, "_export_monitor_timer", None):
                self._export_monitor_timer.stop()
            return
        for job_id in list(self._export_monitors.keys()):
            state = self._export_monitors[job_id]
            success, skipped, error, canceled, failures, denoise_progress, status_by_path = self._export_job_counts(state["log_path"], job_id)
            done = state["source_count"] if canceled else success + skipped + error
            total = state["source_count"]
            label = state["label"]
            self._sync_filmstrip_export_status(state.get("source_paths", []), status_by_path, running=done < total)
            if done < total:
                msg = f"Exporting {label}: {done}/{total}"
                if denoise_progress:
                    tile_done = int(denoise_progress.get("deep_denoise_done") or 0)
                    tile_total = int(denoise_progress.get("deep_denoise_total") or 0)
                    if tile_total > 0:
                        msg += f" | Deep Denoise {tile_done}/{tile_total} tiles"
                if error:
                    msg += f" ({error} failed)"
                self.statusBar().showMessage(msg + "…")
                continue
            # Finished -- stop tracking this job and report.
            self._export_monitors.pop(job_id, None)
            self._sync_filmstrip_export_status(state.get("source_paths", []), status_by_path, running=False)
            self._show_export_summary(label, success, skipped, error, failures, state["output_dir"], canceled=canceled)
        if not self._export_monitors and getattr(self, "_export_monitor_timer", None):
            self._export_monitor_timer.stop()

    def _show_export_summary(self, label, success, skipped, error, failures, output_dir, canceled=0):
        if failures:
            self._mark_filmstrip_export_failures(failures)
        parts = [f"{success} exported"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if error:
            parts.append(f"{error} failed")
        if canceled:
            parts.append("canceled")
        summary = ", ".join(parts)
        verb = "canceled" if canceled else "finished"
        self.statusBar().showMessage(f"Export of {label} {verb} — {summary}")

        box = QMessageBox(self)
        box.setWindowTitle("Export Canceled" if canceled else "Export Complete")
        box.setIcon(QMessageBox.Warning if error or canceled else QMessageBox.Information)
        box.setText(f"Export of {label} {verb}.\n\n{summary}.")
        if error and failures:
            lines = []
            for record in failures[:20]:
                name = os.path.basename(record.get("source_path", "") or "?")
                reason = (record.get("error", "") or "").strip().splitlines()
                reason = reason[0] if reason else "unknown error"
                lines.append(f"• {name}: {reason}")
            if len(failures) > 20:
                lines.append(f"… and {len(failures) - 20} more.")
            box.setDetailedText("\n".join(lines))
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

    def _set_filmstrip_export_status(self, paths, status: str):
        status = status if status in {"queued", "exporting", "done", "failed"} else ""
        normalized = {os.path.abspath(str(path)) for path in (paths or []) if path}
        if not normalized:
            return
        for path in normalized:
            if status:
                self._filmstrip_export_status[path] = status
            else:
                self._filmstrip_export_status.pop(path, None)
            if status == "failed":
                self._filmstrip_failed_paths.add(path)
            elif status in {"queued", "exporting", "done"}:
                self._filmstrip_failed_paths.discard(path)
        for thumb_path, thumb in self._filmstrip_items.items():
            normalized_thumb = os.path.abspath(str(thumb_path))
            if normalized_thumb in normalized:
                thumb.set_export_status(self._filmstrip_export_status.get(normalized_thumb, ""))
        self._update_filmstrip_filter_buttons()

    def _sync_filmstrip_export_status(self, source_paths, status_by_path, running: bool):
        normalized_sources = [os.path.abspath(str(path)) for path in (source_paths or []) if path]
        for path in normalized_sources:
            status = status_by_path.get(path)
            if status is None:
                status = "exporting" if running else ""
            self._set_filmstrip_export_status([path], status)

    def _mark_filmstrip_export_failures(self, failures):
        paths = set()
        for record in failures or []:
            path = record.get("source_path") or record.get("path") or record.get("input_path")
            if path:
                paths.add(os.path.abspath(str(path)))
        if not paths:
            return
        self._set_filmstrip_export_status(paths, "failed")
        for path, thumb in self._filmstrip_items.items():
            thumb.set_export_failed(os.path.abspath(str(path)) in self._filmstrip_failed_paths)
        self._update_filmstrip_filter_buttons()
        if self._filmstrip_filter == "failed":
            self._populate_filmstrip()

    def _read_batch_job_payload(self, job_path):
        try:
            with open(job_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _latest_worker_plan_record(self, job_records):
        for record in reversed(job_records):
            if record.get("status") == "worker_plan":
                return record
        return {}

    def _latest_deep_denoise_progress(self, job_records):
        for record in reversed(job_records):
            if record.get("status") == "deep_denoise_progress":
                return record
        return {}

    def _system_resource_snapshot(self):
        snapshot = {}
        try:
            output = subprocess.check_output(
                ["memory_pressure"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=3,
            )
            lower = output.lower()
            if "no memory pressure" in lower:
                snapshot["memory_pressure"] = "normal"
            elif "critical" in lower:
                snapshot["memory_pressure"] = "critical"
            elif "warn" in lower or "pressure" in lower:
                snapshot["memory_pressure"] = "warning"
            for line in output.splitlines():
                if "System-wide memory free percentage:" in line:
                    try:
                        snapshot["memory_free_percent"] = int(line.rsplit(":", 1)[1].strip().rstrip("%"))
                    except Exception:
                        pass
                    break
        except Exception:
            pass
        try:
            page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
            output = subprocess.check_output(["vm_stat"], text=True, timeout=3)
            pages = {}
            for line in output.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                digits = "".join(ch for ch in value if ch.isdigit())
                if digits:
                    pages[key.strip()] = int(digits)
            reclaimable = (
                pages.get("Pages free", 0)
                + pages.get("Pages inactive", 0)
                + pages.get("Pages speculative", 0)
                + pages.get("Pages purgeable", 0)
            )
            if reclaimable > 0:
                snapshot["available_memory_gb"] = reclaimable * page_size / (1024 ** 3)
        except Exception:
            pass
        return snapshot

    def _batch_job_process_usage(self, payload, job_path):
        pid = payload.get("runner_pid") if isinstance(payload, dict) else None
        pgid = payload.get("runner_pgid") if isinstance(payload, dict) else None
        if not pid:
            pid = self._find_batch_runner_pid_for_job(str(job_path))
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return {}
        try:
            pgid = int(pgid) if pgid else pid
        except (TypeError, ValueError):
            pgid = pid
        try:
            output = subprocess.check_output(
                ["ps", "-o", "pid=,ppid=,pgid=,%cpu=,%mem=,rss=,stat=,command=", "-ax"],
                text=True,
                timeout=3,
            )
        except Exception:
            return {}

        processes = []
        for line in output.splitlines():
            parts = line.strip().split(None, 7)
            if len(parts) < 8:
                continue
            try:
                proc_pid = int(parts[0])
                proc_ppid = int(parts[1])
                proc_pgid = int(parts[2])
                cpu = float(parts[3])
                rss_mb = float(parts[5]) / 1024.0
            except (TypeError, ValueError):
                continue
            if proc_pid != pid and proc_ppid != pid and proc_pgid != pgid:
                continue
            command = parts[7]
            if "portrait_enhancer.batch_runner" in command:
                role = "runner"
            elif "multiprocessing" in command or "spawn_main" in command:
                role = "worker"
            else:
                role = "helper"
            processes.append(
                {
                    "pid": proc_pid,
                    "role": role,
                    "cpu_percent": cpu,
                    "rss_mb": rss_mb,
                    "state": parts[6],
                }
            )

        if not processes:
            return {}
        return {
            "process_count": len(processes),
            "cpu_percent": sum(proc["cpu_percent"] for proc in processes),
            "rss_mb": sum(proc["rss_mb"] for proc in processes),
            "processes": sorted(processes, key=lambda proc: proc["rss_mb"], reverse=True),
        }

    def _load_batch_jobs(self, output_dir):
        jobs_dir = Path(output_dir) / ".batch_jobs"
        log_path = os.path.join(output_dir, "batch_export_log.jsonl")
        runner_log_path = os.path.join(output_dir, "batch_runner_stdout.log")
        records = self._load_batch_log_records(log_path)
        system_resource = self._system_resource_snapshot()
        by_job_id = {}
        for record in records:
            job_id = str(record.get("job_id", "")).strip()
            if not job_id:
                continue
            by_job_id.setdefault(job_id, []).append(record)

        jobs = []
        if jobs_dir.exists():
            for job_path in sorted(jobs_dir.glob("*.json"), reverse=True):
                payload = self._read_batch_job_payload(job_path)
                if not isinstance(payload, dict):
                    continue
                job_id = str(payload.get("job_id", job_path.stem))
                job_records = by_job_id.get(job_id, [])
                success = sum(1 for r in job_records if r.get("status") == "success")
                skipped = sum(1 for r in job_records if r.get("status") == "skipped")
                error = sum(1 for r in job_records if r.get("status") == "error")
                canceled = sum(1 for r in job_records if r.get("status") == "canceled")
                source_paths = payload.get("source_paths") or []
                if source_paths:
                    source_count = len(source_paths)
                else:
                    source_count = len(self._supported_image_paths(payload.get("input_dir", "")))
                status = "queued"
                if job_records:
                    if canceled:
                        status = "canceled"
                    elif error:
                        status = "error"
                    elif success + skipped >= source_count and source_count > 0:
                        status = "complete"
                    else:
                        status = "running"
                resource_usage = (
                    self._batch_job_process_usage(payload, job_path)
                    if status in {"queued", "running"} or payload.get("runner_pid")
                    else {}
                )
                worker_plan = self._latest_worker_plan_record(job_records)
                deep_denoise_progress = self._latest_deep_denoise_progress(job_records)
                jobs.append(
                    {
                        "job_id": job_id,
                        "mode": payload.get("mode", "batch"),
                        "created_at": payload.get("created_at", ""),
                        "preset_path": payload.get("preset_path", ""),
                        "input_dir": payload.get("input_dir", ""),
                        "output_dir": payload.get("output_dir", output_dir),
                        "suffix": payload.get("suffix", ""),
                        "output_format": payload.get("output_format", ""),
                        "skip_completed": bool(payload.get("skip_completed", False)),
                        "source_count": source_count,
                        "success": success,
                        "skipped": skipped,
                        "error": error,
                        "canceled": canceled,
                        "status": status,
                        "latest_error": next((r.get("error") for r in reversed(job_records) if r.get("status") == "error"), ""),
                        "runner_pid": payload.get("runner_pid", ""),
                        "runner_pgid": payload.get("runner_pgid", ""),
                        "resource_usage": resource_usage,
                        "system_resource": system_resource,
                        "worker_plan": worker_plan,
                        "deep_denoise_progress": deep_denoise_progress,
                        # Every failure, not just the most recent -- a job that fails for five
                        # different reasons across five images should not hide four of them.
                        "error_details": [
                            {"source_path": r.get("source_path", ""), "error": r.get("error", "")}
                            for r in job_records
                            if r.get("status") == "error"
                        ],
                        "job_path": str(job_path),
                        "log_path": log_path,
                        "runner_log_path": runner_log_path,
                    }
                )
        return jobs

    def _build_preview_proxy(self, full: np.ndarray):
        h, w = full.shape[:2]
        max_dim = 1600
        scale = min(1.0, float(max_dim) / float(max(h, w)))
        if scale >= 0.999:
            return full.copy(), 1.0
        preview = resize_image(full, (max(1, int(w * scale)), max(1, int(h * scale))), acceleration="auto")
        return preview, scale

    def _build_interactive_preview_proxy(self, preview: np.ndarray):
        h, w = preview.shape[:2]
        max_dim = 900
        scale = min(1.0, float(max_dim) / float(max(h, w)))
        if scale >= 0.999:
            return preview.copy(), 1.0
        interactive = resize_image(preview, (max(1, int(w * scale)), max(1, int(h * scale))), acceleration="auto")
        return interactive, scale

    def _scale_masks(self, masks, shape_hw):
        return _scale_masks_data(masks, shape_hw)

    def _on_face_changed(self, index: int):
        if self.preview_array is None:
            return
        if max(0, index) != self._active_face_index:
            self._begin_document_change()
        self._store_active_face_profile()
        self._active_face_index = max(0, index)
        self._update_face_scope_indicator()
        # If this face was already visited earlier in this session, its masks/sliders/mask
        # edits are sitting in _face_profiles -- restore directly instead of dispatching a
        # new segmentation run (even a disk-cache hit there costs a thread dispatch + ~150-200ms;
        # this is a plain dict copy).
        if self._restore_face_profile(self._active_face_index):
            self.statusBar().showMessage(f"Face {self._active_face_index + 1}: restored (already computed this session)")
            self._perf_stats["detect_ms"] = None
            self._perf_stats["segment_ms"] = None
            self._update_perf_label()
            self._refresh_mask_controls()
            self._schedule_render()
            if self._document_history_index < 0:
                self._push_document_history()
            return
        self._run_segmentation()

    def _face_rect_after_framing(self, box, full_w, full_h, framing):
        """Map a face box (x,y,w,h in full-resolution source pixels) through the same
        flip/straighten/crop pipeline _update_preview_label applies for display -- without
        this, the highlight stays at the pre-transform position the moment any crop/
        straighten/flip is applied, no longer lining up with the face at all. Rotation turns
        the box into a non-axis-aligned quadrilateral; we draw the bounding box of its
        rotated corners rather than an exact rotated rectangle, close enough for an
        indicator. Returns None if the box ends up entirely outside the visible crop."""
        fx, fy, fw, fh = box
        corners = [(fx, fy), (fx + fw, fy), (fx, fy + fh), (fx + fw, fy + fh)]

        if framing.get("flip_h"):
            corners = [(full_w - x, y) for x, y in corners]
        if framing.get("flip_v"):
            corners = [(x, full_h - y) for x, y in corners]

        angle = float(framing.get("angle", 0.0) or 0.0)
        if abs(angle) >= 1e-6:
            cx, cy = full_w / 2.0, full_h / 2.0
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rotated = []
            for x, y in corners:
                dx, dy = x - cx, y - cy
                rotated.append((cx + dx * cos_t + dy * sin_t, cy - dx * sin_t + dy * cos_t))
            corners = rotated

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bx0, bx1 = min(xs), max(xs)
        by0, by1 = min(ys), max(ys)

        crop_x, crop_y, crop_w, crop_h = framing.get("crop", [0.0, 0.0, 1.0, 1.0])
        crop_x0, crop_y0 = crop_x * full_w, crop_y * full_h
        crop_w_px, crop_h_px = crop_w * full_w, crop_h * full_h
        if crop_w_px <= 0 or crop_h_px <= 0:
            return None

        rx0 = (bx0 - crop_x0) / crop_w_px
        ry0 = (by0 - crop_y0) / crop_h_px
        rx1 = (bx1 - crop_x0) / crop_w_px
        ry1 = (by1 - crop_y0) / crop_h_px
        if rx1 <= 0.0 or ry1 <= 0.0 or rx0 >= 1.0 or ry0 >= 1.0:
            return None  # rotated/cropped entirely out of view
        rx0, ry0 = max(0.0, rx0), max(0.0, ry0)
        rx1, ry1 = min(1.0, rx1), min(1.0, ry1)
        return (rx0, ry0, rx1 - rx0, ry1 - ry0)

    def _update_face_scope_indicator(self):
        """Keep the persistent 'Editing: ...' label, the canvas face-highlight box, and the
        canvas's face click-targets in sync with the Face Target combo -- the Face/Skin/Eyes/
        Lips/Person tabs act on whichever face this reflects, so losing track of it should not
        be possible, and clicking any other face on the canvas should retarget to it directly."""
        if not hasattr(self, "face_scope_label"):
            return
        if hasattr(self, "apply_to_all_faces_btn"):
            self.apply_to_all_faces_btn.setVisible(len(self._detected_faces) > 1)
        if not self._detected_faces:
            self.face_scope_label.setText("Editing: Auto (whole scene)")
            self.image_label.set_face_highlight(None)
            self.image_label.set_face_click_targets([], None)
            return
        idx = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        self.face_scope_label.setText(f"Editing: Face {idx + 1} of {len(self._detected_faces)}")
        # _detected_faces boxes come from segmenting self.preview_array (SegmentationTask
        # runs list_faces on the preview, not the full-resolution array), so they're in
        # preview-pixel space -- normalizing against full_array's (larger) dimensions instead
        # was the actual bug behind the highlight not landing on the face at all.
        preview = self.preview_array
        if preview is None or preview.shape[0] <= 0 or preview.shape[1] <= 0:
            self.image_label.set_face_highlight(None)
            self.image_label.set_face_click_targets([], None)
            return
        h, w = preview.shape[:2]
        framing = self._display_framing()
        rect = self._face_rect_after_framing(self._detected_faces[idx], w, h, framing)
        self.image_label.set_face_highlight(rect)
        targets = []
        for face_index, box in enumerate(self._detected_faces):
            face_rect = self._face_rect_after_framing(box, w, h, framing)
            if face_rect is not None:
                targets.append((face_index, *face_rect))
        self.image_label.set_face_click_targets(targets, self._on_face_clicked_on_canvas)

    def _on_face_clicked_on_canvas(self, face_index: int):
        if not self._detected_faces:
            return
        idx = int(np.clip(face_index, 0, len(self._detected_faces) - 1))
        if idx == self._active_face_index:
            return  # already targeted -- nothing to do, avoid a no-op status message
        self.face_combo.setCurrentIndex(idx)
        self.statusBar().showMessage(f"Editing: Face {idx + 1} of {len(self._detected_faces)}")

    def _apply_active_face_to_all(self):
        """Group-portrait fan-out: copy the active face's Face/Skin/Eyes/Lips/Hair/Person
        adjustment values onto every other detected face, uniformly (same values, not
        per-face-adapted -- the predictable behavior users expect from "apply to all").
        Each face keeps its own mask; only the slider values are copied. Faces not yet
        visited this session get a placeholder profile (mirroring the existing paste-settings
        path) -- _restore_face_profile already applies stored selective_params before checking
        whether masks need a fresh segmentation, so this is safe even for unvisited faces."""
        if len(self._detected_faces) <= 1:
            return
        per_face_layers = tuple(layer for layer in MASK_ORDER if layer not in ("background", "subjects"))
        self._begin_document_change()  # snapshots active face's current live state first
        active_idx = self._active_face_index
        source = self._face_profiles.get(active_idx, {}).get("selective_params", {})
        source_values = {layer: dict(source.get(layer, {})) for layer in per_face_layers}
        other_count = 0
        for idx in range(len(self._detected_faces)):
            if idx == active_idx:
                continue
            other_count += 1
            if idx not in self._face_profiles:
                self._face_profiles[idx] = {
                    "selective_params": {layer: {} for layer in MASK_ORDER},
                    "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
                    "layer_order": list(self._layer_order),
                    "mask_adjustments": {},
                    "preview_masks": {},
                    "full_masks": {},
                    "auto_preview_masks": {},
                    "auto_full_masks": {},
                    "preview_guides": {},
                    "full_guides": {},
                }
            for layer in per_face_layers:
                self._face_profiles[idx]["selective_params"][layer] = dict(source_values[layer])
        self._push_document_history()
        plural = "" if other_count == 1 else "s"
        self.statusBar().showMessage(
            f"Applied Face {active_idx + 1}'s adjustments to {other_count} other face{plural} "
            "(Ctrl+Z to undo)."
        )

    def _on_zoom_changed(self, zoom: float):
        if not hasattr(self, "zoom_indicator"):
            return
        if zoom <= 1.0 + 1e-6:
            self.zoom_indicator.setText("Fit")
        else:
            self.zoom_indicator.setText(f"{int(round(zoom * 100))}%")

    def _on_slider_drag_started(self):
        self._slider_drag_active += 1

    def _on_slider_drag_finished(self):
        self._slider_drag_active = max(0, self._slider_drag_active - 1)
        self._schedule_render()

    def _on_layer_changed(self, index: int):
        self._active_layer = self._tab_layers[max(0, min(index, len(self._tab_layers) - 1))]
        self._sync_quick_layer_buttons()
        self._refresh_mask_controls()
        self._update_preview_label()

    def _on_mask_view_toggled(self, checked: bool):
        changed = bool(checked) != self._show_mask
        if changed:
            self._begin_document_change()
        self._show_mask = bool(checked)
        self._update_mask_debug_label()
        self._update_preview_label()
        if changed:
            self._push_document_history()

    def _on_sharpen_mask_preview_toggled(self, checked: bool):
        self._show_sharpen_mask_preview = bool(checked)
        if not self._show_sharpen_mask_preview:
            self._sharpen_mask_preview = None
            self._update_preview_label()
            self.statusBar().showMessage("Sharpen Mask preview off")
            return
        self._sharpen_mask_preview = None
        self._schedule_render()
        self.statusBar().showMessage("Sharpen Mask preview on: white = sharpened, black = protected")

    def _on_mask_edit_toggled(self, checked: bool):
        self._mask_edit_enabled = bool(checked)
        if self._mask_edit_enabled:
            if self._crop_edit_enabled and hasattr(self, "crop_edit_btn"):
                self.crop_edit_btn.setChecked(False)  # mutually exclusive edit modes
            if self._wb_pick_enabled and hasattr(self, "wb_pick_btn"):
                self.wb_pick_btn.setChecked(False)
            if self._click_mask_enabled and hasattr(self, "click_mask_btn"):
                self.click_mask_btn.setChecked(False)
            if self._denoise_point_enabled and hasattr(self, "denoise_point_btn"):
                self.denoise_point_btn.setChecked(False)
            self._set_section_expanded("Masks", True)
        self._refresh_mask_controls()
        self._update_preview_label()

    def _sync_framing_controls(self):
        """Push the current framing state into the Geometry controls and preview."""
        if not hasattr(self, "straighten_slider"):
            return
        framing = normalize_framing(self._framing)
        self._framing = framing
        self.straighten_slider.blockSignals(True)
        self.straighten_slider.setValue(int(round(framing["angle"])))
        self.straighten_slider.blockSignals(False)
        self.straighten_value_label.setText(f"{int(round(framing['angle']))}°")
        if hasattr(self, "crop_edit_btn"):
            self.crop_edit_btn.blockSignals(True)
            self.crop_edit_btn.setChecked(self._crop_edit_enabled)
            self.crop_edit_btn.blockSignals(False)
        self._update_preview_label()

    def _source_aspect(self) -> float | None:
        if self.preview_array is None:
            return None
        h, w = self.preview_array.shape[:2]
        if h <= 0:
            return None
        return float(w) / float(h)

    def _on_crop_edit_toggled(self, checked: bool):
        self._crop_edit_enabled = bool(checked)
        if self._crop_edit_enabled:
            if self._mask_edit_enabled and hasattr(self, "mask_edit_btn"):
                self.mask_edit_btn.setChecked(False)  # mutually exclusive edit modes
            if self._wb_pick_enabled and hasattr(self, "wb_pick_btn"):
                self.wb_pick_btn.setChecked(False)
            if self._click_mask_enabled and hasattr(self, "click_mask_btn"):
                self.click_mask_btn.setChecked(False)
            if self._denoise_point_enabled and hasattr(self, "denoise_point_btn"):
                self.denoise_point_btn.setChecked(False)
            self._set_section_expanded("Geometry", True)
        self._update_preview_label()

    def _on_aspect_preset_changed(self, index: int):
        if index < 0 or index >= len(self._aspect_presets):
            return
        _label, value = self._aspect_presets[index]
        if value == "original":
            aspect = self._source_aspect()
        else:
            aspect = value
        self._begin_document_change()
        width = height = 1
        if self.preview_array is not None:
            height, width = self.preview_array.shape[:2]
        self._framing["crop"] = framing_ops.centered_crop_for_aspect(aspect, width, height)
        if hasattr(self, "crop_lock_check") and self.crop_lock_check.isChecked():
            # Re-lock to whichever ratio was just picked, rather than leaving drag-resize
            # enforcing the *previous* preset's ratio.
            self.image_label.set_crop_aspect_lock(aspect)
        self._update_preview_label()
        self._push_document_history()

    def _on_crop_lock_toggled(self, checked: bool):
        if not checked:
            self.image_label.set_crop_aspect_lock(None)
            return
        idx = self.aspect_combo.currentIndex() if hasattr(self, "aspect_combo") else -1
        aspect = None
        if 0 <= idx < len(self._aspect_presets):
            _label, value = self._aspect_presets[idx]
            aspect = self._source_aspect() if value == "original" else value
        if not aspect:
            # "Free" selected (or no preset yet) -- lock to the crop box's own current
            # true-pixel shape rather than doing nothing, so Lock is useful even without
            # first picking a named ratio.
            crop = self._framing.get("crop")
            canvas_aspect = self._source_aspect()
            if crop and canvas_aspect:
                _x, _y, w, h = crop
                if h > 0:
                    aspect = (w / h) * canvas_aspect
        self.image_label.set_crop_aspect_lock(aspect)

    def _on_straighten_changed(self, value: int):
        self.straighten_value_label.setText(f"{int(value)}°")
        if abs(float(value) - float(self._framing.get("angle", 0.0))) < 1e-6:
            return
        self._begin_document_change()
        self._framing["angle"] = float(value)
        self._update_preview_label()

    def _on_flip(self, key: str):
        self._begin_document_change()
        self._framing[key] = not bool(self._framing.get(key, False))
        self._update_preview_label()
        self._push_document_history()

    def _reset_framing(self):
        if framing_ops.is_identity(self._framing):
            return
        self._begin_document_change()
        self._framing = default_framing()
        self._sync_framing_controls()
        self._sync_wb_controls()
        self._sync_tone_curve_controls()
        self._sync_hsl_controls()
        self._push_document_history()

    def _on_crop_drag_start(self):
        self._begin_document_change()

    def _on_crop_changed(self, rect):
        self._framing["crop"] = list(rect)

    def _on_crop_committed(self):
        self._push_document_history()

    # --- White balance ---------------------------------------------------

    def _current_wb_temp_k(self):
        return wb_ops.clamp_temp(self._color_settings.get("wb_temp_k", wb_ops.NEUTRAL_K))

    def _current_wb_tint(self):
        return wb_ops.clamp_tint(self._color_settings.get("wb_tint", 0))

    def _matching_wb_preset(self, temp_k, tint):
        for name, ptemp, ptint in wb_ops.PRESETS:
            if int(ptemp) == int(temp_k) and int(ptint) == int(tint):
                return name
        return "Custom"

    def _wb_status_text(self, temp_k, tint):
        if temp_k == wb_ops.NEUTRAL_K and tint == 0:
            return "White balance: neutral"
        return f"White balance: {int(temp_k)}K, tint {int(tint):+d}"

    def _sync_raw_decode_controls(self):
        """Keep the RAW Decode section visible, greying its (RAW-only) controls for non-RAW
        images, and mirror the combos/checkboxes to the document's color settings without
        triggering a re-decode."""
        if self._raw_decode_section is None or self._raw_decode_content is None:
            return
        is_raw = bool(self.file_path) and is_raw_path(self.file_path)
        # Greying the content widget disables every child at once; the section header stays.
        self._raw_decode_content.setEnabled(is_raw)
        self._syncing_raw_controls = True
        try:
            for key, combo in (
                ("raw_white_balance", self.raw_wb_combo),
                ("raw_colorspace", self.raw_colorspace_combo),
                ("raw_highlight_mode", self.raw_highlight_combo),
                ("raw_demosaic", self.raw_demosaic_combo),
            ):
                idx = combo.findData(self._color_settings.get(key))
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            scene_linear_on = self._color_settings.get("working_space") == "linear"
            self.raw_scene_linear_check.setChecked(scene_linear_on)
            self.raw_scene_linear_denoise_check.setChecked(bool(self._color_settings.get("scene_linear_denoise", False)))
            self.raw_learned_denoise_check.setChecked(bool(self._color_settings.get("use_learned_denoise", True)))
            # Mutually-exclusive engines: VST denoise needs scene-linear; the learned denoiser
            # is the sRGB-mode engine. (Only effective when the section is enabled for RAW.)
            self.raw_scene_linear_denoise_check.setEnabled(is_raw and scene_linear_on)
            self.raw_learned_denoise_check.setEnabled(is_raw and not scene_linear_on)
        finally:
            self._syncing_raw_controls = False

    def _on_scene_linear_toggled(self, checked):
        if self._syncing_raw_controls:
            return
        new_space = "linear" if checked else "srgb"
        # The VST luma denoise only applies in scene-linear mode (see _scene_linear_denoise);
        # the learned denoiser is the sRGB-mode engine, so the two enabled states are inverse.
        self.raw_scene_linear_denoise_check.setEnabled(bool(checked))
        self.raw_learned_denoise_check.setEnabled(not checked)
        if self._color_settings.get("working_space") == new_space:
            return
        # A render-path change, not a decode change: update the working space and re-render
        # (the stage-cache digest includes color_settings, so this recomputes from the warp).
        self._begin_document_change()
        self._color_settings["working_space"] = new_space
        self._schedule_render()
        self._push_document_history()

    def _on_scene_linear_denoise_toggled(self, checked):
        if self._syncing_raw_controls:
            return
        if self._color_settings.get("scene_linear_denoise", False) == bool(checked):
            return
        self._begin_document_change()
        self._color_settings["scene_linear_denoise"] = bool(checked)
        self._schedule_render()
        self._push_document_history()

    def _on_learned_denoise_toggled(self, checked):
        if self._syncing_raw_controls:
            return
        if self._color_settings.get("use_learned_denoise", True) == bool(checked):
            return
        self._begin_document_change()
        self._color_settings["use_learned_denoise"] = bool(checked)
        self._schedule_render()
        self._push_document_history()

    def _on_raw_decode_combo_changed(self, key, combo):
        if self._syncing_raw_controls:
            return
        value = combo.currentData()
        if value is None or self._color_settings.get(key) == value:
            return
        self._color_settings[key] = value
        self._reload_source_for_decode_settings()

    def _reload_source_for_decode_settings(self):
        """Re-decode the current RAW with the updated decode settings while preserving the
        edit state. Reuses the project-state round-trip: the snapshot carries the new
        color_settings, the decode worker reads them, and _apply_project_state restores every
        adjustment (and the existing masks) onto the freshly decoded pixels."""
        if not self.file_path or not is_raw_path(self.file_path) or self.preview_array is None:
            return False
        snapshot = self._serialize_project_state()
        self.statusBar().showMessage(f"Re-decoding {os.path.basename(self.file_path)} ...")
        self._start_image_decode_task(self.file_path, project_state=snapshot)
        return True

    def _sync_wb_controls(self):
        if not hasattr(self, "wb_status_label"):
            return
        temp_k, tint = self._current_wb_temp_k(), self._current_wb_tint()
        self.wb_temp_slider.blockSignals(True)
        self.wb_temp_slider.setValue(int(temp_k))
        self.wb_temp_slider.blockSignals(False)
        self.wb_temp_value_label.setText(f"{int(temp_k)}K")
        self.wb_tint_slider.blockSignals(True)
        self.wb_tint_slider.setValue(int(tint))
        self.wb_tint_slider.blockSignals(False)
        self.wb_tint_value_label.setText(f"{int(tint):+d}" if tint else "0")
        if hasattr(self, "wb_pick_btn"):
            self.wb_pick_btn.blockSignals(True)
            self.wb_pick_btn.setChecked(self._wb_pick_enabled)
            self.wb_pick_btn.blockSignals(False)
        if hasattr(self, "wb_preset_combo"):
            self.wb_preset_combo.blockSignals(True)
            self.wb_preset_combo.setCurrentText(self._matching_wb_preset(temp_k, tint))
            self.wb_preset_combo.blockSignals(False)
        gains = wb_ops.kelvin_tint_to_gains(temp_k, tint)
        self.wb_status_label.setText(self._wb_status_text(temp_k, tint))
        self.wb_status_label.setToolTip(f"Gain R/G/B: {gains[0]:.2f}, {gains[1]:.2f}, {gains[2]:.2f}")

    def _set_wb_kelvin_tint(self, temp_k, tint):
        """Set the canonical (Kelvin, tint) WB and re-render (pick/auto/preset)."""
        self._begin_document_change()
        self._color_settings["wb_temp_k"] = wb_ops.clamp_temp(temp_k)
        self._color_settings["wb_tint"] = wb_ops.clamp_tint(tint)
        self._sync_wb_controls()
        self._schedule_render()  # WB is applied inside process_global, so re-render
        self._push_document_history()

    def _on_wb_temp_changed(self, value: int):
        self._color_settings["wb_temp_k"] = wb_ops.clamp_temp(value)
        self.wb_temp_value_label.setText(f"{int(value)}K")
        self._sync_wb_status_only()
        self._schedule_render()

    def _on_wb_tint_changed(self, value: int):
        self._color_settings["wb_tint"] = wb_ops.clamp_tint(value)
        self.wb_tint_value_label.setText(f"{int(value):+d}" if value else "0")
        self._sync_wb_status_only()
        self._schedule_render()

    def _on_wb_preset_chosen(self, index: int):
        if index <= 0:  # "Custom"
            return
        name, temp_k, tint = wb_ops.PRESETS[index - 1]
        self._set_wb_kelvin_tint(temp_k, tint)

    def _sync_wb_status_only(self):
        temp_k, tint = self._current_wb_temp_k(), self._current_wb_tint()
        gains = wb_ops.kelvin_tint_to_gains(temp_k, tint)
        self.wb_status_label.setText(self._wb_status_text(temp_k, tint))
        self.wb_status_label.setToolTip(f"Gain R/G/B: {gains[0]:.2f}, {gains[1]:.2f}, {gains[2]:.2f}")
        if hasattr(self, "wb_preset_combo"):
            self.wb_preset_combo.blockSignals(True)
            self.wb_preset_combo.setCurrentText(self._matching_wb_preset(temp_k, tint))
            self.wb_preset_combo.blockSignals(False)

    def _on_wb_pick_toggled(self, checked: bool):
        self._wb_pick_enabled = bool(checked)
        if self._wb_pick_enabled:
            if self._crop_edit_enabled and hasattr(self, "crop_edit_btn"):
                self.crop_edit_btn.setChecked(False)
            if self._mask_edit_enabled and hasattr(self, "mask_edit_btn"):
                self.mask_edit_btn.setChecked(False)
            if self._click_mask_enabled and hasattr(self, "click_mask_btn"):
                self.click_mask_btn.setChecked(False)
            if self._denoise_point_enabled and hasattr(self, "denoise_point_btn"):
                self.denoise_point_btn.setChecked(False)
            self._set_section_expanded("White Balance", True)
        self.image_label.set_wb_pick_state(self._wb_pick_enabled, self._on_wb_picked)
        self._update_preview_label()

    def _on_click_mask_toggled(self, checked: bool):
        self._click_mask_enabled = bool(checked)
        if self._click_mask_enabled:
            if self._crop_edit_enabled and hasattr(self, "crop_edit_btn"):
                self.crop_edit_btn.setChecked(False)  # mutually exclusive edit modes
            if self._mask_edit_enabled and hasattr(self, "mask_edit_btn"):
                self.mask_edit_btn.setChecked(False)
            if self._wb_pick_enabled and hasattr(self, "wb_pick_btn"):
                self.wb_pick_btn.setChecked(False)
            if self._denoise_point_enabled and hasattr(self, "denoise_point_btn"):
                self.denoise_point_btn.setChecked(False)
            self._set_section_expanded("Masks", True)
        self.image_label.set_click_mask_state(self._click_mask_enabled, self._on_object_clicked)
        self._update_preview_label()

    def _on_denoise_point_toggled(self, checked: bool):
        self._denoise_point_enabled = bool(checked)
        if self._denoise_point_enabled:
            if self._crop_edit_enabled and hasattr(self, "crop_edit_btn"):
                self.crop_edit_btn.setChecked(False)  # mutually exclusive edit modes
            if self._mask_edit_enabled and hasattr(self, "mask_edit_btn"):
                self.mask_edit_btn.setChecked(False)
            if self._wb_pick_enabled and hasattr(self, "wb_pick_btn"):
                self.wb_pick_btn.setChecked(False)
            if self._click_mask_enabled and hasattr(self, "click_mask_btn"):
                self.click_mask_btn.setChecked(False)
            # The Denoise Point preview thumbnails live in the Adjust tab's "Auto" section.
            if getattr(self, "_inspector_tabs", None) is not None:
                self._inspector_tabs.setCurrentIndex(0)  # Adjust
            self._set_section_expanded("Auto", True)
            self.statusBar().showMessage("Denoise Point: click a spot on the image to preview a small denoised square there.")
        self.image_label.set_denoise_point_state(self._denoise_point_enabled, self._on_denoise_point_clicked)
        self._update_preview_label()

    def _sample_wb_pixel(self, nx: float, ny: float):
        arr = self.preview_array
        if arr is None:
            return None
        h, w = arr.shape[:2]
        px = int(round(nx * (w - 1)))
        py = int(round(ny * (h - 1)))
        radius = 2  # 5x5 neighborhood to reduce noise
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        region = arr[y0:y1, x0:x1, :3]
        if region.size == 0:
            return None
        return region.reshape(-1, 3).mean(axis=0)

    def _on_wb_picked(self, nx: float, ny: float):
        sample = self._sample_wb_pixel(nx, ny)
        if sample is None:
            return
        # preview_array is in display/source (sRGB) space, so linearize from sRGB.
        # Invert the neutralizing gains to (Kelvin, tint) so the sliders stay in sync.
        temp_k, tint = wb_ops.neutral_sample_to_kelvin_tint(sample, working_space="srgb")
        self._set_wb_kelvin_tint(temp_k, tint)

    def _wb_auto_gray_world(self):
        if self.preview_array is None:
            return
        gains = wb_ops.gray_world_gains(self.preview_array, working_space="srgb")
        self._set_wb_kelvin_tint(*wb_ops.gains_to_kelvin_tint(gains))

    def _wb_auto_white_patch(self):
        if self.preview_array is None:
            return
        gains = wb_ops.white_patch_gains(self.preview_array, working_space="srgb")
        self._set_wb_kelvin_tint(*wb_ops.gains_to_kelvin_tint(gains))

    def _estimate_auto_ai_wb(self, array):
        est = getattr(self, "_wb_estimator", None)
        gains = None
        if est is not None and est.available:
            gains = est.estimate_gains(array, working_space="srgb")
        if gains is None:
            fallback = wb_ops.gray_world_gains(array, working_space="srgb")
            reason = "model not installed" if (est is None or not est.available) else "estimation failed"
            k, t = wb_ops.gains_to_kelvin_tint(fallback)
            return k, t, reason
        k, t = wb_ops.gains_to_kelvin_tint(gains)
        return k, t, ""

    def _apply_wb_to_override_payload(self, payload, temp_k, tint):
        payload = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        color_settings = self._default_color_settings()
        if isinstance(payload.get("color_settings"), dict):
            color_settings.update(payload["color_settings"])
        color_settings["wb_temp_k"] = wb_ops.clamp_temp(temp_k)
        color_settings["wb_tint"] = wb_ops.clamp_tint(tint)
        payload["color_settings"] = color_settings
        return payload

    def _wb_auto_ai_selected_filmstrip_images(self, paths):
        if not self._active_collection or self._active_collection not in self._collections:
            self.statusBar().showMessage("Auto WB (AI): select images from a collection first")
            return
        collection = self._collections[self._active_collection]
        collection_paths = set(collection.get("images", []))
        paths = [path for path in paths if path in collection_paths]
        if len(paths) <= 1:
            return

        self._save_current_image_override()
        # Decode + estimate runs off the GUI thread below; the active image's already-decoded
        # preview is handed in directly (it's read-only here, never mutated), everything else
        # gets decoded fresh inside the worker via the pure (no self/_source_metadata side
        # effect) decode function.
        items = [
            (path, self.preview_array if path == self.file_path and self.preview_array is not None else None)
            for path in paths
        ]

        self.statusBar().showMessage(f"Auto WB (AI): analyzing {len(paths)} selected image(s)...")
        self._wb_auto_ai_job_id += 1
        job_id = self._wb_auto_ai_job_id
        task = WBAutoAITask(job_id, items, self._build_preview_proxy, self._estimate_auto_ai_wb)
        self._wb_auto_ai_task = task
        task.signals.finished.connect(self._on_wb_auto_ai_finished)
        task.signals.failed.connect(self._on_wb_auto_ai_failed)
        self._wb_auto_ai_pool.start(task)

    def _on_wb_auto_ai_finished(self, job_id, results):
        self._wb_auto_ai_task = None
        if job_id != self._wb_auto_ai_job_id:
            return  # superseded by a newer request
        if not self._active_collection or self._active_collection not in self._collections:
            return  # active collection changed/closed while this was analyzing
        collection = self._collections[self._active_collection]
        overrides = collection.setdefault("image_overrides", {})
        applied = 0
        fallback_count = 0
        failures = []
        active_updated = False
        for path, temp_k, tint, fallback_reason, error in results:
            if error is not None:
                failures.append((path, error))
                continue
            if fallback_reason:
                fallback_count += 1
            if path == self.file_path and self.preview_array is not None:
                self._set_wb_kelvin_tint(temp_k, tint)
                overrides[path] = self._build_settings_payload()
                active_updated = True
            else:
                overrides[path] = self._apply_wb_to_override_payload(overrides.get(path), temp_k, tint)
            applied += 1

        self._save_collections_state()
        if active_updated:
            self._schedule_render()
        if applied == 0 and failures:
            QMessageBox.warning(self, "Auto WB (AI)", f"Could not apply Auto WB to {len(failures)} selected image(s).")
            return
        message = f"Auto WB (AI): applied to {applied} selected image(s)"
        if fallback_count:
            message += f" ({fallback_count} used Gray World fallback)"
        if failures:
            message += f"; {len(failures)} failed"
        self.statusBar().showMessage(message)

    def _on_wb_auto_ai_failed(self, job_id, message):
        self._wb_auto_ai_task = None
        if job_id != self._wb_auto_ai_job_id:
            return
        QMessageBox.warning(self, "Auto WB (AI)", self._friendly_error_message(RuntimeError(message)))

    def _wb_auto_ai(self):
        """Learned auto white balance: estimate the scene illuminant with the ONNX model and
        set Temperature/Tint from the neutralizing gains. Falls back to Gray World when the
        model isn't installed or estimation fails, so the button always does something useful."""
        selected_paths = self._selected_filmstrip_paths()
        if len(selected_paths) > 1:
            self._wb_auto_ai_selected_filmstrip_images(selected_paths)
            return
        if self.preview_array is None:
            return
        k, t, reason = self._estimate_auto_ai_wb(self.preview_array)
        self._set_wb_kelvin_tint(k, t)
        if reason:
            self.statusBar().showMessage(f"Auto WB (AI): {reason} — used Gray World fallback")
            return
        self.statusBar().showMessage(f"Auto WB (AI): {k}K, tint {t:+d}")

    def _reset_wb(self):
        self._set_wb_kelvin_tint(wb_ops.NEUTRAL_K, 0)

    # --- Tone curve ------------------------------------------------------

    def _current_tone_curve(self):
        return tc_ops.normalize_curve(self._color_settings.get("tone_curve", tc_ops.default_curve()))

    def _sync_tone_curve_controls(self):
        if not hasattr(self, "tone_curve_widget"):
            return
        self.tone_curve_widget.set_points(self._current_tone_curve())

    def _on_tone_curve_start(self):
        self._begin_document_change()

    def _on_tone_curve_changed(self, points):
        # Reassign (never mutate in place) so history snapshots stay independent.
        self._color_settings["tone_curve"] = [list(p) for p in tc_ops.normalize_curve(points)]
        self._schedule_render()

    def _on_tone_curve_commit(self):
        self._push_document_history()

    def _reset_tone_curve(self):
        if tc_ops.is_identity(self._current_tone_curve()):
            return
        self._begin_document_change()
        self._color_settings["tone_curve"] = [list(p) for p in tc_ops.default_curve()]
        self._sync_tone_curve_controls()
        self._schedule_render()
        self._push_document_history()

    # --- HSL color mixer -------------------------------------------------

    def _current_color_mixer(self):
        return cm_ops.normalize_color_mixer(self._color_settings.get("color_mixer"))

    def _active_hsl_band(self):
        idx = self.hsl_band_combo.currentIndex() if hasattr(self, "hsl_band_combo") else 0
        return cm_ops.BAND_NAMES[max(0, min(len(cm_ops.BAND_NAMES) - 1, idx))]

    def _sync_hsl_controls(self):
        if not hasattr(self, "hsl_sliders"):
            return
        band = self._current_color_mixer()[self._active_hsl_band()]
        for key, slider in self.hsl_sliders.items():
            value = int(band.get(key, 0))
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            self.hsl_value_labels[key].setText(f"{value:+d}" if value else "0")

    def _on_hsl_band_changed(self, _index: int):
        self._sync_hsl_controls()  # load the newly selected band's values (no history)

    def _on_hsl_slider_changed(self, key: str, value: int):
        self.hsl_value_labels[key].setText(f"{int(value):+d}" if value else "0")
        # Reassign a fresh mixer dict so history snapshots stay independent.
        mixer = self._current_color_mixer()
        mixer[self._active_hsl_band()][key] = int(value)
        self._color_settings["color_mixer"] = mixer
        self._schedule_render()

    def _reset_hsl_band(self):
        band = self._active_hsl_band()
        mixer = self._current_color_mixer()
        if all(v == 0 for v in mixer[band].values()):
            return
        self._begin_document_change()
        mixer[band] = {p: 0 for p in cm_ops.PARAMS}
        self._color_settings["color_mixer"] = mixer
        self._sync_hsl_controls()
        self._schedule_render()
        self._push_document_history()

    def _reset_hsl_all(self):
        if cm_ops.is_identity(self._current_color_mixer()):
            return
        self._begin_document_change()
        self._color_settings["color_mixer"] = cm_ops.default_color_mixer()
        self._sync_hsl_controls()
        self._schedule_render()
        self._push_document_history()

    def _on_mask_mode_changed(self, mode: str):
        self._mask_paint_mode = str(mode or "paint")

    def _on_mask_debug_mode_changed(self, mode: str):
        if str(mode or "tint") != self._mask_debug_mode:
            self._begin_document_change()
        self._mask_debug_mode = str(mode or "tint")
        self._update_mask_debug_label()
        self._update_preview_label()

    def _on_guides_toggled(self, checked: bool):
        if bool(checked) != self._show_expression_guides:
            self._begin_document_change()
        self._show_expression_guides = bool(checked)
        self._update_preview_label()

    def _on_brush_size_changed(self, value: int):
        self._mask_brush_size = int(value)
        self.brush_value_label.setText(str(self._mask_brush_size))
        source_size = None
        if self.preview_array is not None:
            source_size = (self.preview_array.shape[1], self.preview_array.shape[0])
        self.image_label.set_brush_preview(
            self._mask_brush_size,
            source_image_size=source_size,
            hardness=self._mask_brush_hardness,
        )

    def _on_brush_hardness_changed(self, value: int):
        self._mask_brush_hardness = int(value)
        self.hardness_value_label.setText(f"{self._mask_brush_hardness}%")
        source_size = None
        if self.preview_array is not None:
            source_size = (self.preview_array.shape[1], self.preview_array.shape[0])
        self.image_label.set_brush_preview(
            self._mask_brush_size,
            source_image_size=source_size,
            hardness=self._mask_brush_hardness,
        )

    def _on_mask_adjustment_changed(self, key: str, value: int):
        if self._active_layer not in MASK_ORDER:
            return
        current = self._mask_adjustments.setdefault(self._active_layer, default_mask_adjustments([self._active_layer])[self._active_layer])
        old_value = int(round(float(current.get(key, 100 if key == "strength" else 0))))
        if int(value) != old_value:
            self._begin_document_change()
        current[key] = float(value)
        if key in self._mask_adjustment_labels:
            self._mask_adjustment_labels[key].setText(self._format_mask_adjustment_value(key, value))
        self._update_mask_debug_label()
        self._update_preview_label()
        self._schedule_render()

    def _reset_active_mask_settings(self):
        if self._active_layer not in MASK_ORDER:
            return
        self._begin_document_change()
        defaults = default_mask_adjustments([self._active_layer])[self._active_layer]
        self._mask_adjustments[self._active_layer] = dict(defaults)
        self._sync_mask_adjustment_controls()
        self._update_mask_debug_label()
        self._update_preview_label()
        self._schedule_render()

    def _sync_mask_adjustment_controls(self):
        editable = self._active_layer in MASK_ORDER and self.preview_masks is not None
        defaults = default_mask_adjustments([self._active_layer])[self._active_layer] if self._active_layer in MASK_ORDER else {}
        settings = normalize_mask_adjustments({self._active_layer: self._mask_adjustments.get(self._active_layer, defaults)}).get(self._active_layer, defaults)
        for key, slider in self._mask_adjustment_sliders.items():
            value = int(round(float(settings.get(key, defaults.get(key, 100 if key == "strength" else 0)))))
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            slider.setEnabled(editable)
            if key in self._mask_adjustment_labels:
                self._mask_adjustment_labels[key].setText(self._format_mask_adjustment_value(key, value))
                self._mask_adjustment_labels[key].setEnabled(editable)
        if hasattr(self, "reset_mask_settings_btn"):
            self.reset_mask_settings_btn.setEnabled(editable)

    def _select_mask_review_layer(self, layer: str) -> bool:
        if layer not in MASK_ORDER:
            return False
        self._activate_layer(layer)
        if self.preview_masks is None or layer not in self.preview_masks:
            self._update_mask_review_panel()
            return False
        if hasattr(self, "mask_view_btn") and not self.mask_view_btn.isChecked():
            self.mask_view_btn.setChecked(True)
        self._set_section_expanded("Masks", True)
        self._update_mask_review_panel()
        return True

    def _review_recalculate_mask(self, layer: str):
        if self._select_mask_review_layer(layer):
            self._recalculate_active_mask()

    def _review_ai_select_mask(self, layer: str):
        if not self._select_mask_review_layer(layer):
            return
        if hasattr(self, "click_mask_btn"):
            self.click_mask_btn.setChecked(True)
        label = LAYER_NAMES.get(layer, layer)
        self.statusBar().showMessage(f"AI Select: click the {label} target on the image.")

    def _review_reset_mask(self, layer: str):
        if self._select_mask_review_layer(layer):
            self._reset_active_mask()

    def _review_edit_mask(self, layer: str):
        if not self._select_mask_review_layer(layer):
            return
        if hasattr(self, "mask_edit_btn"):
            self.mask_edit_btn.setChecked(True)

    def _mask_review_layer_stats(self, layer: str, mask):
        if mask is None:
            return "Pending", "Open an image to generate masks.", "pending"
        arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if arr.size == 0:
            return "Empty", "Mask has no pixels.", "bad"
        max_value = float(np.max(arr))
        cov10 = float(np.mean(arr > 0.10))
        cov50 = float(np.mean(arr > 0.50))
        mean_value = float(np.mean(arr))
        detail = f"mean={mean_value:.2f} max={max_value:.2f} >0.10={cov10*100:.1f}% >0.50={cov50*100:.1f}%"

        edited = False
        if self._auto_preview_masks is not None and layer in self._auto_preview_masks:
            auto = self._auto_preview_masks[layer]
            if getattr(auto, "shape", None) == arr.shape:
                edited = bool(np.max(np.abs(arr - np.asarray(auto, dtype=np.float32))) > 1e-4)

        if max_value < 0.10 or cov10 < 0.0005:
            return "Empty", f"{detail}. Try Recalc, AI Select, or Edit.", "bad"

        face_cov10 = None
        if self.preview_masks is not None and layer != "face" and "face" in self.preview_masks:
            face = np.asarray(self.preview_masks["face"], dtype=np.float32)
            if getattr(face, "shape", None) == arr.shape:
                face_cov10 = float(np.mean(face > 0.10))

        if face_cov10 is not None and face_cov10 > 0.0005:
            relative_to_face = cov10 / face_cov10
            detail = f"{detail} face-relative={relative_to_face:.2f}x"
            if layer == "skin" and relative_to_face < 0.25:
                return "Weak", f"{detail}. Skin is unusually sparse for this face.", "warn"
            if layer == "hair" and relative_to_face > 2.25:
                return "Broad", f"{detail}. Hair may be spilling outside the target person.", "warn"
            if layer == "eyes" and (relative_to_face < 0.015 or relative_to_face > 0.22):
                return "Weak", f"{detail}. Eye coverage is outside the expected range.", "warn"
            if layer == "lips" and (relative_to_face < 0.008 or relative_to_face > 0.16):
                return "Weak", f"{detail}. Lip coverage is outside the expected range.", "warn"

        if layer in ("eyes", "lips") and cov50 < 0.0002:
            return "Weak", f"{detail}. The confident core is very small.", "warn"
        if layer in ("person", "face") and cov10 > 0.75:
            return "Broad", f"{detail}. This mask covers most of the image.", "warn"
        if edited:
            return "Edited", f"{detail}. Differs from the automatic baseline.", "edited"
        return "Good", detail, "good"

    def _mask_review_thumbnail(self, layer: str, mask) -> QPixmap:
        w, h = 48, 32
        if mask is None:
            rgb = np.full((h, w, 3), 15, dtype=np.uint8)
        else:
            arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
            if arr.ndim != 2 or arr.size == 0:
                small = np.zeros((h, w), dtype=np.float32)
            else:
                pil = Image.fromarray((arr * 255).astype(np.uint8))
                pil = pil.resize((w, h), Image.BILINEAR)
                small = np.asarray(pil, dtype=np.float32) / 255.0
            color_hex = LAYER_COLORS.get(layer, "#f59e0b")
            if color_hex.startswith("#") and len(color_hex) == 7:
                color = np.array(
                    [int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)],
                    dtype=np.float32,
                )
            else:
                color = np.array([245, 158, 11], dtype=np.float32)
            bg = np.array([15, 23, 42], dtype=np.float32)
            rgb = (bg[None, None, :] * (1.0 - small[..., None]) + color[None, None, :] * small[..., None]).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimage)

    def _update_mask_review_panel(self):
        if not getattr(self, "_mask_review_rows", None):
            return
        has_masks = self.preview_masks is not None
        instance = getattr(self.segmenter, "_instance", None)
        sam_ready = bool(instance is not None and getattr(instance, "available", False))
        status_styles = {
            "good": "color: #86efac; background: #0f2417; border: 1px solid #22c55e; border-radius: 4px; padding: 2px;",
            "warn": "color: #fde68a; background: #2a2110; border: 1px solid #f59e0b; border-radius: 4px; padding: 2px;",
            "bad": "color: #fecaca; background: #2a1114; border: 1px solid #ef4444; border-radius: 4px; padding: 2px;",
            "edited": "color: #bae6fd; background: #0c2330; border: 1px solid #38bdf8; border-radius: 4px; padding: 2px;",
            "pending": "color: #cbd5e1; background: #151b24; border: 1px solid #64748b; border-radius: 4px; padding: 2px;",
        }
        for layer, row in self._mask_review_rows.items():
            mask = self.preview_masks.get(layer) if has_masks and layer in self.preview_masks else None
            status_text, detail, style_key = self._mask_review_layer_stats(layer, mask)
            active = layer == self._active_layer

            row["thumb"].setPixmap(self._mask_review_thumbnail(layer, mask))
            row["thumb"].setToolTip(detail)
            row["thumb"].setStyleSheet(
                "border: 2px solid #38bdf8; background: #0f172a;" if active else "border: 1px solid #334155; background: #0f172a;"
            )
            row["layer"].blockSignals(True)
            row["layer"].setChecked(active)
            row["layer"].blockSignals(False)
            row["layer"].setEnabled(has_masks and layer in self.preview_masks)
            row["layer"].setToolTip(detail)
            row["status"].setText(status_text)
            row["status"].setToolTip(detail)
            row["status"].setStyleSheet(status_styles.get(style_key, status_styles["pending"]))
            row["recalc"].setEnabled(has_masks and layer in self.preview_masks and self._mask_recalc_task is None)
            row["ai"].setEnabled(has_masks and layer in self.preview_masks and sam_ready)
            row["reset"].setEnabled(has_masks and self._auto_preview_masks is not None and layer in self._auto_preview_masks)
            row["edit"].setEnabled(has_masks and layer in self.preview_masks)

    def _refresh_mask_controls(self):
        editable = self._active_layer in MASK_ORDER and self.preview_masks is not None
        self.mask_view_btn.setEnabled(editable)
        self.mask_edit_btn.setEnabled(editable)
        self.click_mask_btn.setEnabled(editable)
        self.mask_debug_combo.setEnabled(editable)
        self.guides_btn.setEnabled(self.preview_guides is not None)
        self.mask_mode_combo.setEnabled(editable)
        self.brush_slider.setEnabled(editable)
        self.hardness_slider.setEnabled(editable)
        self.reset_mask_btn.setEnabled(editable)
        self.recalculate_mask_btn.setEnabled(editable and self._mask_recalc_task is None)
        self.feather_mask_btn.setEnabled(editable)
        self._sync_mask_adjustment_controls()
        if not editable and self._mask_edit_enabled:
            self.mask_edit_btn.blockSignals(True)
            self.mask_edit_btn.setChecked(False)
            self.mask_edit_btn.blockSignals(False)
            self._mask_edit_enabled = False
        if not editable and self._click_mask_enabled:
            self.click_mask_btn.blockSignals(True)
            self.click_mask_btn.setChecked(False)
            self.click_mask_btn.blockSignals(False)
            self._click_mask_enabled = False
            self.image_label.set_click_mask_state(False, self._on_object_clicked)
        self.image_label.set_edit_state(editable and self._mask_edit_enabled, self._paint_active_mask_at, self._begin_mask_stroke, self._commit_mask_stroke)
        source_size = None
        if self.preview_array is not None:
            source_size = (self.preview_array.shape[1], self.preview_array.shape[0])
        self.image_label.set_brush_preview(
            self._mask_brush_size,
            source_image_size=source_size,
            hardness=self._mask_brush_hardness,
        )
        self._update_mask_review_panel()
        self._update_mask_debug_label()
        self._update_document_history_actions()

    def _on_compare_mode_changed(self, mode: str):
        changed = str(mode or "off") != self._compare_mode
        if changed:
            self._begin_document_change()
        self._compare_mode = str(mode or "off")
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.split_slider.setEnabled(self._compare_mode == "split")
        self._update_preview_label()
        if changed:
            self._push_document_history()

    def keyPressEvent(self, event):
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Space and self.preview_image is not None:
            if self._compare_mode != "before":
                self._compare_restore_mode = self._compare_mode
                self.compare_combo.blockSignals(True)
                self.compare_combo.setCurrentText("before")
                self.compare_combo.blockSignals(False)
                self._compare_mode = "before"
                self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
                self.split_slider.setEnabled(False)
                self._update_preview_label()
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._exit_active_edit_mode():
            event.accept()
            return
        super().keyPressEvent(event)

    def _exit_active_edit_mode(self) -> bool:
        """Back out of whichever modal canvas tool (crop / mask-edit / white-balance pick /
        AI select) is currently active, so Escape always has an obvious, universal effect
        instead of requiring the user to find and re-click that tool's own toggle button.

        Falls back to exiting Focus mode: Focus hides the canvas header, side panels, status
        bar, and toolbar -- including the very Focus button that turned it on -- so without this,
        Ctrl+Shift+F is the *only* way out and isn't discoverable once nothing is visible."""
        for attr, flag_attr in (
            ("crop_edit_btn", "_crop_edit_enabled"),
            ("mask_edit_btn", "_mask_edit_enabled"),
            ("wb_pick_btn", "_wb_pick_enabled"),
            ("click_mask_btn", "_click_mask_enabled"),
            ("denoise_point_btn", "_denoise_point_enabled"),
        ):
            if hasattr(self, attr) and getattr(self, flag_attr, False):
                getattr(self, attr).setChecked(False)
                return True
        if getattr(self, "_focus_mode", False):
            self._focus_action.setChecked(False)
            return True
        return False

    def keyReleaseEvent(self, event):
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit)):
            super().keyReleaseEvent(event)
            return
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return
        if event.key() == Qt.Key_Space and self._compare_restore_mode is not None:
            restore_mode = self._compare_restore_mode
            self._compare_restore_mode = None
            self.compare_combo.blockSignals(True)
            self.compare_combo.setCurrentText(restore_mode)
            self.compare_combo.blockSignals(False)
            self._compare_mode = restore_mode
            self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
            self.split_slider.setEnabled(self._compare_mode == "split")
            self._update_preview_label()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _on_split_slider_changed(self, value: int):
        new_value = max(0.0, min(1.0, float(value) / 100.0))
        if abs(new_value - self._split_position) > 1e-6:
            self._begin_document_change()
        self._split_position = new_value
        if self._compare_mode == "split":
            self._update_preview_label()

    def _set_split_position(self, value: float):
        new_value = max(0.0, min(1.0, float(value)))
        if abs(new_value - self._split_position) > 1e-6:
            self._begin_document_change()
        self._split_position = new_value
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(int(round(self._split_position * 100)))
        self.split_slider.blockSignals(False)
        if self._compare_mode == "split":
            self._update_preview_label()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.cr2 *.CR2 *.nef *.NEF *.arw *.ARW *.dng *.DNG *.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        try:
            self._load_image_path(path)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", self._friendly_error_message(ex))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Portrait Project (*.peproj *.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                project = json.load(fh)
            image_path = project.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Source image not found: {image_path}")
            self._load_image_path(image_path, project_state=project)
        except Exception as ex:
            QMessageBox.critical(self, "Project Load Error", self._friendly_error_message(ex))

    def save_project(self):
        if not self.file_path:
            QMessageBox.information(self, "Save Project", "Open an image first.")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "portrait_enhancer_qt_project.peproj",
            "Portrait Project (*.peproj *.json)",
        )
        if not out_path:
            return
        try:
            project = self._serialize_project_state()
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(project, fh)
            self.statusBar().showMessage(f"Project saved -> {os.path.basename(out_path)}")
        except Exception as ex:
            QMessageBox.critical(self, "Project Save Error", self._friendly_error_message(ex))

    # RAW decode settings only take effect at decode time. The decode is snapshotted below
    # before _on_image_decoded applies the saved project/override, so these keys must be
    # pre-applied or a saved RAW would decode with default settings.
    _RAW_DECODE_KEYS = (
        "raw_white_balance", "raw_colorspace", "raw_lut_enabled", "raw_lut_path",
        "raw_highlight_mode", "raw_demosaic", "raw_auto_brightness",
    )

    def _preapply_raw_decode_settings(self, path: str, project_state):
        """Merge just the RAW *decode* keys from the saved project/collection settings into
        _color_settings before the decode snapshot, so opening a saved RAW honors its
        highlight/demosaic/WB/colorspace/LUT/auto-brightness choices. The full settings are
        still applied post-decode as usual; render-time keys don't affect the decode."""
        if not is_raw_path(path):
            return
        saved = None
        if isinstance(project_state, dict) and isinstance(project_state.get("color_settings"), dict):
            saved = project_state["color_settings"]
        else:
            override = self._get_collection_image_override(path)
            if isinstance(override, dict) and isinstance(override.get("color_settings"), dict):
                saved = override["color_settings"]
        if not saved:
            return
        for key in self._RAW_DECODE_KEYS:
            if key in saved:
                self._color_settings[key] = saved[key]

    def _start_image_decode_task(self, path: str, project_state=None):
        self._preapply_raw_decode_settings(path, project_state)
        self._image_load_job_id += 1
        job_id = self._image_load_job_id
        task = ImageLoadTask(job_id, path, dict(self._color_settings))
        self._image_load_project_state = project_state
        # Pin the task on self -- otherwise the only reference is this local variable, and
        # nothing guarantees the Python wrapper survives until the worker thread emits back
        # to the GUI thread (observed as an intermittent segfault without this).
        self._image_load_task = task
        task.signals.preview_ready.connect(self._on_image_preview_decoded)
        task.signals.finished.connect(self._on_image_decoded)
        task.signals.failed.connect(self._on_image_decode_failed)
        self._image_load_pool.start(task)

    def _load_image_path(self, path: str, project_state=None):
        if project_state is None and self.file_path and self.file_path != path:
            self._save_current_image_override()
        self.file_path = path
        self.statusBar().showMessage(f"Loading {os.path.basename(path)} ...")
        self.image_label.reset_view()
        self._update_filmstrip_active()
        self._invalidate_active_image_work()
        self._lazy_source_path = None
        self._lazy_source_project_state = None
        self._lazy_source_loading = False
        self._lazy_hydrate_callbacks = []
        self._lazy_hydrate_settings_payload = None
        # Decode (RAW demosaic especially) runs off the GUI thread -- on a large RAW file this
        # took multiple seconds synchronously before, freezing the whole window on every image
        # open/switch. job_id guards against a slower, older decode overwriting a newer one if
        # the user clicks through several filmstrip images in quick succession.
        self._start_image_decode_task(path, project_state=project_state)
        self._update_empty_state()

    def _hydrate_active_image_if_needed(
        self,
        reason: str = "edit",
        on_complete=None,
        preserve_current_settings: bool = False,
    ) -> bool:
        if self.full_array is not None:
            if on_complete is not None:
                on_complete()
            return False
        if not self.file_path or self._lazy_source_path != self.file_path:
            return False
        if preserve_current_settings:
            self._lazy_hydrate_settings_payload = self._build_settings_payload()
        if on_complete is not None:
            self._lazy_hydrate_callbacks.append(on_complete)
        if self._lazy_source_loading:
            self.statusBar().showMessage(f"Loading source for {reason}...")
            return True
        self._lazy_source_loading = True
        self.statusBar().showMessage(f"Loading source for {reason}...")
        self._start_image_decode_task(self._lazy_source_path, project_state=self._lazy_source_project_state)
        return True

    def _run_lazy_hydrate_callbacks(self):
        callbacks = list(self._lazy_hydrate_callbacks)
        self._lazy_hydrate_callbacks = []
        for callback in callbacks:
            try:
                callback()
            except Exception as ex:
                QMessageBox.critical(self, "Source Load Action Error", self._friendly_error_message(ex))

    def _invalidate_active_image_work(self):
        """Mark in-flight preview/analysis work as stale when the active image changes."""
        self._image_load_job_id += 1
        self._image_load_task = None
        if hasattr(self, "_render_timer"):
            self._render_timer.stop()
        self._render_job_counter += 1
        self._active_render_job_id = self._render_job_counter
        self._render_pending = False

        self._segmentation_job_counter += 1
        self._active_segmentation_job_id = self._segmentation_job_counter
        self._segmentation_pending_callback = None

        self._mask_click_job_id += 1
        self._mask_recalc_job_id += 1

    def _reset_face_analysis_state(self):
        self._detected_faces = []
        self._active_face_index = 0
        if hasattr(self, "face_combo"):
            self.face_combo.blockSignals(True)
            self.face_combo.clear()
            self.face_combo.addItem("Auto")
            self.face_combo.setCurrentIndex(0)
            self.face_combo.blockSignals(False)
        self._update_face_scope_indicator()

    def _show_decoded_source_preview(self):
        """Paint the decoded image immediately, before masks/auto settings finish."""
        if self.preview_array is None:
            return
        self.preview_image = Image.fromarray(to_uint8(self.preview_array)).convert("RGB")
        self._perf_stats["detect_ms"] = None
        self._perf_stats["segment_ms"] = None
        self._perf_stats["render_ms"] = None
        self._update_perf_label()
        self._refresh_mask_controls()
        self._update_preview_label()
        self._update_histogram()
        self._update_empty_state()

    def _source_file_signature(self, path: str | None = None):
        path = path or self.file_path
        if not path:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return {
            "path": os.path.abspath(path),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            "size": int(stat.st_size),
        }

    def _current_preview_render_state_payload(self):
        if self.preview_array is None or not self.file_path or self._source_pixels_modified:
            return None
        source_sig = self._source_file_signature(self.file_path)
        if source_sig is None:
            return None
        shape = tuple(int(v) for v in self.preview_array.shape[:2])
        return {
            "version": int(self.PREVIEW_RENDER_CACHE_VERSION),
            "source": source_sig,
            "preview_shape": list(shape),
            "preview_state_revision": int(self._preview_state_revision),
            "analysis": self._preview_analysis_signature,
            "params": self._copy_params(self._all_params()),
            "layer_order": list(self._layer_order),
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "color_settings": dict(self._color_settings),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "runtime_settings": dict(self._runtime_settings),
        }

    def _current_preview_render_cache_key(self):
        payload = self._current_preview_render_state_payload()
        if payload is None:
            return None
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _collection_preview_records(self, path: str | None = None):
        if not self._active_collection or self._active_collection not in self._collections:
            return None
        collection = self._collections[self._active_collection]
        image_path = path or self.file_path
        if image_path not in collection.get("images", []):
            return None
        records = collection.setdefault("rendered_previews", {})
        if not isinstance(records, dict):
            records = {}
            collection["rendered_previews"] = records
        return records

    def _touch_preview_render_cache_key(self, key):
        self._preview_render_cache_order = [existing for existing in self._preview_render_cache_order if existing != key]
        self._preview_render_cache_order.append(key)

    def _drop_preview_render_cache(self, path: str | None = None):
        self._preview_render_cache.clear()
        self._preview_render_cache_order = []
        if path and self._active_collection in self._collections:
            records = self._collections[self._active_collection].get("rendered_previews", {})
            if isinstance(records, dict):
                records.pop(path, None)
                self._save_collections_state()

    def _segmentation_result_cache_key(self, path: str | None = None, face_index: int | None = None):
        if self.preview_array is None or self.full_array is None or self._source_pixels_modified:
            return None
        image_path = path or self.file_path
        source_sig = self._source_file_signature(image_path)
        if source_sig is None:
            return None
        payload = {
            "analysis_cache_version": int(ANALYSIS_CACHE_VERSION),
            "source": source_sig,
            "preview_shape": [int(v) for v in self.preview_array.shape[:2]],
            "full_shape": [int(v) for v in self.full_array.shape[:2]],
            "preview_scale": round(float(self.preview_scale), 8),
            "face_index": max(0, int(self._active_face_index if face_index is None else face_index)),
            "segmenter": segmenter_cache_signature(self.segmenter),
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def _segmentation_cache_entry_bytes(self, entry: dict) -> int:
        total = 0
        for mask in (entry.get("preview_masks") or {}).values():
            total += int(getattr(mask, "nbytes", 0))
        return total

    def _copy_segmentation_result_for_cache(self, result: dict) -> dict | None:
        if not isinstance(result, dict):
            return None
        preview_masks = self._copy_masks(result.get("preview_masks"))
        if not preview_masks:
            return None
        entry = {
            "detected_faces": [tuple(int(v) for v in face) for face in result.get("detected_faces", [])],
            "face_index": int(result.get("face_index", 0)),
            "mask_face_index": int(result.get("mask_face_index", result.get("face_index", 0))),
            "preview_masks": preview_masks,
            "preview_guides": self._copy_guides(result.get("preview_guides")),
        }
        entry["bytes"] = self._segmentation_cache_entry_bytes(entry)
        return entry

    def _remember_segmentation_result_cache(self, key: str | None, result: dict):
        if key is None:
            return
        entry = self._copy_segmentation_result_for_cache(result)
        if entry is None:
            return
        existing = self._segmentation_result_cache.pop(key, None)
        if existing is not None:
            self._segmentation_result_cache_bytes -= int(existing.get("bytes", 0))
            self._segmentation_result_cache_order = [
                existing_key for existing_key in self._segmentation_result_cache_order if existing_key != key
            ]
        self._segmentation_result_cache[key] = entry
        source_sig = self._source_file_signature(self.file_path)
        if source_sig is not None:
            entry["source_path"] = source_sig.get("path")
        self._segmentation_result_cache_order.append(key)
        self._segmentation_result_cache_bytes += int(entry.get("bytes", 0))
        while (
            len(self._segmentation_result_cache_order) > self._segmentation_result_cache_limit
            or self._segmentation_result_cache_bytes > self._segmentation_result_cache_byte_limit
        ):
            old_key = self._segmentation_result_cache_order.pop(0)
            old_entry = self._segmentation_result_cache.pop(old_key, None)
            if old_entry is not None:
                self._segmentation_result_cache_bytes -= int(old_entry.get("bytes", 0))

    def _drop_segmentation_result_cache(self, path: str | None = None):
        if not path:
            self._segmentation_result_cache.clear()
            self._segmentation_result_cache_order = []
            self._segmentation_result_cache_bytes = 0
            return
        source_sig = self._source_file_signature(path)
        if source_sig is None:
            return
        abs_path = source_sig.get("path")
        kept = {}
        kept_order = []
        kept_bytes = 0
        for key in self._segmentation_result_cache_order:
            entry = self._segmentation_result_cache.get(key)
            if entry is None:
                continue
            entry_path = entry.get("source_path")
            if entry_path == abs_path:
                continue
            kept[key] = entry
            kept_order.append(key)
            kept_bytes += int(entry.get("bytes", 0))
        self._segmentation_result_cache = kept
        self._segmentation_result_cache_order = kept_order
        self._segmentation_result_cache_bytes = kept_bytes

    def _segmentation_result_from_cache_entry(self, entry: dict):
        if self.full_array is None:
            return None
        preview_masks = self._copy_masks(entry.get("preview_masks"))
        preview_guides = self._copy_guides(entry.get("preview_guides"))
        full_masks = _scale_masks_data(preview_masks, self.full_array.shape[:2])
        scale = 1.0 / max(self.preview_scale, 1e-6)
        full_guides = _scale_expression_guides(self._copy_guides(preview_guides), scale, scale)
        return {
            "detected_faces": [tuple(int(v) for v in face) for face in entry.get("detected_faces", [])],
            "face_index": int(entry.get("face_index", 0)),
            "mask_face_index": int(entry.get("mask_face_index", entry.get("face_index", 0))),
            "preview_masks": preview_masks,
            "preview_guides": preview_guides,
            "full_masks": full_masks,
            "full_guides": full_guides,
            "detect_ms": 0.0,
            "segment_ms": 0.0,
            "cache_status": "memory",
        }

    def _restore_segmentation_result_cache(self, key: str | None) -> dict | None:
        if key is None:
            return None
        entry = self._segmentation_result_cache.get(key)
        if entry is None:
            return None
        self._segmentation_result_cache_order = [
            existing for existing in self._segmentation_result_cache_order if existing != key
        ]
        self._segmentation_result_cache_order.append(key)
        return self._segmentation_result_from_cache_entry(entry)

    def _load_segmentation_disk_cache_result(self, face_index: int):
        if self.preview_array is None or self.full_array is None or not self.file_path:
            return None
        requested_face_index = max(0, int(face_index))
        cache_signature = segmenter_cache_signature(self.segmenter)
        cached = load_analysis(
            self.file_path,
            kind=CACHE_KIND_SINGLE_FACE,
            image_shape=self.preview_array.shape[:2],
            face_index=requested_face_index,
            backend_signature=cache_signature,
        )
        if cached is None:
            return None
        detected_faces = cached.get("faces") or []
        if detected_faces and requested_face_index >= len(detected_faces):
            return None
        preview_masks = self._copy_masks(cached.get("masks"))
        if not preview_masks:
            return None
        preview_guides = self._copy_guides(cached.get("guides"))
        full_masks = _scale_masks_data(preview_masks, self.full_array.shape[:2])
        scale = 1.0 / max(self.preview_scale, 1e-6)
        full_guides = _scale_expression_guides(self._copy_guides(preview_guides), scale, scale)
        resolved_face_index = min(requested_face_index, len(detected_faces) - 1) if detected_faces else 0
        return {
            "detected_faces": [tuple(int(v) for v in face) for face in detected_faces],
            "face_index": int(resolved_face_index),
            "mask_face_index": int(resolved_face_index),
            "preview_masks": preview_masks,
            "preview_guides": preview_guides,
            "full_masks": full_masks,
            "full_guides": full_guides,
            "detect_ms": 0.0,
            "segment_ms": 0.0,
            "cache_status": "disk",
        }

    def _remember_preview_render_cache(self):
        if self._show_sharpen_mask_preview:
            return
        if self.preview_image is None or self.preview_array is None or self._slider_drag_active > 0:
            return
        expected_size = (int(self.preview_array.shape[1]), int(self.preview_array.shape[0]))
        if self.preview_image.size != expected_size:
            return
        key = self._current_preview_render_cache_key()
        if key is None:
            return
        self._preview_render_cache[key] = self.preview_image.copy()
        self._touch_preview_render_cache_key(key)
        while len(self._preview_render_cache_order) > self._preview_render_cache_limit:
            old_key = self._preview_render_cache_order.pop(0)
            self._preview_render_cache.pop(old_key, None)
        self._save_collection_rendered_preview(key)

    def _save_collection_rendered_preview(self, key: str):
        records = self._collection_preview_records()
        if records is None or self.preview_image is None or self.preview_array is None:
            return
        # Collection previews are durable only for file-backed source pixels and persisted
        # auto-analysis state. In-memory source swaps or manual mask revisions intentionally
        # stay session-local because those pixels/masks are not part of the collection record.
        if self._source_pixels_modified or self._preview_state_revision != 0:
            return
        expected_size = (int(self.preview_array.shape[1]), int(self.preview_array.shape[0]))
        if self.preview_image.size != expected_size:
            return
        preview_path = self._rendered_preview_path(key)
        existing = records.get(self.file_path)
        if isinstance(existing, dict) and existing.get("cache_key") == key and preview_path.is_file():
            return
        try:
            self.preview_image.convert("RGB").save(str(preview_path), "JPEG", quality=92, optimize=True)
        except Exception:
            return
        records[self.file_path] = {
            "cache_key": key,
            "file": preview_path.name,
            "source": self._source_file_signature(self.file_path),
            "width": int(self.preview_image.width),
            "height": int(self.preview_image.height),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_collections_state()
        self._prune_collection_rendered_previews()

    def _prune_collection_rendered_previews(self):
        try:
            files = sorted(
                self._rendered_preview_dir().glob("*.jpg"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return
        for path in files[self._preview_render_disk_cache_limit:]:
            try:
                path.unlink()
            except OSError:
                pass

    def _load_collection_saved_preview(
        self,
        path: str | None = None,
        cache_key: str | None = None,
        require_preview_size: bool = True,
    ):
        if self.preview_array is None and require_preview_size:
            return None
        image_path = path or self.file_path
        if not image_path:
            return None
        records = self._collection_preview_records(image_path)
        record = records.get(image_path) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            return None
        if cache_key is not None and record.get("cache_key") != cache_key:
            return None
        saved_source = record.get("source")
        if isinstance(saved_source, dict):
            current_source = self._source_file_signature(image_path)
            if current_source is None or saved_source != current_source:
                return None
        filename = Path(str(record.get("file", ""))).name
        if not filename:
            return None
        disk_path = self._rendered_preview_dir() / filename
        if not disk_path.is_file():
            return None
        try:
            cached = Image.open(disk_path).convert("RGB")
        except Exception:
            return None
        if self.preview_array is not None:
            expected_size = (int(self.preview_array.shape[1]), int(self.preview_array.shape[0]))
            if cached.size != expected_size:
                return None
        else:
            record_w = int(record.get("width", 0) or 0)
            record_h = int(record.get("height", 0) or 0)
            if record_w > 0 and record_h > 0 and cached.size != (record_w, record_h):
                return None
        return cached

    def _show_collection_saved_preview_before_decode(self, path: str) -> bool:
        cached = self._load_collection_saved_preview(path, require_preview_size=False)
        if cached is None:
            return False

        self.full_array = None
        self.preview_array = None
        self.preview_scale = 1.0
        self.image_label.set_hires_tile(None, None)
        self._interactive_preview_array = None
        self._interactive_preview_ratio = 1.0
        self.preview_image = cached.copy()
        self._source_metadata = {}
        self._preview_state_revision = 0
        self._preview_analysis_signature = None
        self._source_pixels_modified = False
        # Fresh stage cache per image -- replace, don't clear, so any in-flight worker render
        # keeps its own orphaned cache and can't race this GUI-thread reset.
        self._stage_pipeline_cache = StagePipelineCache(limit=24)
        self._face_profiles = {}
        self._mask_adjustments = default_mask_adjustments()
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self._mask_face_index = None
        self.preview_guides = None
        self.full_guides = None
        self._reset_face_analysis_state()
        self._reset_deep_denoise_cache()
        self._reset_fix_eyes_cache()

        self._reset_editing_state_to_defaults()
        self._framing = default_framing()
        override = self._get_collection_image_override(path)
        if override:
            self._apply_pre_segmentation_settings_payload(override)
        else:
            self._sync_framing_controls()
            self._sync_wb_controls()
            self._sync_tone_curve_controls()
            self._sync_hsl_controls()

        # This deferred preview path shows a cached preview and returns before any decode, so it
        # must sync the RAW Decode panel itself -- otherwise the panel stays greyed (reads as
        # non-RAW) for a RAW until the source is hydrated on first edit. file_path is the RAW
        # here and _color_settings is already loaded, so the panel reflects the right state.
        self._sync_raw_decode_controls()

        self._perf_stats["detect_ms"] = None
        self._perf_stats["segment_ms"] = None
        self._perf_stats["render_ms"] = 0.0
        self._update_perf_label()
        self._refresh_mask_controls()
        self._update_preview_label()
        self._update_histogram()
        self._sync_simple_controls()
        return True

    def _restore_collection_saved_preview_optimistic(self, path: str | None = None) -> bool:
        cached = self._load_collection_saved_preview(path)
        if cached is None:
            return False
        self.preview_image = cached.copy()
        self._perf_stats["detect_ms"] = None
        self._perf_stats["segment_ms"] = None
        self._perf_stats["render_ms"] = 0.0
        self._update_perf_label()
        self._refresh_mask_controls()
        self._update_preview_label()
        self._update_histogram()
        self._sync_simple_controls()
        return True

    def _restore_preview_render_cache(self) -> bool:
        if self._show_sharpen_mask_preview:
            return False
        key = self._current_preview_render_cache_key()
        if key is None:
            return False
        cached = self._preview_render_cache.get(key)
        if cached is None:
            cached = self._load_collection_saved_preview(self.file_path, cache_key=key)
            if cached is None:
                return False
            self._preview_render_cache[key] = cached.copy()
        self.preview_image = cached.copy()
        self._touch_preview_render_cache_key(key)
        self._perf_stats["render_ms"] = 0.0
        self._update_perf_label()
        self._update_preview_label()
        self._update_histogram()
        self._sync_simple_controls()
        return True

    def _mark_preview_pixels_changed(self):
        self._preview_state_revision += 1

    def _mark_source_pixels_modified(self):
        self._source_pixels_modified = True
        self._mark_preview_pixels_changed()
        self._drop_preview_render_cache(self.file_path)
        self._drop_segmentation_result_cache(self.file_path)
        # The deep-denoise proxy/full caches were computed on the now-replaced pixels (e.g. a
        # Fix Eyes swap) -- drop them so a stale denoise isn't shown or exported. Re-toggle to
        # recompute against the new pixels.
        self._reset_deep_denoise_cache()

    def _on_image_decode_failed(self, job_id, path, message):
        if job_id != self._image_load_job_id:
            return
        self._image_load_task = None
        self._lazy_source_loading = False
        self._lazy_hydrate_callbacks = []
        QMessageBox.critical(self, "Load Error", self._friendly_error_message(RuntimeError(message)))
        self.statusBar().showMessage(f"Failed to load {os.path.basename(path)}")
        self._update_empty_state()

    def _on_image_preview_decoded(self, job_id, path, full, metadata):
        """Fast half-resolution RAW preview, fired before the full decode completes (see
        ImageLoadSignals.preview_ready) -- shows pixel-accurate pixels on the canvas in roughly
        a quarter of the full decode's time, instead of a blank/placeholder canvas the whole
        wait. Display-only and intentionally minimal: it does NOT touch mask_adjustments,
        face_profiles, segmentation, history, or color-settings sync -- _on_image_decoded still
        owns all of that and runs moments later with the real full-resolution array, cleanly
        superseding this preview. Editing during this brief window is unguarded (a slider drag
        would render against the half-res array), but the gap is typically 1-4 seconds and the
        full decode's reset makes any such interim render harmless and short-lived."""
        if job_id != self._image_load_job_id:
            return  # superseded by a newer load (e.g. rapid filmstrip clicks)
        self._source_metadata = metadata
        preview, scale = self._build_preview_proxy(full)
        self.full_array = full
        self.preview_array = preview
        self.preview_scale = scale
        self._update_empty_state()
        self.image_label.set_full_res_ratio(1.0 / scale if scale > 0 else 1.0)
        self.image_label.set_hires_tile(None, None)
        self._show_decoded_source_preview()
        self.info_label.setText(f"Loaded {os.path.basename(path)} - decoding full resolution...")
        self.statusBar().showMessage(f"Loaded {os.path.basename(path)} - decoding full resolution...")

    def _on_image_decoded(self, job_id, path, full, metadata):
        if job_id != self._image_load_job_id:
            return  # superseded by a newer load (e.g. rapid filmstrip clicks)
        project_state = self._image_load_project_state
        pending_settings_payload = self._lazy_hydrate_settings_payload
        self._lazy_hydrate_settings_payload = None
        self._lazy_source_loading = False
        self._image_load_task = None
        self._source_metadata = metadata
        preview, scale = self._build_preview_proxy(full)
        interactive_preview, interactive_ratio = self._build_interactive_preview_proxy(preview)
        self.full_array = full
        self.preview_array = preview
        self.preview_scale = scale
        self._update_empty_state()
        self.image_label.set_full_res_ratio(1.0 / scale if scale > 0 else 1.0)
        self.image_label.set_hires_tile(None, None)
        self._interactive_preview_array = interactive_preview
        self._interactive_preview_ratio = interactive_ratio
        self.preview_image = None
        self._preview_state_revision = 0
        self._preview_analysis_signature = None
        self._source_pixels_modified = False
        # Fresh stage cache per image -- replace, don't clear, so any in-flight worker render
        # keeps its own orphaned cache and can't race this GUI-thread reset.
        self._stage_pipeline_cache = StagePipelineCache(limit=24)
        self._face_profiles = {}
        self._mask_adjustments = default_mask_adjustments()
        self._reset_face_analysis_state()
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self._mask_face_index = None
        self.preview_guides = None
        self.full_guides = None
        self._reset_deep_denoise_cache()
        self._reset_fix_eyes_cache()
        if project_state is not None:
            self._framing = default_framing()
            self._show_decoded_source_preview()
            self.info_label.setText(f"Loaded {os.path.basename(path)} - applying project...")
            self.statusBar().showMessage(f"Loaded {os.path.basename(path)} - applying project...")
            self._apply_project_state(project_state)
        else:
            self._reset_editing_state_to_defaults()
            override = pending_settings_payload if pending_settings_payload is not None else self._get_collection_image_override(path)
            # Fresh RAW loads default to the scene-linear pipeline + VST denoise (better
            # noise/exposure handling). A saved override keeps whatever it stored, and non-RAW
            # images stay sRGB. Done before the syncs below so the RAW Decode panel reflects it.
            if not override and is_raw_path(path):
                self._color_settings["working_space"] = "linear"
                self._color_settings["scene_linear_denoise"] = True
            self._framing = default_framing()
            self._sync_framing_controls()
            self._sync_wb_controls()
            self._sync_raw_decode_controls()
            self._sync_tone_curve_controls()
            self._sync_hsl_controls()
            # Restore the saved active-face target before segmentation runs, so the
            # right face's masks get generated and the face picker reflects it.
            if override:
                self._apply_pre_segmentation_settings_payload(override)
            else:
                self._active_face_index = 0
            restored_saved_preview = self._restore_collection_saved_preview_optimistic(path)
            if restored_saved_preview:
                self.info_label.setText(f"Loaded {os.path.basename(path)} - validating saved preview...")
                self.statusBar().showMessage(f"Loaded {os.path.basename(path)} - validating saved preview...")
            else:
                self._show_decoded_source_preview()
                self.info_label.setText(f"Loaded {os.path.basename(path)} - analyzing in background...")
                self.statusBar().showMessage(f"Loaded {os.path.basename(path)} - analyzing in background...")
            # Segmentation now runs off the GUI thread -- _apply_settings_payload (which
            # needs the resulting masks) and the history push must wait for it, so they run
            # in on_complete instead of immediately after this call returns.
            self._run_segmentation(
                on_complete=lambda override=override: self._finish_load_image_path(override),
                schedule_render=False,
            )

    def _finish_load_image_path(self, override):
        self._apply_auto_global_suggestions()
        if override:
            self._apply_settings_payload(override)
        self._clear_document_history()
        self._push_document_history()
        # Auto-enhance is opt-in (Preferences) and only ever touches an image that has no saved
        # settings yet -- an override means this image was already customized, so applying a
        # generic recipe on top would clobber that customization rather than help. The baseline
        # above is pushed first so this lands as its own, separately undoable history entry.
        if not override and self._preferences.get("auto_enhance_on_open", False):
            recipes = self._guided_recipes()
            if recipes:
                self._apply_preset_state(self._recipe_to_preset(recipes[0]))
                self._push_document_history()
                self.statusBar().showMessage(
                    f"Auto-enhanced with \"{recipes[0]['name']}\" -- Ctrl+Z to undo, or turn off in Preferences."
                )
        if not self._restore_preview_render_cache():
            self._schedule_render()
        self._lazy_source_path = None
        self._lazy_source_project_state = None
        self._lazy_source_loading = False
        self._run_lazy_hydrate_callbacks()

    # Layers whose "noise_red" gets its own region-masked estimate (suggest_region_noise_red)
    # instead of the whole-image one -- see _apply_auto_global_suggestions/_auto_correct_slider.
    _REGION_NOISE_LAYERS = ("skin", "background", "person")

    def _set_slider_silently(self, layer: str, key: str, value):
        slider = self._sliders.get(layer, {}).get(key)
        if slider is None:
            return
        slider.blockSignals(True)
        slider.setValue(int(value))
        slider.blockSignals(False)
        label = self._slider_value_labels.get(layer, {}).get(key)
        if label is not None:
            label.setText(f"{int(value):+d}" if value else "0")

    def _apply_auto_global_suggestions(self):
        """Seed Noise Reduc./Color NR/Sharpness with starting values estimated from this
        image's own measured noise, instead of always starting from 0 -- and seed each of
        Skin/Background/Person's own Noise Reduc. from that region's own mask, since a region
        can have a meaningfully different actual noise level than the whole-image average
        (e.g. a shadowed background, smoother midtone skin). Fully user-adjustable afterward;
        a collection override or pasted preset applied right after this still takes
        precedence."""
        if self.full_array is None or "global" not in self._sliders:
            return
        suggestion = suggest_global_auto_values(self.full_array)
        for key, value in suggestion.items():
            self._set_slider_silently("global", key, value)

        full_masks = self.full_masks or {}
        for layer in self._REGION_NOISE_LAYERS:
            if layer not in self._sliders or "noise_red" not in self._sliders[layer]:
                continue
            value = suggest_region_noise_red(self.full_array, full_masks.get(layer))
            self._set_slider_silently(layer, "noise_red", value)

    def _auto_correct_slider(self, layer: str, key: str):
        """Re-estimate this image's noise/tone characteristics on demand and apply the
        result to a single slider. Skin/Background/Person's own Noise Reduc. measures that
        layer's own mask (matching how _apply_auto_global_suggestions seeds it on image open)
        instead of the whole-image estimate every other Auto button uses."""
        if self.full_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        slider = self._sliders.get(layer, {}).get(key)
        if slider is None:
            return
        if key == "noise_red" and layer in self._REGION_NOISE_LAYERS:
            mask = (self.full_masks or {}).get(layer)
            if mask is None:
                self.statusBar().showMessage(f"Auto: no detected {layer} region in this image to measure")
                return
            value = suggest_region_noise_red(self.full_array, mask)
        else:
            suggestion = {**suggest_global_auto_values(self.full_array), **suggest_auto_tone(self.full_array)}
            value = suggestion.get(key)
            if value is None:
                return
        self._begin_document_change()
        slider.setValue(int(value))
        self._push_document_history()
        self.statusBar().showMessage(f"Auto-corrected {key.replace('_', ' ')} -> {int(value)}")

    def _apply_auto_tone(self):
        """Single-click Auto Tone: set Exposure/Blacks/Whites together from this image's
        own histogram (classic auto-levels), leaving every other slider untouched."""
        if self.full_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        sliders = self._sliders.get("global", {})
        suggestion = suggest_auto_tone(self.full_array)
        self._begin_document_change()
        for key in ("exposure", "blacks", "whites"):
            slider = sliders.get(key)
            value = suggestion.get(key)
            if slider is not None and value is not None:
                slider.setValue(int(value))
        self._push_document_history()
        self.statusBar().showMessage(
            "Auto Tone: exposure {exposure:+d}, blacks {blacks:+d}, whites {whites:+d}".format(**suggestion)
        )

    def _apply_auto_tone_face(self):
        """Single-click Auto Tone (Face): same Exposure/Blacks/Whites correction as Auto
        Tone, but the Exposure target is driven by the detected face's own brightness
        instead of the whole frame -- useful when the face is backlit or in shadow relative
        to the rest of the scene. Falls back to the whole-image estimate if no face mask is
        available yet."""
        if self.full_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        face_mask = (self.full_masks or {}).get("face")
        if face_mask is None:
            suggestion = suggest_auto_tone(self.full_array)
        else:
            suggestion = suggest_auto_tone_for_face(self.full_array, face_mask)
        sliders = self._sliders.get("global", {})
        self._begin_document_change()
        for key in ("exposure", "blacks", "whites"):
            slider = sliders.get(key)
            value = suggestion.get(key)
            if slider is not None and value is not None:
                slider.setValue(int(value))
        self._push_document_history()
        self.statusBar().showMessage(
            "Auto Tone (Face): exposure {exposure:+d}, blacks {blacks:+d}, whites {whites:+d}".format(**suggestion)
        )

    def _apply_auto_subject(self):
        """Single-click Auto Subject: expose for the subject and gently separate it from the
        background, by setting the Subjects/Background layer sliders from the subject mask.
        Conservative -- skips adjustments the image doesn't need. Requires the model-based
        Subject backend (the heuristic mask is too rough to drive this well)."""
        if self.preview_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        # Analyze on the preview-resolution arrays already in memory (region medians are
        # scale-invariant) -- avoids touching the full ~26MP frame, so it stays instant even
        # on large group photos.
        masks = self.preview_masks or {}
        subject_mask = masks.get("subjects")
        if subject_mask is None:
            self.statusBar().showMessage("Subject mask not ready yet — adjust or reopen the image first")
            return

        suggestion = suggest_auto_subject(self.preview_array, subject_mask, masks.get("background"))
        changes = [
            (layer, key, value)
            for layer in ("subjects", "background")
            for key, value in suggestion.get(layer, {}).items()
            if self._sliders.get(layer, {}).get(key) is not None
        ]
        if not changes:
            self.statusBar().showMessage("Auto Subject: image already balanced — no change")
            return

        self._begin_document_change()
        for layer, key, value in changes:
            self._sliders[layer][key].setValue(int(value))
        self._push_document_history()
        summary = ", ".join(f"{layer[:3]}.{key} {int(value):+d}" for layer, key, value in changes)
        self.statusBar().showMessage(f"Auto Subject: {summary}")

    def _current_aspect_target(self) -> float | None:
        """Resolve the Aspect combo's current selection to a numeric aspect (None = free)."""
        if not hasattr(self, "aspect_combo"):
            return None
        idx = self.aspect_combo.currentIndex()
        if idx < 0 or idx >= len(self._aspect_presets):
            return None
        _label, value = self._aspect_presets[idx]
        if value == "original":
            return self._source_aspect()
        return value

    def _apply_auto_crop(self):
        """Single-click Auto Crop: crop centered on the subject (fit to the selected Aspect
        preset, same as the Aspect dropdown but subject-aware instead of frame-centered) and,
        when confident, straighten the horizon. Conservative -- does nothing when the frame
        already looks good (subject fills it, no clear horizon tilt)."""
        if self.preview_array is None:
            if self._hydrate_active_image_if_needed("auto crop", on_complete=self._apply_auto_crop):
                return
            self.statusBar().showMessage("Open an image first")
            return
        masks = self.preview_masks or {}
        subject_mask = masks.get("subjects")
        if subject_mask is None:
            self.statusBar().showMessage("Subject mask not ready yet — adjust or reopen the image first")
            return

        suggestion = suggest_auto_crop(
            self.preview_array,
            subject_mask,
            self._current_aspect_target(),
            faces=self._detected_faces,
            guides=self.preview_guides,
        )
        if not suggestion:
            self.statusBar().showMessage("Auto Crop: frame already looks good — no change")
            return

        self._begin_document_change()
        parts = []
        if "crop" in suggestion:
            self._framing["crop"] = suggestion["crop"]
            parts.append("cropped to subject")
        if "angle" in suggestion:
            self._framing["angle"] = suggestion["angle"]
            parts.append(f"straightened {suggestion['angle']:+.1f}°")
        self._framing = normalize_framing(self._framing)
        self._sync_framing_controls()
        self._update_preview_label()
        self._push_document_history()
        self.statusBar().showMessage(f"Auto Crop: {', '.join(parts)}")

    def _reset_fix_eyes_cache(self):
        """Drop the per-image Fix Eyes cache (called on image load) and reset the button.
        Also invalidates any in-flight analysis -- without this, switching images mid-analysis
        would let a stale result land on whatever image happens to be open when it finishes."""
        self._fix_eyes_original = None
        self._fix_eyes_result = None
        self._fix_eyes_active = False
        self._fix_eyes_job_id += 1
        self._update_fix_eyes_btn()

    def _update_fix_eyes_btn(self):
        btn = getattr(self, "fix_eyes_btn", None)
        if btn is None:
            return
        if self._fix_eyes_result is None:
            btn.setText("Fix Eyes")
        elif self._fix_eyes_active:
            btn.setText("Revert Eyes")
        else:
            btn.setText("Re-apply Eyes")

    def _burst_group_for_active_image(self):
        """Resolve the burst group (list of paths) containing the currently open image, using
        the active collection's import-time quality data (blur/ahash/time). Returns None if
        there's no active collection, or a single-element list if this image isn't part of a
        detected burst (nothing to fetch a donor frame from)."""
        if not self._active_collection or self._active_collection not in self._collections:
            return None
        collection = self._collections[self._active_collection]
        quality = collection.get("image_quality", {})
        images = collection.get("images", [])
        if self.file_path not in images:
            return None
        items = [
            {"path": p, "ahash": quality.get(p, {}).get("ahash"), "time": quality.get(p, {}).get("time", 0.0)}
            for p in images
        ]
        groups = culling_ops.group_bursts(items)
        for group in groups:
            if self.file_path in group:
                return group
        return None

    def _run_fix_eyes(self):
        """Single-click Fix Eyes: if the open image is part of a detected burst and has a
        blinking face, replace it with the open-eyes version of the same face from another
        frame in the burst (matched by position -- burst frames are the same composition, so
        no face-recognition model is needed). Caches both pixel states so a second click
        toggles Revert/Re-apply instantly rather than re-running the analysis."""
        if self.full_array is None:
            self.statusBar().showMessage("Open an image first")
            return

        if self._fix_eyes_result is not None:
            if self._fix_eyes_active:
                self._swap_full_array(self._fix_eyes_original)
                self._fix_eyes_active = False
                self.statusBar().showMessage("Fix Eyes: reverted to original (re-apply is instant)")
            else:
                self._swap_full_array(self._fix_eyes_result)
                self._fix_eyes_active = True
                self.statusBar().showMessage("Fix Eyes: re-applied")
            self._update_fix_eyes_btn()
            return

        group = self._burst_group_for_active_image()
        if not group or len(group) < 2:
            self.statusBar().showMessage(
                "Fix Eyes: this image isn't part of a detected burst (run Review / Cull on "
                "the collection first, or this shot has no near-duplicate frames)"
            )
            return
        other_paths = [p for p in group if p != self.file_path]

        self.statusBar().showMessage(f"Fix Eyes: analyzing {len(other_paths)} nearby frame(s)...")
        self.fix_eyes_btn.setEnabled(False)
        self._fix_eyes_job_id += 1
        job_id = self._fix_eyes_job_id
        task = FixEyesTask(job_id, self.full_array, other_paths, self.segmenter)
        self._fix_eyes_task = task
        task.signals.finished.connect(self._on_fix_eyes_finished)
        task.signals.no_result.connect(self._on_fix_eyes_no_result)
        task.signals.failed.connect(self._on_fix_eyes_failed)
        self._fix_eyes_pool.start(task)

    def _on_fix_eyes_finished(self, job_id, result, fixed_count, other_count):
        self._fix_eyes_task = None
        if job_id != self._fix_eyes_job_id:
            return  # superseded by a newer request (e.g. image switched mid-analysis)
        self.fix_eyes_btn.setEnabled(True)
        self._fix_eyes_original = self.full_array
        self._fix_eyes_result = result
        self._fix_eyes_active = True
        self._swap_full_array(result)
        self._update_fix_eyes_btn()
        plural = "" if fixed_count == 1 else "s"
        self.statusBar().showMessage(f"Fix Eyes: fixed {fixed_count} face{plural} from {other_count} nearby frame(s)")

    def _on_fix_eyes_no_result(self, job_id, message):
        self._fix_eyes_task = None
        if job_id != self._fix_eyes_job_id:
            return
        self.fix_eyes_btn.setEnabled(True)
        self.statusBar().showMessage(message)

    def _on_fix_eyes_failed(self, job_id, message):
        self._fix_eyes_task = None
        if job_id != self._fix_eyes_job_id:
            return
        self.fix_eyes_btn.setEnabled(True)
        self.statusBar().showMessage(f"Fix Eyes failed: {self._friendly_error_message(RuntimeError(message))}")

    def _reset_deep_denoise_cache(self):
        """Drop the per-image denoise cache (called on image load / source-pixel change) and
        reset the button. The background full-res pass is left to finish; its path guard
        discards a result that no longer matches the open image."""
        self._deep_denoise_enabled = False
        self._deep_denoise_preview = None
        self._deep_denoise_interactive = None
        self._deep_denoise_full = None
        self._update_deep_denoise_btn()
        if hasattr(self, "denoise_point_before_label"):
            self._set_denoise_point_thumb(self.denoise_point_before_label, None)
        if hasattr(self, "denoise_point_after_label"):
            self._set_denoise_point_thumb(self.denoise_point_after_label, None)

    def _update_deep_denoise_btn(self):
        """Label reflects state: running -> 'Denoising…'; off -> 'Deep Denoise'; on -> 'Denoise ✓'
        (click to turn off). Toggling an already-computed denoise on/off is instant."""
        btn = getattr(self, "deep_denoise_btn", None)
        if btn is None:
            return
        if self._deep_denoise_running:
            btn.setText("Denoising…")
        elif self._deep_denoise_enabled:
            btn.setText("Denoise ✓")
        else:
            btn.setText("Deep Denoise")

    def _run_deep_denoise(self):
        """Deep denoise (NAFNet) at full preview resolution -- no downscale, so the preview is
        sharp/true to the model (a downscale-then-upscale proxy blurred it). NAFNet is heavy, so
        this takes a while (progress shows on the button); it's cached, so toggling off/on after
        is instant. A full-resolution pass runs in the background and is what export uses."""
        if self.preview_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        if self._deep_denoise_running:
            return

        # Already computed for this image -> instant on/off toggle, no recompute.
        if self._deep_denoise_preview is not None:
            self._deep_denoise_enabled = not self._deep_denoise_enabled
            self._mark_preview_pixels_changed()  # base pixels change -> invalidate stage cache
            self._update_deep_denoise_btn()
            self._schedule_render()
            if self._deep_denoise_enabled:
                self.statusBar().showMessage("Deep Denoise on (full-resolution applied at export)")
                self._start_deep_denoise_full()  # ensure the export-quality pass is underway
            else:
                self.statusBar().showMessage("Deep Denoise off")
            return

        deep = getattr(self, "_deep_denoiser", None)
        if deep is None or not deep.available:
            self.statusBar().showMessage("Deep Denoise: model not installed (see System Check)")
            return

        self._deep_denoise_running = True
        self._deep_denoise_path = self.file_path  # guard: only apply if still on this image
        source = self.preview_array.copy()  # full preview res, no downscale; thread only reads it
        self.deep_denoise_btn.setEnabled(False)
        self._update_deep_denoise_btn()
        self.statusBar().showMessage("Deep Denoise: previewing at full resolution…")

        def _worker():
            try:
                result = deep.denoise(source, progress=lambda d, t: self._deep_denoise_progress.emit(d, t))
            except Exception:
                result = None
            self._deep_denoise_done.emit(result)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_deep_denoise_progress(self, done, total):
        pct = int(round(100.0 * done / max(1, total)))
        # Surface progress on the button too, not just the status bar -- a static "Denoising…"
        # for 10-15s reads as frozen.
        btn = getattr(self, "deep_denoise_btn", None)
        if btn is not None and self._deep_denoise_running:
            btn.setText(f"Denoising {pct}%")
        region_btn = getattr(self, "deep_denoise_region_btn", None)
        if region_btn is not None and self._deep_denoise_region_running:
            region_btn.setText(f"Mask {pct}%")
        point_btn = getattr(self, "denoise_point_btn", None)
        if point_btn is not None and self._denoise_point_running:
            point_btn.setText(f"Point {pct}%")
        self.statusBar().showMessage(f"Deep Denoise: previewing {pct}% ({done}/{total} tiles)…")

    def _on_deep_denoise_done(self, result):
        self._deep_denoise_running = False
        self.deep_denoise_btn.setEnabled(True)
        if result is None:
            self.statusBar().showMessage("Deep Denoise: failed (see System Check)")
            self._update_deep_denoise_btn()
            return
        # If the user switched images while it ran, the result is for the old image -- discard.
        if self.file_path != getattr(self, "_deep_denoise_path", self.file_path):
            self.statusBar().showMessage("Deep Denoise: finished, but you switched images — discarded")
            self._update_deep_denoise_btn()
            return
        # Result is the denoised preview at full preview resolution (no downscale). Cache it
        # (settled + interactive res) and switch it on live.
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        self._deep_denoise_preview = result
        self._deep_denoise_interactive, _ = self._build_interactive_preview_proxy(result)
        self._deep_denoise_enabled = True
        self._mark_preview_pixels_changed()
        self._update_deep_denoise_btn()
        self._schedule_render()
        self.statusBar().showMessage("Deep Denoise on (full-resolution applied at export)")
        self._start_deep_denoise_full()

    def _active_denoise_region_mask(self):
        if self.preview_array is None or self.preview_masks is None:
            return None, ""
        layer = self._active_layer
        if layer not in MASK_ORDER or layer not in self.preview_masks:
            return None, ""
        mask = np.clip(np.asarray(self.preview_masks[layer], dtype=np.float32), 0.0, 1.0)
        if mask.shape[:2] != self.preview_array.shape[:2]:
            mask = self._scale_masks({layer: mask}, self.preview_array.shape[:2]).get(layer)
            if mask is None:
                return None, ""
        if float(np.max(mask)) < 0.05:
            return None, layer
        return mask, layer

    def _run_deep_denoise_region(self):
        """Preview-only region denoise for fast live testing.

        The active mask chooses the region. We denoise only its padded bounding box and blend
        through a softly feathered copy of that mask, leaving the full-resolution export source
        untouched until the workflow is proven useful.
        """
        if self.preview_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        if self._deep_denoise_running or self._deep_denoise_region_running:
            return
        deep = getattr(self, "_deep_denoiser", None)
        if deep is None or not deep.available:
            self.statusBar().showMessage("Denoise Mask: model not installed (see System Check)")
            return
        mask, layer = self._active_denoise_region_mask()
        if mask is None:
            self.statusBar().showMessage("Denoise Mask: choose a mask layer with a non-empty mask first")
            return

        selected = mask > 0.03
        ys, xs = np.where(selected)
        if len(xs) == 0 or len(ys) == 0:
            self.statusBar().showMessage(f"Denoise Mask: {layer} mask is empty")
            return
        h, w = self.preview_array.shape[:2]
        pad = 64
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(h, int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(w, int(xs.max()) + pad + 1)
        if y1 <= y0 or x1 <= x0:
            self.statusBar().showMessage(f"Denoise Mask: {layer} region is empty")
            return

        source = self._deep_denoise_preview if self._deep_denoise_enabled and self._deep_denoise_preview is not None else self.preview_array
        source = source.copy()
        crop = source[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].copy()
        self._deep_denoise_region_running = True
        self._deep_denoise_path = self.file_path
        btn = getattr(self, "deep_denoise_region_btn", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("Denoising Mask...")
        self.statusBar().showMessage(f"Denoise Mask: {layer} crop {x1 - x0}x{y1 - y0}...")

        def _worker():
            try:
                denoised_crop = deep.denoise(crop, progress=lambda d, t: self._deep_denoise_progress.emit(d, t))
                if denoised_crop is None:
                    self._deep_denoise_region_done.emit(None)
                    return
                alpha = smooth_mask(crop_mask, sigma=6.0, acceleration=self._runtime_settings.get("acceleration_mode", "auto"))
                alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)[:, :, None]
                result = source.copy()
                result[y0:y1, x0:x1] = np.clip(
                    crop * (1.0 - alpha) + denoised_crop * alpha,
                    0.0,
                    1.0,
                )
                self._deep_denoise_region_done.emit(result.astype(np.float32))
            except Exception:
                self._deep_denoise_region_done.emit(None)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_deep_denoise_region_done(self, result):
        self._deep_denoise_region_running = False
        btn = getattr(self, "deep_denoise_region_btn", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText("Denoise Mask")
        if result is None:
            self.statusBar().showMessage("Denoise Mask: failed")
            return
        if self.file_path != getattr(self, "_deep_denoise_path", self.file_path):
            self.statusBar().showMessage("Denoise Mask: finished, but you switched images -- discarded")
            return
        self._deep_denoise_preview = np.clip(result, 0.0, 1.0).astype(np.float32)
        self._deep_denoise_interactive, _ = self._build_interactive_preview_proxy(self._deep_denoise_preview)
        self._deep_denoise_enabled = True
        self._mark_preview_pixels_changed()
        self._update_deep_denoise_btn()
        self._schedule_render()
        self.statusBar().showMessage("Denoise Mask applied to preview (export/full-res unchanged)")

    def _on_denoise_point_clicked(self, nx: float, ny: float):
        """Denoise Point: one click -> deep-denoise a small square centered there, live preview
        only. Lets you judge detail loss/recovery at a specific spot fast, without waiting for a
        full-image or full-mask pass. The tool stays armed after a click so you can probe
        several spots in a row."""
        if self.preview_array is None:
            self.statusBar().showMessage("Open an image first")
            return
        if self._deep_denoise_running or self._deep_denoise_region_running or self._denoise_point_running:
            return
        deep = getattr(self, "_deep_denoiser", None)
        if deep is None or not deep.available:
            self.statusBar().showMessage("Denoise Point: model not installed (see System Check)")
            return

        h, w = self.preview_array.shape[:2]
        px = int(round(float(np.clip(nx, 0.0, 1.0)) * (w - 1)))
        py = int(round(float(np.clip(ny, 0.0, 1.0)) * (h - 1)))
        half = 128
        x0 = max(0, px - half)
        x1 = min(w, px + half)
        y0 = max(0, py - half)
        y1 = min(h, py + half)
        if (x1 - x0) < 32 or (y1 - y0) < 32:
            self.statusBar().showMessage("Denoise Point: too close to the edge -- click further inside the image")
            return

        source = self._deep_denoise_preview if self._deep_denoise_enabled and self._deep_denoise_preview is not None else self.preview_array
        source = source.copy()
        crop = source[y0:y1, x0:x1].copy()
        self._denoise_point_running = True
        self._deep_denoise_path = self.file_path
        btn = getattr(self, "denoise_point_btn", None)
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("Denoising...")
        self.statusBar().showMessage(f"Denoise Point: {x1 - x0}x{y1 - y0} square at ({px},{py})...")
        # Show "Before" the instant the click registers -- it needs no inference, and waiting
        # for the worker to emit it alongside "After" made the sidebar look unresponsive.
        if hasattr(self, "denoise_point_before_label"):
            self._set_denoise_point_thumb(self.denoise_point_before_label, crop)
        if hasattr(self, "denoise_point_after_label"):
            self.denoise_point_after_label.setPixmap(QPixmap())
            self.denoise_point_after_label.setText("…")

        def _worker():
            try:
                denoised_crop = deep.denoise(crop, progress=lambda d, t: self._deep_denoise_progress.emit(d, t))
                if denoised_crop is None:
                    self._denoise_point_preview_ready.emit(None)
                    self._denoise_point_done.emit(None)
                    return
                # Raw denoiser output, before the edge feather blends it back into the
                # surroundings -- this is what the "After" thumbnail shows, since that's the
                # full-strength result the user actually wants to judge.
                self._denoise_point_preview_ready.emit(denoised_crop)
                # Feathered square: zero out a thin border, then blur it so the patch fades to
                # 0 exactly at the crop edge -- it blends into the untouched surroundings
                # instead of leaving a visible seam, with no mask object to drive the blend.
                ch, cw = crop.shape[:2]
                margin = max(4, min(24, ch // 4, cw // 4))
                base = np.ones((ch, cw), dtype=np.float32)
                base[:margin, :] = 0.0
                base[-margin:, :] = 0.0
                base[:, :margin] = 0.0
                base[:, -margin:] = 0.0
                alpha = smooth_mask(base, sigma=margin / 2.0, acceleration=self._runtime_settings.get("acceleration_mode", "auto"))
                alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)[:, :, None]
                result = source.copy()
                result[y0:y1, x0:x1] = np.clip(
                    crop * (1.0 - alpha) + denoised_crop * alpha,
                    0.0,
                    1.0,
                )
                self._denoise_point_done.emit(result.astype(np.float32))
            except Exception:
                self._denoise_point_done.emit(None)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_denoise_point_done(self, result):
        self._denoise_point_running = False
        btn = getattr(self, "denoise_point_btn", None)
        if btn is not None:
            btn.setEnabled(True)
            btn.setText("Denoise Point")
        if result is None:
            self.statusBar().showMessage("Denoise Point: failed")
            return
        if self.file_path != getattr(self, "_deep_denoise_path", self.file_path):
            self.statusBar().showMessage("Denoise Point: finished, but you switched images -- discarded")
            return
        self._deep_denoise_preview = np.clip(result, 0.0, 1.0).astype(np.float32)
        self._deep_denoise_interactive, _ = self._build_interactive_preview_proxy(self._deep_denoise_preview)
        self._deep_denoise_enabled = True
        self._mark_preview_pixels_changed()
        self._update_deep_denoise_btn()
        self._schedule_render()
        self.statusBar().showMessage(
            "Denoise Point applied to preview (export/full-res unchanged) -- click elsewhere to test another spot."
        )

    def _set_denoise_point_thumb(self, label, arr):
        """Fill one sidebar before/after thumbnail with a center-cropped, square-scaled view of
        a denoise-point crop, or reset it to the empty placeholder when arr is None."""
        if arr is None:
            label.setPixmap(QPixmap())
            label.setText("--")
            return
        size = label.width()
        pil = Image.fromarray(to_uint8(np.clip(arr, 0.0, 1.0))).convert("RGB")
        w, h = pil.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        pil = pil.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR)
        data = pil.tobytes("raw", "RGB")
        qimage = QImage(data, size, size, size * 3, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimage))
        label.setText("")

    def _on_denoise_point_preview_ready(self, after):
        if hasattr(self, "denoise_point_after_label"):
            self._set_denoise_point_thumb(self.denoise_point_after_label, after)

    def _start_deep_denoise_full(self):
        """Kick off the full-resolution NAFNet pass in the background so it's ready (cached) by
        export time -- the user keeps editing on the fast proxy meanwhile. Cheap to call
        repeatedly: it no-ops if the result is already cached or a pass is already running."""
        if self._deep_denoise_full is not None or self._deep_denoise_full_running:
            return
        if self.full_array is None:
            return
        deep = getattr(self, "_deep_denoiser", None)
        if deep is None or not deep.available:
            return
        self._deep_denoise_full_running = True
        path = self.file_path
        source = self.full_array  # read-only hand-off
        deep_ref = deep

        def _worker():
            try:
                result = deep_ref.denoise(source)
            except Exception:
                result = None
            self._deep_denoise_full_done.emit(path or "", result)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_deep_denoise_full_done(self, path, result):
        self._deep_denoise_full_running = False
        # Discard a result whose image is no longer open, or that failed.
        if result is None or path != (self.file_path or ""):
            return
        self._deep_denoise_full = result

    def _swap_full_array(self, arr):
        """Replace the source pixels with `arr`, rebuild the downscaled proxies, drop cached
        masks so they re-derive, and re-render. Shared by the denoise apply/revert/re-apply."""
        self._mark_source_pixels_modified()
        self.full_array = arr
        preview, scale = self._build_preview_proxy(arr)
        interactive_preview, interactive_ratio = self._build_interactive_preview_proxy(preview)
        self.preview_array = preview
        self.preview_scale = scale
        self.image_label.set_full_res_ratio(1.0 / scale if scale > 0 else 1.0)
        self.image_label.set_hires_tile(None, None)
        self._interactive_preview_array = interactive_preview
        self._interactive_preview_ratio = interactive_ratio
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self._mask_face_index = None
        self._schedule_render()

    def _populate_filmstrip(self):
        """Populate the filmstrip with the active collection's images."""
        # Clear old filmstrip.
        for widget in self._filmstrip_items.values():
            widget.deleteLater()
        self._filmstrip_items.clear()
        if self._filmstrip_placeholder is not None:
            self._filmstrip_placeholder.deleteLater()
            self._filmstrip_placeholder = None
        self._filmstrip_images = list(self._imported_images)
        valid_paths = set(self._filmstrip_images)
        self._filmstrip_selected_paths.intersection_update(valid_paths)
        if self._filmstrip_selection_anchor not in valid_paths:
            self._filmstrip_selection_anchor = None
        self._import_queue = [p for p in self._import_queue if p in self._filmstrip_images]

        if not self._filmstrip_images:
            self._filmstrip_selected_paths.clear()
            self._filmstrip_selection_anchor = None
            self._update_filmstrip_filter_buttons()
            self._update_filmstrip_count_label(0, 0)
            placeholder = QWidget(self)
            placeholder_layout = QHBoxLayout(placeholder)
            placeholder_layout.setContentsMargins(0, 0, 0, 0)
            placeholder_layout.setSpacing(8)
            placeholder_label = QLabel("No images yet.", self)
            placeholder_label.setObjectName("MutedLabel")
            placeholder_layout.addWidget(placeholder_label)
            # A real button, not just a keyboard-shortcut hint -- a first-time user has no
            # reason to already know Ctrl+I does anything.
            placeholder_import_btn = QPushButton("Import Images...", self)
            placeholder_import_btn.clicked.connect(self._import_folder)
            placeholder_layout.addWidget(placeholder_import_btn)
            placeholder_layout.addStretch(1)
            self._filmstrip_layout.addWidget(placeholder)
            self._filmstrip_placeholder = placeholder
            self._update_empty_state()
            return

        # Create thumbnails for imported images.
        collection = self._collections.get(self._active_collection, {}) if self._active_collection else {}
        overrides = collection.get("image_overrides", {})
        culled_paths = set(collection.get("culled_images", []))
        shown_count = 0
        for img_path in self._filmstrip_images:
            if not Path(img_path).is_file():
                continue
            if not self._filmstrip_filter_matches(img_path, overrides, culled_paths):
                continue
            thumb = FilmstripThumbnail(img_path, self)
            thumb.clicked.connect(self._on_filmstrip_image_clicked)
            thumb.remove_requested.connect(self._remove_image_from_collection)
            thumb.set_has_override(img_path in overrides)
            state, preset_name = self._preset_state_for_override(overrides.get(img_path))
            thumb.set_preset_state(state, preset_name)
            thumb.set_culled(img_path in culled_paths)
            export_status = self._filmstrip_export_status.get(os.path.abspath(str(img_path)), "")
            thumb.set_export_status(export_status)
            thumb.set_export_failed(os.path.abspath(str(img_path)) in self._filmstrip_failed_paths)
            self._filmstrip_layout.addWidget(thumb)
            self._filmstrip_items[img_path] = thumb
            shown_count += 1

            # Use cached thumbnail if available, otherwise queue it for background load.
            cached_thumb = self._get_cached_thumbnail(img_path)
            if cached_thumb is not None:
                thumb.set_pixmap(cached_thumb)
            elif img_path not in self._import_queue:
                self._import_queue.append(img_path)

        if shown_count == 0:
            placeholder = QLabel(f"No images match filter: {self._filmstrip_filter}", self)
            placeholder.setObjectName("MutedLabel")
            self._filmstrip_layout.addWidget(placeholder)
            self._filmstrip_placeholder = placeholder

        self._filmstrip_layout.addStretch(1)
        self._update_filmstrip_active()
        self._update_filmstrip_filter_buttons()
        self._update_filmstrip_count_label(shown_count, len(self._filmstrip_images))

        # Start the import worker if not running.
        if self._import_queue and not self._import_worker_timer.isActive():
            self._import_worker_timer.start()
        self._update_empty_state()

    def _import_folder(self):
        """Open import dialog and import images from a folder into a collection."""
        dialog = ImportDialog(self, self.SUPPORTED_IMAGE_EXTS, existing_collections=list(self._collections.keys()))
        if dialog.exec() != QDialog.Accepted:
            return

        folder = dialog.selected_folder()
        collection_name = dialog.collection_name()
        if not folder or not collection_name:
            return

        # Reuse the dialog's own (already backgrounded, already recursive-aware) scan instead
        # of repeating it synchronously here -- avoids a second multi-second folder walk on
        # the UI thread right after the one the dialog just did off it.
        all_paths = dialog.scanned_paths()

        collection = self._collections.setdefault(
            collection_name,
            {
                "images": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_folder": folder,
                "image_overrides": {},
                "image_quality": {},
                "culled_images": [],
            },
        )
        existing = set(collection["images"])
        new_images = [p for p in all_paths if p not in existing]
        if not new_images:
            self.statusBar().showMessage(f"No new images to import into \"{collection_name}\"")
            return

        collection["images"].extend(new_images)
        collection["source_folder"] = folder

        self._active_collection = collection_name
        self._import_queue.extend(new_images)
        self._refresh_collection_combo()
        self._apply_active_collection()
        self._save_collections_state()

        if not self._import_worker_timer.isActive():
            self._import_worker_timer.start()

        self.statusBar().showMessage(
            f"Imported {len(new_images)} new image(s) into \"{collection_name}\" ({len(collection['images'])} total)"
        )

    def _remove_image_from_collection(self, path: str):
        """Remove an image from the active collection (does not delete the file)."""
        if not self._active_collection or self._active_collection not in self._collections:
            return
        collection = self._collections[self._active_collection]
        images = collection.get("images", [])
        if path not in images:
            return
        reply = QMessageBox.question(
            self,
            "Remove from Collection",
            f"Remove \"{os.path.basename(path)}\" from \"{self._active_collection}\"? "
            "This only removes it from the collection (and discards any edits saved for it "
            "here) -- the image file on disk is not affected.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        images.remove(path)
        self._filmstrip_selected_paths.discard(path)
        if self._filmstrip_selection_anchor == path:
            self._filmstrip_selection_anchor = None
        collection.get("image_overrides", {}).pop(path, None)
        normalized_path = os.path.abspath(str(path))
        self._filmstrip_failed_paths.discard(normalized_path)
        self._filmstrip_export_status.pop(normalized_path, None)
        self._save_collections_state()
        self._refresh_collection_combo()
        self._apply_active_collection()
        self.statusBar().showMessage(f"Removed from \"{self._active_collection}\"")

    def _review_and_cull(self):
        """Group burst/near-duplicate frames and flag soft (blurry) ones, then let the user
        move rejects into the collection's culled list (never deletes files)."""
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        images = list(collection.get("images", []))
        if len(images) < 2:
            QMessageBox.information(self, "Review / Cull", "Need at least two images to review.")
            return
        quality = collection.get("image_quality", {})
        missing = [p for p in images if p not in quality or quality[p].get("ahash") is None]
        if missing:
            QMessageBox.information(
                self, "Review / Cull",
                f"{len(missing)} image(s) are still being analyzed (import in progress). "
                "Wait for thumbnails to finish, then try again.",
            )
            return

        items = [{"path": p, "ahash": quality[p].get("ahash"), "time": quality[p].get("time", 0.0)} for p in images]
        groups = culling_ops.group_bursts(items)
        # Blink analysis only matters for keeper choice inside multi-frame bursts -- run it
        # there (cached, so re-opening is instant) before computing suggestions.
        self._analyze_blink_for_burst_groups(groups, quality)
        blurs = [float(quality[p].get("blur", 0.0)) for p in images]
        median_blur = float(np.median(blurs)) if blurs else 0.0
        soft_threshold = 0.5 * median_blur

        # Keeper suggestion per burst, and soft singles flagged separately.
        suggestions = {}  # path -> "keeper" | "cull" | "soft"
        for g in groups:
            if len(g) > 1:
                keeper = culling_ops.pick_keeper(g, quality)
                for p in g:
                    suggestions[p] = "keeper" if p == keeper else "cull"
            else:
                p = g[0]
                if float(quality[p].get("blur", 0.0)) < soft_threshold:
                    suggestions[p] = "soft"

        has_bursts = any(len(g) > 1 for g in groups)
        has_soft = any(v == "soft" for v in suggestions.values())
        if not has_bursts and not has_soft:
            QMessageBox.information(
                self, "Review / Cull",
                "No burst duplicates or notably soft frames found — nothing to cull.",
            )
            return

        dialog = ReviewCullDialog(self, groups, quality, suggestions, self._get_cached_thumbnail)
        if dialog.exec() != QDialog.Accepted:
            return
        to_cull = dialog.culled_paths()
        if not to_cull:
            self.statusBar().showMessage("Review / Cull: nothing culled")
            return

        culled_list = collection.setdefault("culled_images", [])
        imgs = collection["images"]
        for p in to_cull:
            if p in imgs:
                imgs.remove(p)
            if p not in culled_list:
                culled_list.append(p)
            normalized_path = os.path.abspath(str(p))
            self._filmstrip_failed_paths.discard(normalized_path)
            self._filmstrip_export_status.pop(normalized_path, None)
            self._filmstrip_selected_paths.discard(p)
        self._save_collections_state()
        self._refresh_collection_combo()
        self._apply_active_collection()
        self.statusBar().showMessage(
            f"Culled {len(to_cull)} image(s) into the collection's culled list (files not deleted)"
        )

    def _view_culled_images(self):
        if not self._active_collection or self._active_collection not in self._collections:
            QMessageBox.information(self, "No Collection", "Select a collection first.")
            return
        collection = self._collections[self._active_collection]
        culled_list = collection.get("culled_images", [])
        if not culled_list:
            QMessageBox.information(self, "Culled Images", "This collection has no culled images.")
            return

        dialog = CulledImagesDialog(self, culled_list, self._get_cached_thumbnail)
        dialog.exec()
        restored = dialog.restored_paths()
        if not restored:
            return
        imgs = collection.setdefault("images", [])
        for p in restored:
            if p in culled_list:
                culled_list.remove(p)
            if p not in imgs:
                imgs.append(p)
        self._save_collections_state()
        self._refresh_collection_combo()
        self._apply_active_collection()
        self.statusBar().showMessage(f"Restored {len(restored)} image(s) back into \"{self._active_collection}\"")

    def _analyze_blink_for_burst_groups(self, groups, quality):
        """Run blink detection on the frames of each multi-image burst group that hasn't been
        analyzed yet, caching blinks/eyes_open into `quality`. Full-res face analysis is slow,
        so this shows a cancelable progress dialog and persists results so it only runs once."""
        model = getattr(self.segmenter, "_model", None)
        blink_backend = getattr(model, "landmark_backend", "none")
        blink_version = culling_ops.BLINK_ANALYSIS_VERSION
        targets = [
            p for g in groups if len(g) > 1 for p in g
            if (
                not quality.get(p, {}).get("blink_checked")
                or quality.get(p, {}).get("blink_backend") != blink_backend
                or quality.get(p, {}).get("blink_version") != blink_version
            )
        ]
        if not targets:
            return
        if blink_backend == "none":
            for path in targets:
                q = quality.setdefault(path, {})
                q["blink_checked"] = True
                q["blink_backend"] = blink_backend
                q["blink_version"] = blink_version
                q["blink_faces"] = 0
                q["blinks"] = None
                q["eyes_open"] = None
            self._save_collections_state()
            return
        dlg = QProgressDialog("Analyzing eyes (blink detection)…", "Skip", 0, len(targets), self)
        dlg.setWindowTitle("Review / Cull")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        analyzed_any = False
        for i, path in enumerate(targets):
            if dlg.wasCanceled():
                break
            try:
                img = self._read_image_file(path)
                blinks, n = culling_ops.analyze_face_blinks(self.segmenter, img)
                q = quality.setdefault(path, {})
                q["blink_checked"] = True
                q["blink_backend"] = blink_backend
                q["blink_version"] = blink_version
                q["blink_faces"] = int(n)
                q["blinks"] = int(blinks) if n > 0 else None
                q["eyes_open"] = (blinks == 0) if n > 0 else None
                analyzed_any = True
            except Exception:
                pass
            # A modal QProgressDialog pumps the event loop itself on setValue (repaint +
            # Cancel handling); an explicit processEvents() here re-enters the window's timers
            # and can deadlock, so we rely on setValue alone.
            dlg.setValue(i + 1)
        dlg.close()
        if analyzed_any:
            self._save_collections_state()

    def _get_cached_thumbnail(self, img_path: str):
        """Get a disk-cached thumbnail pixmap if available, None otherwise."""
        cache_path = self._thumbnail_cache_path(img_path)
        if cache_path.is_file():
            pixmap = QPixmap(str(cache_path))
            if not pixmap.isNull():
                return pixmap
        return None

    def _process_import_queue(self):
        """Process next image in import queue (called by timer every 500ms)."""
        if not self._import_queue:
            self._import_worker_timer.stop()
            # Cull signals (blur/ahash) finished computing during import -- persist them.
            self._save_collections_state()
            if self._imported_images:
                self.statusBar().showMessage(f"Imported {len(self._imported_images)} images")
            return

        img_path = self._import_queue.pop(0)
        if img_path not in self._filmstrip_items:
            # Image not in filmstrip yet (shouldn't happen, but handle gracefully).
            return

        # Decode + cache the thumbnail off the GUI thread, then hand back to the main thread
        # (via _thumbnail_ready) to build the QPixmap there -- QPixmap is not thread-safe.
        def decode_and_cache():
            if self._write_thumbnail_cache(img_path):
                self._thumbnail_ready.emit(img_path)

        import threading
        thread = threading.Thread(target=decode_and_cache, daemon=True)
        thread.start()

        # Update status bar with progress.
        remaining = len(self._import_queue)
        total = len(self._imported_images)
        processed = total - remaining
        self.statusBar().showMessage(f"Importing: {processed}/{total} thumbnails...")

    def _on_thumbnail_ready(self, img_path: str):
        """GUI-thread slot: load the freshly cached thumbnail and show it on the filmstrip,
        and persist any cull signals the worker stashed for this image."""
        quality = self._import_quality_buffer.pop(img_path, None)
        if quality is not None and self._active_collection in self._collections:
            self._collections[self._active_collection].setdefault("image_quality", {})[img_path] = quality
        if img_path not in self._filmstrip_items:
            return
        pixmap = self._get_cached_thumbnail(img_path)
        if pixmap is not None:
            self._filmstrip_items[img_path].set_pixmap(pixmap)

    def _refresh_active_thumbnail(self):
        """Regenerate the open image's filmstrip thumbnail from the edited preview, so it
        reflects the current adjustments (global/selective/framing) rather than the original.
        Runs on the GUI thread, so building the QPixmap here is safe."""
        path = self.file_path
        if not path or path not in self._filmstrip_items or self.preview_image is None:
            return
        try:
            edited = self.preview_image.convert("RGB")
            edited = apply_framing(edited, normalize_framing(self._framing))
            edited.thumbnail((200, 200), Image.LANCZOS)
            cache_path = self._thumbnail_cache_path(path)
            edited.save(str(cache_path), "JPEG", quality=85)
            pixmap = QPixmap(str(cache_path))
            if not pixmap.isNull():
                self._filmstrip_items[path].set_pixmap(pixmap)
        except Exception:
            pass

    def _write_thumbnail_cache(self, img_path: str) -> bool:
        """Decode an image, downscale it, and write the JPEG thumbnail to the disk cache.
        Safe to run on a worker thread: it touches no Qt GUI objects (no QPixmap/widget) and
        uses PIL (not cv2, which isn't imported here) for the resize/encode.

        Also computes the model-free culling signals (blur + average hash) from the freshly
        decoded image while we have it, stashing them in a buffer the GUI thread drains into
        the collection -- so import builds the cull data for free, with no extra decode."""
        try:
            ext = Path(img_path).suffix.lower()
            if is_raw_path(img_path):
                # Settings-less thumbnail/culling decode -- camera-WB defaults (color_settings
                # None) match the per-image edit decode's defaults. If rawpy is missing,
                # decode_raw raises and the outer except returns False, as before.
                img8 = (np.clip(decode_raw(img_path)[0], 0.0, 1.0) * 255).astype(np.uint8)
                pil = Image.fromarray(img8, mode="RGB")
            else:
                pil = Image.open(img_path).convert("RGB")

            try:
                self._import_quality_buffer[img_path] = {
                    "blur": culling_ops.blur_score(img8 if is_raw_path(img_path) else np.asarray(pil)),
                    "ahash": culling_ops.average_hash(img8 if is_raw_path(img_path) else np.asarray(pil)),
                    "time": culling_ops.capture_time(img_path),
                    "eyes_open": None,
                    "blinks": None,
                    "blink_faces": 0,
                    "blink_checked": False,
                    "blink_backend": "none",
                    "blink_version": culling_ops.BLINK_ANALYSIS_VERSION,
                }
            except Exception:
                pass

            # Downscale in place, preserving aspect ratio (max 200px long edge).
            pil.thumbnail((200, 200), Image.LANCZOS)
            pil.save(str(self._thumbnail_cache_path(img_path)), "JPEG", quality=85)
            return True
        except Exception:
            return False

    def _update_filmstrip_active(self):
        """Highlight the current image in the filmstrip."""
        active_thumb = None
        visible_paths = set(self._filmstrip_items.keys())
        self._filmstrip_selected_paths.intersection_update(visible_paths)
        for path, thumb in self._filmstrip_items.items():
            is_active = path == self.file_path
            thumb.set_active(is_active)
            thumb.set_selected(path in self._filmstrip_selected_paths)
            if is_active:
                active_thumb = thumb
        self._current_filmstrip_path = self.file_path
        self._update_filmstrip_count_label()
        # Keep the active thumbnail on-screen -- without this, selecting an image near either
        # end of a long filmstrip can leave its own highlighted thumbnail scrolled out of view.
        if active_thumb is not None and hasattr(self, "_filmstrip_scroll"):
            self._filmstrip_scroll.ensureWidgetVisible(active_thumb)

    def _visible_filmstrip_paths(self):
        return [path for path in self._filmstrip_images if path in self._filmstrip_items]

    def _selected_filmstrip_paths(self):
        paths = [path for path in self._visible_filmstrip_paths() if path in self._filmstrip_selected_paths]
        if paths:
            return paths
        if self.file_path and self.file_path in self._filmstrip_items:
            return [self.file_path]
        return []

    def _queue_background_analysis_for_paths(self, paths):
        """Accumulate newly-selected filmstrip paths for background preload, debounced so a
        fast multi-select/drag coalesces into one subprocess launch instead of one per path."""
        queued = 0
        for path in paths:
            if not path or path in self._analysis_preload_in_flight:
                continue
            if not os.path.isfile(path):
                continue
            self._analysis_preload_in_flight.add(path)
            self._analysis_preload_pending.add(path)
            queued += 1
        if queued:
            now = time.time()
            if now - self._analysis_preload_last_status > 0.5:
                self.statusBar().showMessage(f"Queued background analysis for {queued} selected image(s)")
                self._analysis_preload_last_status = now
            self._analysis_preload_debounce_timer.start()

    def _flush_analysis_preload_queue(self):
        """Debounce timeout: dispatch every path accumulated since the last flush to a single
        detached analysis_preload_runner subprocess (own OS process, own GIL -- see the
        comment on _analysis_preload_in_flight in __init__ for why that matters here)."""
        if not self._analysis_preload_pending:
            return
        paths = sorted(self._analysis_preload_pending)
        self._analysis_preload_pending = set()

        items = []
        for path in paths:
            override = self._get_collection_image_override(path) or {}
            face_index = int(override.get("active_face_index", 0))
            if path == self.file_path:
                face_index = int(self._active_face_index)
            items.append({"path": os.path.abspath(path), "face_index": face_index})

        self._analysis_preload_job_counter += 1
        job_id = f"preload_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{self._analysis_preload_job_counter}"
        jobs_dir = self._collections_dir() / ".preload_jobs"
        jobs_dir.mkdir(exist_ok=True)
        log_path = jobs_dir / f"{job_id}.jsonl"
        job_path = jobs_dir / f"{job_id}.json"
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump({"job_id": job_id, "log_path": str(log_path), "items": items}, fh)

        runner_log = jobs_dir / f"{job_id}.stdout.log"
        try:
            with open(runner_log, "ab") as fh:
                subprocess.Popen(
                    [sys.executable, "-m", "portrait_enhancer.analysis_preload_runner", "--job", str(job_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    cwd=os.getcwd(),
                )
        except Exception:
            # Couldn't launch -- don't leave these paths stuck "in flight" forever.
            for path in paths:
                self._analysis_preload_in_flight.discard(path)
            return

        self._analysis_preload_jobs[job_id] = {"log_path": str(log_path), "paths": set(paths), "offset": 0}
        if not self._analysis_preload_poll_timer.isActive():
            self._analysis_preload_poll_timer.start()

    def _poll_analysis_preload_jobs(self):
        """Tail each in-flight preload job's JSONL log (append-only, so re-reading from the
        last byte offset is safe) and clear paths off _analysis_preload_in_flight as their
        process reports them done."""
        if not self._analysis_preload_jobs:
            self._analysis_preload_poll_timer.stop()
            return
        for job_id in list(self._analysis_preload_jobs.keys()):
            state = self._analysis_preload_jobs[job_id]
            try:
                with open(state["log_path"], "r", encoding="utf-8") as fh:
                    fh.seek(state["offset"])
                    new_lines = fh.readlines()
                    state["offset"] = fh.tell()
            except OSError:
                new_lines = []
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                path = record.get("path")
                if not path:
                    continue
                state["paths"].discard(path)
                self._analysis_preload_in_flight.discard(path)
                if record.get("status") == "stored":
                    self.statusBar().showMessage(f"Background analysis cached -> {os.path.basename(path)}")
            if not state["paths"]:
                self._analysis_preload_jobs.pop(job_id, None)
        if not self._analysis_preload_jobs:
            self._analysis_preload_poll_timer.stop()

    def _set_filmstrip_selection(self, paths, anchor=None, queue_analysis: bool = True):
        valid = set(self._visible_filmstrip_paths())
        self._filmstrip_selected_paths = {path for path in paths if path in valid}
        if anchor in valid:
            self._filmstrip_selection_anchor = anchor
        elif self._filmstrip_selection_anchor not in valid:
            self._filmstrip_selection_anchor = None
        self._update_filmstrip_active()
        if queue_analysis and self._filmstrip_selected_paths:
            self._queue_background_analysis_for_paths(self._filmstrip_selected_paths)

    def _select_filmstrip_range(self, path: str, additive: bool = False):
        visible = self._visible_filmstrip_paths()
        if path not in visible:
            return
        anchor = self._filmstrip_selection_anchor if self._filmstrip_selection_anchor in visible else self.file_path
        if anchor not in visible:
            anchor = path
        start = visible.index(anchor)
        end = visible.index(path)
        lo, hi = sorted((start, end))
        selected = set(self._filmstrip_selected_paths) if additive else set()
        selected.update(visible[lo : hi + 1])
        self._set_filmstrip_selection(selected, anchor=anchor)

    def _on_filmstrip_image_clicked(self, path: str):
        """Load the clicked image from the filmstrip."""
        modifiers = QApplication.keyboardModifiers()
        toggle = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
        range_select = bool(modifiers & Qt.ShiftModifier)
        if range_select:
            self._select_filmstrip_range(path, additive=toggle)
            return
        if toggle:
            selected = set(self._filmstrip_selected_paths)
            if path in selected:
                selected.remove(path)
            else:
                selected.add(path)
            self._set_filmstrip_selection(selected, anchor=path)
            return
        self._set_filmstrip_selection([path], anchor=path, queue_analysis=False)
        if path == self.file_path:
            return
        if not Path(path).is_file():
            self.statusBar().showMessage(f"File not found: {path}")
            return
        try:
            self._load_image_path(path)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", self._friendly_error_message(ex))

    def _friendly_error_message(self, ex: Exception) -> str:
        """Translate a handful of common low-level exceptions into plain-language guidance --
        a raw rawpy/CUDA/model-loading exception string means little to someone editing family
        photos rather than debugging the app. Falls back to the original message untouched
        for anything not recognized, so nothing is ever hidden, just clarified when possible."""
        text = str(ex)
        lowered = text.lower()
        if "rawpy is not installed" in lowered:
            return f"RAW file support isn't installed. Install the base dependencies and try again. ({text})"
        if "out of memory" in lowered or ("cuda" in lowered and "memory" in lowered):
            return (
                "Ran out of GPU memory. Try Preferences -> Acceleration -> CPU only, or close "
                f"other GPU-heavy apps, then try again. ({text})"
            )
        if any(s in lowered for s in ("corrupt", "truncated", "unsupported file", "cannot identify image", "not a valid")):
            return f"This file appears to be corrupted or incomplete and couldn't be opened. ({text})"
        if isinstance(ex, FileNotFoundError) or "no such file" in lowered or "not found" in lowered:
            return f"That file couldn't be found -- it may have been moved, renamed, or deleted. ({text})"
        if "model" in lowered and any(s in lowered for s in ("not found", "missing", "no such file")):
            return f"A required AI model file is missing. Use System Check to verify your model installation. ({text})"
        return text

    def _read_image_file(self, path: str):
        full, metadata = _decode_image_file(path, self._color_settings)
        self._source_metadata = metadata
        return full

    def batch_export(self):
        selected_entry = self._selected_browser_entry()
        last_options = self._last_batch_options()
        profiles = self._batch_profiles()
        selected_profile = str(self._browser_state_payload().get("last_batch_profile", ""))
        dialog = BatchExportDialog(
            self,
            preset_path=last_options.get("preset_path") or (str(selected_entry["path"]) if selected_entry is not None else ""),
            input_dir=last_options.get("input_dir", ""),
            output_dir=last_options.get("output_dir", "") or self._preferences.get("default_export_folder", ""),
            suffix=last_options.get("suffix", "_enhanced"),
            output_format=last_options.get("format", "jpeg"),
            skip_completed=bool(last_options.get("skip_completed", True)),
            deep_denoise=bool(last_options.get("deep_denoise", False)),
            profiles=profiles,
            selected_profile=selected_profile,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.options()
        self._save_batch_profiles(options.get("profiles", {}), options.get("selected_profile", ""))
        self._save_last_batch_options(
            {
                "preset_path": options["preset_path"],
                "input_dir": options["input_dir"],
                "output_dir": options["output_dir"],
                "suffix": options["suffix"],
                "format": options["format"],
                "skip_completed": options["skip_completed"],
                "deep_denoise": options.get("deep_denoise", False),
            }
        )
        preset_path = options["preset_path"]
        input_dir = options["input_dir"]
        output_dir = options["output_dir"]
        suffix = options["suffix"]
        output_format = options["format"]
        skip_completed = bool(options["skip_completed"])
        deep_denoise = bool(options.get("deep_denoise", False))

        if not preset_path or not os.path.isfile(preset_path):
            QMessageBox.warning(self, "Batch Export", "Select a valid preset file.")
            return
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "Batch Export", "Select a valid input folder.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Batch Export", "Select a valid output folder.")
            return
        if output_format not in self.BATCH_OUTPUT_FORMATS:
            QMessageBox.warning(self, "Batch Export", "Select a valid output format.")
            return

        try:
            with open(preset_path, "r", encoding="utf-8") as fh:
                preset = json.load(fh)
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))
            return

        image_paths = self._supported_image_paths(input_dir)
        if not image_paths:
            QMessageBox.information(self, "Batch Export", "No supported images found in the selected input folder.")
            return

        if not skip_completed:
            output_ext = self.BATCH_OUTPUT_FORMATS[output_format]
            existing = sum(
                1 for p in image_paths
                if os.path.exists(self._batch_output_path(p, output_dir, suffix=suffix, output_ext=output_ext))
            )
            if existing:
                reply = QMessageBox.question(
                    self,
                    "Files Will Be Overwritten",
                    f"{existing} of {len(image_paths)} output file(s) already exist in this folder "
                    "and will be overwritten (\"Skip already completed\" is off). Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        job_path = self._write_batch_job_file(
            {
                "mode": "batch",
                "preset_path": os.path.abspath(preset_path),
                "input_dir": os.path.abspath(input_dir),
                "output_dir": os.path.abspath(output_dir),
                "suffix": suffix,
                "output_format": output_format,
                "skip_completed": skip_completed,
                "deep_denoise": deep_denoise,
                "runtime_settings": dict(self._runtime_settings),
            },
            output_dir,
        )
        runner_log = self._launch_background_batch_job(job_path, output_dir)
        QMessageBox.information(
            self,
            "Batch Started",
            f"Background batch started.\nYou can close the app.\n\nJob: {job_path}\nLog: {os.path.join(output_dir, 'batch_export_log.jsonl')}\nRunner stdout: {runner_log}",
        )
        self.statusBar().showMessage("Background batch started")

    def retry_failed_batch(self):
        log_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Batch Log",
            "",
            "JSON Lines (*.jsonl);;All Files (*)",
        )
        if not log_path:
            return
        records = self._load_batch_log_records(log_path)
        group = self._latest_failed_batch_records(records)
        if not group:
            QMessageBox.information(self, "Retry Failed", "No failed batch group found in the selected log.")
            return

        sample = group[0]
        preset_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Matching Preset",
            "",
            "Portrait Preset (*.pepreset *.json)",
        )
        if not preset_path:
            return
        try:
            with open(preset_path, "r", encoding="utf-8") as fh:
                preset = json.load(fh)
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", self._friendly_error_message(ex))
            return

        preset_hash = self._preset_hash(preset)
        if sample.get("preset_hash") and sample.get("preset_hash") != preset_hash:
            QMessageBox.warning(
                self,
                "Retry Failed",
                "The selected preset does not match the preset content hash recorded in the failed batch.",
            )
            return

        failed_sources = [r["source_path"] for r in group if r.get("status") == "error" and r.get("source_path")]
        if not failed_sources:
            QMessageBox.information(self, "Retry Failed", "No failed source files found in the selected batch group.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(Path(sample.get("output_path", "")).parent) if sample.get("output_path") else "",
        )
        if not output_dir:
            return

        suffix = str(sample.get("suffix", "_enhanced"))
        output_format = str(sample.get("output_format", "jpeg"))
        job_path = self._write_batch_job_file(
            {
                "mode": "retry_failed",
                "preset_path": os.path.abspath(preset_path),
                "output_dir": os.path.abspath(output_dir),
                "suffix": suffix,
                "output_format": output_format,
                "skip_completed": False,
                "runtime_settings": dict(self._runtime_settings),
                "source_paths": [os.path.abspath(p) for p in failed_sources],
            },
            output_dir,
        )
        runner_log = self._launch_background_batch_job(job_path, output_dir)
        QMessageBox.information(
            self,
            "Retry Started",
            f"Background retry started.\nYou can close the app.\n\nJob: {job_path}\nLog: {os.path.join(output_dir, 'batch_export_log.jsonl')}\nRunner stdout: {runner_log}",
        )
        self.statusBar().showMessage("Background retry started")

    def view_batch_jobs(self):
        start_dir = self._active_export_output_dir()
        if not start_dir:
            start_dir = self._last_batch_options().get("output_dir", "") or self._preferences.get("default_export_folder", "")
        if self.file_path:
            start_dir = start_dir or os.path.dirname(self.file_path)
        dialog = BatchJobsDialog(
            self,
            output_dir=start_dir,
            load_jobs_callback=self._load_batch_jobs,
            cancel_job_callback=self._cancel_batch_job,
        )
        dialog.exec()

    def _active_export_output_dir(self):
        monitors = getattr(self, "_export_monitors", None) or {}
        if not monitors:
            return ""
        latest = max(monitors.values(), key=lambda state: state.get("started_at", 0.0))
        return latest.get("output_dir", "")

    def _run_segmentation(self, on_complete=None, schedule_render: bool = True):
        """Detect faces and compute masks/guides for the active image. Recently visited images
        restore from the foreground RAM cache immediately; cold disk-cache/model work runs off
        the GUI thread on the segmentation pool (default/normal OS priority -- separate from the
        low-priority background preload pool, so this never waits behind queued-but-not-open
        filmstrip selections).

        `on_complete` (optional, no-arg callable) runs on the GUI thread right after the
        result is applied. A later call to `_run_segmentation` before this one finishes
        supersedes it -- its result (and `on_complete`) are silently dropped via job-id
        comparison, exactly like the existing preview-render pool's stale-job handling."""
        if self.preview_array is None or self.full_array is None:
            return
        self._segmentation_job_counter += 1
        job_id = self._segmentation_job_counter
        self._active_segmentation_job_id = job_id
        cache_key = self._segmentation_result_cache_key(self.file_path, self._active_face_index)
        cached_result = self._restore_segmentation_result_cache(cache_key)
        if cached_result is not None:
            self._segmentation_in_flight = False
            self._segmentation_pending_callback = None
            self._apply_segmentation_result(cached_result, schedule_render=schedule_render)
            if on_complete is not None:
                on_complete()
            return
        cached_result = self._load_segmentation_disk_cache_result(self._active_face_index)
        if cached_result is not None:
            self._segmentation_in_flight = False
            self._segmentation_pending_callback = None
            self._remember_segmentation_result_cache(cache_key, cached_result)
            self._apply_segmentation_result(cached_result, schedule_render=schedule_render)
            if on_complete is not None:
                on_complete()
            return
        self._segmentation_in_flight = True
        self._segmentation_pending_callback = on_complete
        task = SegmentationTask(
            job_id=job_id,
            file_path=self.file_path,
            preview_array=self.preview_array,
            full_shape=self.full_array.shape[:2],
            preview_scale=self.preview_scale,
            active_face_index=self._active_face_index,
            segmenter=self.segmenter,
        )
        task.schedule_render = bool(schedule_render)
        task.cache_key = cache_key
        # Pin on self -- without this the only reference is this local variable, and nothing
        # guarantees the Python wrapper survives until the worker thread emits back to the
        # GUI thread (observed directly: an unpinned task here raised "Signal source has been
        # deleted" / segfaulted under GC pressure during the image-load threading work).
        self._segmentation_task = task
        task.signals.finished.connect(self._on_segmentation_finished)
        task.signals.failed.connect(self._on_segmentation_failed)
        self._segmentation_pool.start(task)

    def _on_segmentation_finished(self, job_id: int, result: dict):
        if int(job_id) != self._active_segmentation_job_id:
            return  # superseded by a newer request (e.g. rapid face-index/image switches)
        task = self._segmentation_task
        schedule_render = bool(getattr(task, "schedule_render", True))
        cache_key = getattr(task, "cache_key", None)
        self._segmentation_in_flight = False
        self._segmentation_task = None
        self._remember_segmentation_result_cache(cache_key, result)
        self._apply_segmentation_result(result, schedule_render=schedule_render)
        callback = self._segmentation_pending_callback
        self._segmentation_pending_callback = None
        if callback is not None:
            callback()

    def _on_segmentation_failed(self, job_id: int, message: str):
        if int(job_id) != self._active_segmentation_job_id:
            return
        self._segmentation_in_flight = False
        self._segmentation_task = None
        self.statusBar().showMessage(f"Segmentation failed: {self._friendly_error_message(RuntimeError(message))}")
        callback = self._segmentation_pending_callback
        self._segmentation_pending_callback = None
        if callback is not None:
            callback()

    def _run_mask_diagnostics(self):
        if self.preview_array is None:
            self.statusBar().showMessage("No image loaded -- nothing to diagnose")
            return
        if self._mask_diag_task is not None:
            return
        self.mask_diagnostics_btn.setEnabled(False)
        self.mask_diagnostics_btn.setText("Diagnosing...")
        self.statusBar().showMessage("Diagnosing masks for every detected face -- this re-runs segmentation per face...")
        # Snapshot what the live document is *actually* holding right now, before the fresh
        # recompute runs -- MaskDiagnosticsTask always talks to the segmenter directly and
        # never touches self.preview_masks, so on its own it can only prove the segmenter
        # *can* produce good data, not that the currently-displayed document has it. Comparing
        # the two side by side is what actually pins down a "data" vs. "display" bug.
        self._mask_diag_live_snapshot = (self._active_face_index, self._copy_masks(self.preview_masks))
        # Capture the display-gating flags too -- when the data is confirmed present but Mask
        # View still paints nothing, the cause is one of these flags being in a state that
        # suppresses the overlay (wrong active layer, show-mask off, a compare/edit mode
        # intercepting the canvas). Showing them removes the guesswork.
        self._mask_diag_state = {
            "active_layer": self._active_layer,
            "active_layer_in_MASK_ORDER": self._active_layer in MASK_ORDER,
            "active_layer_in_preview_masks": bool(self.preview_masks) and self._active_layer in self.preview_masks,
            "show_mask": bool(self._show_mask),
            "mask_view_btn_checked": self.mask_view_btn.isChecked() if hasattr(self, "mask_view_btn") else None,
            "mask_edit_enabled": bool(self._mask_edit_enabled),
            "crop_edit_enabled": bool(getattr(self, "_crop_edit_enabled", False)),
            "wb_pick_enabled": bool(getattr(self, "_wb_pick_enabled", False)),
            "click_mask_enabled": bool(getattr(self, "_click_mask_enabled", False)),
            "compare_mode": self._compare_mode,
            "mask_debug_mode": self._mask_debug_mode,
            "preview_image_is_None": self.preview_image is None,
            "mask_strength": (self._mask_adjustments.get(self._active_layer) or {}).get("strength"),
            "live_mask_face_index": self._mask_face_index,
        }
        task = MaskDiagnosticsTask(self.preview_array, self.segmenter)
        # Pinned on self for the same reason every other worker task here is -- an unpinned
        # local-only QRunnable isn't guaranteed to survive until its signal reaches the GUI
        # thread.
        self._mask_diag_task = task
        task.signals.finished.connect(self._on_mask_diagnostics_finished)
        task.signals.failed.connect(self._on_mask_diagnostics_failed)
        self._segmentation_pool.start(task)

    def _on_mask_diagnostics_finished(self, faces: list, report: dict):
        self._mask_diag_task = None
        self.mask_diagnostics_btn.setEnabled(True)
        self.mask_diagnostics_btn.setText("Diagnose Masks")
        self.statusBar().showMessage(f"Mask diagnostics ready -- {len(faces)} face(s) detected")
        live_face_index, live_masks = getattr(self, "_mask_diag_live_snapshot", (None, None))
        live_state = getattr(self, "_mask_diag_state", None)
        dialog = MaskDiagnosticsDialog(
            faces, report, self,
            live_face_index=live_face_index, live_masks=live_masks, live_state=live_state,
        )
        dialog.exec()

    def _on_mask_diagnostics_failed(self, message: str):
        self._mask_diag_task = None
        self.mask_diagnostics_btn.setEnabled(True)
        self.mask_diagnostics_btn.setText("Diagnose Masks")
        QMessageBox.critical(self, "Mask Diagnostics Error", self._friendly_error_message(RuntimeError(message)))

    def _apply_segmentation_result(self, result: dict, schedule_render: bool = True):
        """GUI-thread application of a SegmentationTask result -- mirrors the previous
        synchronous _run_segmentation body, just reading from `result` instead of locals."""
        self._detected_faces = result["detected_faces"]
        self.face_combo.blockSignals(True)
        self.face_combo.clear()
        if not self._detected_faces:
            self.face_combo.addItem("Auto")
            self.face_combo.setCurrentIndex(0)
            self._active_face_index = 0
        else:
            for idx, (x, y, w, h) in enumerate(self._detected_faces):
                self.face_combo.addItem(f"Face {idx + 1} ({w}x{h} @ {x},{y})")
            self._active_face_index = result["face_index"]
            self.face_combo.setCurrentIndex(self._active_face_index)
        self.face_combo.blockSignals(False)
        self._update_face_scope_indicator()

        self.preview_masks = self._copy_masks(result["preview_masks"])
        self.preview_guides = self._copy_guides(result["preview_guides"])
        self._mask_face_index = int(result.get("mask_face_index", result.get("face_index", self._active_face_index)))
        auto_preview_masks = self._copy_masks(self.preview_masks)
        auto_full_masks = self._copy_masks(result["full_masks"])
        auto_full_guides = self._copy_guides(result["full_guides"])
        self._auto_preview_masks = auto_preview_masks
        self._auto_full_masks = self._copy_masks(auto_full_masks)
        self.full_masks = self._copy_masks(auto_full_masks)
        self.full_guides = self._copy_guides(auto_full_guides)
        self._preview_analysis_signature = {
            "analysis_cache_version": int(ANALYSIS_CACHE_VERSION),
            "segmenter": segmenter_cache_signature(self.segmenter),
            "detected_faces": [list(face) for face in self._detected_faces],
            "face_index": int(result.get("face_index", self._active_face_index)),
            "mask_face_index": int(self._mask_face_index) if self._mask_face_index is not None else None,
        }
        restored = self._restore_face_profile(self._active_face_index)
        if not restored:
            self.preview_masks = self._copy_masks(auto_preview_masks)
            self.full_masks = self._copy_masks(auto_full_masks)
            self.preview_guides = self._copy_guides(result["preview_guides"])
            self.full_guides = self._copy_guides(auto_full_guides)
            # Make sure there's always at least one baseline document-history entry to undo
            # back to -- doesn't push a redundant extra entry if history already has one.
            if self._document_history_index < 0:
                self._push_document_history()
        self._perf_stats["detect_ms"] = result["detect_ms"]
        self._perf_stats["segment_ms"] = result["segment_ms"]
        if not self._detected_faces:
            status = (
                f"Loaded {os.path.basename(self.file_path)} -- no faces detected. Global, "
                "Subjects, and Background edits still apply; Face/Skin/Eyes/Lips/Hair/Person "
                "need a detected face, so try a tighter crop if this is a group or landscape shot."
            )
        else:
            status = (
                f"Loaded {os.path.basename(self.file_path)} | "
                f"seg={self.segmenter.backend_label} | "
                f"{self.segmenter.detector_backend_label} | "
                f"{getattr(self.segmenter, 'subject_backend_label', 'subjects=heuristic')} | "
                f"{getattr(self.segmenter, 'face_backend_label', 'face=heuristic')} | "
                f"{getattr(self.segmenter, 'face_part_backend_label', 'parts=semantic')} | "
                f"{getattr(self.segmenter, 'person_backend_label', 'person=watershed')} | "
                f"{getattr(self.segmenter, 'facial_hair_backend_label', 'f_hair=fallback')} | "
                f"faces={len(self._detected_faces)} | cache={result['cache_status']}"
            )
        self.statusBar().showMessage(status)
        self.info_label.setText(status)
        self._update_perf_label()
        self._refresh_mask_controls()
        if schedule_render:
            self._schedule_render()
        if self._document_history_index < 0:
            self._push_document_history()

    def _render_preview(self):
        if self.preview_array is None:
            return
        if self._render_in_flight:
            self._render_pending = True
            self._update_perf_label()
            return
        image, masks, geometry = self._current_preview_render_inputs()
        if image is None:
            return
        self._render_in_flight = True
        self._render_pending = False
        self._update_perf_label()
        self._render_job_counter += 1
        job_id = self._render_job_counter
        self._active_render_job_id = job_id
        task = PreviewRenderTask(
            job_id=job_id,
            image=image,
            params=self._copy_params(self._all_params()),
            masks=masks,
            geometry=geometry,
            layer_order=list(self._layer_order),
            layer_options={layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            color_settings=dict(self._color_settings),
            runtime_settings={
                **dict(self._runtime_settings),
                "fast_interactive_preview": bool(self._slider_drag_active > 0),
            },
            stage_cache=self._stage_pipeline_cache,
            inputs_token=self._stage_inputs_token(image),
            want_sharpen_mask=self._show_sharpen_mask_preview,
        )
        # Pin on self -- an unpinned local-only task here was reproduced to segfault/raise
        # "Signal source has been deleted" under GC pressure (caught while threading image
        # load elsewhere); this is the most frequently dispatched task in the app (every
        # slider tick), so it's the highest-value place to not have that race.
        self._render_task = task
        task.signals.finished.connect(self._on_preview_render_finished)
        task.signals.failed.connect(self._on_preview_render_failed)
        self._render_pool.start(task)

    def _on_preview_render_finished(self, job_id: int, result, elapsed_ms: float):
        sharpen_mask = self._render_task.debug_sink.get("sharpen_mask") if self._render_task is not None else None
        self._render_in_flight = False
        self._render_task = None
        if int(job_id) != self._active_render_job_id:
            if self._render_pending:
                self._update_perf_label()
                self._schedule_render()
            return
        self._last_completed_render_job_id = int(job_id)
        self._perf_stats["render_ms"] = float(elapsed_ms)
        self._sharpen_mask_preview = sharpen_mask
        self.preview_image = result
        if not self._render_pending:
            self._remember_preview_render_cache()
        self._update_perf_label()
        self._update_preview_label()
        self._update_histogram()
        self._sync_simple_controls()
        if self._render_pending:
            self._schedule_render()
        else:
            # Edits have settled -- refresh this image's filmstrip thumbnail (debounced).
            self._thumb_refresh_timer.start()

    def _on_preview_render_failed(self, job_id: int, message: str):
        self._render_in_flight = False
        self._render_task = None
        if int(job_id) != self._active_render_job_id:
            if self._render_pending:
                self._update_perf_label()
                self._schedule_render()
            return
        self.statusBar().showMessage(f"Preview render failed: {message}")
        self._update_perf_label()
        if self._render_pending:
            self._schedule_render()

    def _update_histogram(self):
        widget = getattr(self, "histogram_widget", None)
        curve = getattr(self, "tone_curve_widget", None)
        if widget is None:
            return
        if self.preview_image is None:
            widget.clear()
            if curve is not None:
                curve.set_histogram(None)
            return
        try:
            hist = compute_histogram(self.preview_image)
            widget.set_histogram(hist)
            if curve is not None:
                curve.set_histogram(hist)
        except Exception:
            widget.clear()
            if curve is not None:
                curve.set_histogram(None)

    def _update_perf_label(self):
        def fmt(value):
            return "--" if value is None else f"{value:.1f}"

        face_params = self._all_params().get("face", {})
        expr_mode = expression_warp_mode(face_params, self.preview_guides)
        refine_value = float(face_params.get("refine", 0.0))
        if refine_value <= 0.0:
            refine_mode = "off"
        else:
            refiner = get_face_refiner()
            ready = refiner.available or refiner._ensure_session()
            refine_mode = refiner.backend_label if ready else "unavailable"
        if self._render_in_flight:
            preview_state = "rendering+queued" if self._render_pending else "rendering"
        elif self._render_pending:
            preview_state = "queued"
        elif self.preview_image is None:
            preview_state = "idle"
        else:
            preview_state = "ready"
        self.perf_label.setText(
            f"Perf: detect={fmt(self._perf_stats.get('detect_ms'))} ms | "
            f"segment={fmt(self._perf_stats.get('segment_ms'))} ms | "
            f"render={fmt(self._perf_stats.get('render_ms'))} ms | "
            f"preview={preview_state} | "
            f"expr={expr_mode} | "
            f"refine={refine_mode}"
        )

    def _display_framing(self):
        """Framing to apply to the preview display for the current edit mode.

        Mask editing operates in source space, so it sees no framing; crop-edit
        mode shows the full straightened frame with a crop-box overlay; otherwise
        the crop is baked into the preview.
        """
        if self._mask_edit_enabled or self._wb_pick_enabled or self._click_mask_enabled or self._denoise_point_enabled:
            return default_framing()
        framing = normalize_framing(self._framing)
        if self._crop_edit_enabled:
            return {**framing, "crop": list(framing_ops.DEFAULT_CROP)}
        return framing

    def _update_preview_label(self):
        if self.preview_image is None:
            return
        edited = self.preview_image.convert("RGB")
        edited = self._apply_mask_overlay(edited)
        edited = self._apply_sharpen_mask_preview_overlay(edited)
        edited = self._apply_expression_guide_overlay(edited)
        display_framing = self._display_framing()
        edited = apply_framing(edited, display_framing)
        if self.preview_array is None:
            image = edited
        else:
            original = Image.fromarray(to_uint8(self.preview_array)).convert("RGB")
            original = apply_framing(original, display_framing)
            image = self._compose_compare_image(original, edited)
        crop_overlay = self._framing["crop"] if self._crop_edit_enabled else None
        self.image_label.set_crop_overlay(crop_overlay)
        self.image_label.set_hires_enabled(self._hires_enabled_now())
        data = image.tobytes("raw", "RGB")
        qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format_RGB888).copy()
        self.image_label.set_preview_pixmap(QPixmap.fromImage(qimage))
        # Re-derive the face highlight against whatever framing/mode is now displayed --
        # single point of truth, so it can't go stale no matter which code path (crop drag,
        # straighten slider, flip, mode toggle) got us here.
        self._update_face_scope_indicator()

    def _update_mask_debug_label(self):
        if (
            not hasattr(self, "mask_debug_label")
            or not self._show_mask
            or self.preview_masks is None
            or self._active_layer not in MASK_ORDER
            or self._active_layer not in self.preview_masks
        ):
            if hasattr(self, "mask_debug_label"):
                self.mask_debug_label.setText("Mask: --")
            return

        mask = self._active_preview_mask()
        if mask is None or mask.size == 0:
            self.mask_debug_label.setText("Mask: --")
            return

        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        mean_v = float(mask.mean())
        max_v = float(mask.max())
        coverage_lo = float((mask > 0.10).mean()) * 100.0
        coverage_hi = float((mask > 0.50).mean()) * 100.0
        self.mask_debug_label.setText(
            f"Mask {self._active_layer}: mode={self._mask_debug_mode} | "
            f"mean={mean_v:.2f} | max={max_v:.2f} | "
            f">0.10={coverage_lo:.1f}% | >0.50={coverage_hi:.1f}%"
        )

    def _apply_expression_guide_overlay(self, edited: Image.Image) -> Image.Image:
        if not self._show_expression_guides or not self.preview_guides:
            return edited

        guides = self.preview_guides[0] if isinstance(self.preview_guides, list) and self.preview_guides else self.preview_guides
        if not isinstance(guides, dict):
            return edited

        out = edited.copy()
        draw = ImageDraw.Draw(out)

        def point(key):
            value = guides.get(key)
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                return None
            return float(value[0]), float(value[1])

        def draw_point(pt, color, radius=4):
            if pt is None:
                return
            x, y = pt
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(8, 8, 10))

        def draw_link(a, b, color, width=2):
            if a is None or b is None:
                return
            draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)

        mouth_left = point("mouth_left")
        mouth_right = point("mouth_right")
        mouth_upper = point("mouth_upper")
        mouth_lower = point("mouth_lower")
        left_eye_upper = point("left_eye_upper")
        left_eye_lower = point("left_eye_lower")
        right_eye_upper = point("right_eye_upper")
        right_eye_lower = point("right_eye_lower")
        left_brow = point("left_brow")
        right_brow = point("right_brow")

        draw_link(mouth_left, mouth_right, (214, 91, 91), width=2)
        draw_link(mouth_upper, mouth_lower, (214, 91, 91), width=2)
        draw_link(left_eye_upper, left_eye_lower, (91, 155, 213), width=2)
        draw_link(right_eye_upper, right_eye_lower, (91, 155, 213), width=2)
        draw_link(left_brow, left_eye_upper, (212, 168, 83), width=2)
        draw_link(right_brow, right_eye_upper, (212, 168, 83), width=2)

        for pt in (mouth_left, mouth_right, mouth_upper, mouth_lower):
            draw_point(pt, (214, 91, 91))
        for pt in (left_eye_upper, left_eye_lower, right_eye_upper, right_eye_lower):
            draw_point(pt, (91, 155, 213))
        for pt in (left_brow, right_brow):
            draw_point(pt, (212, 168, 83))

        return out

    def _compose_compare_image(self, original: Image.Image, edited: Image.Image) -> Image.Image:
        mode = self._compare_mode
        if mode == "before":
            return original
        if mode == "split":
            if original.size != edited.size:
                original = original.resize(edited.size, Image.LANCZOS)
            split_px = int(max(0, min(edited.width, round(self._split_position * edited.width))))
            composed = original.copy()
            if split_px < edited.width:
                composed.paste(edited.crop((split_px, 0, edited.width, edited.height)), (split_px, 0))
            draw = ImageDraw.Draw(composed)
            draw.line((split_px, 0, split_px, edited.height), fill=(212, 168, 83), width=3)
            return composed
        if mode == "side_by_side":
            if original.size != edited.size:
                original = original.resize(edited.size, Image.LANCZOS)
            gap = 12
            composed = Image.new("RGB", (original.width * 2 + gap, original.height), (12, 12, 14))
            composed.paste(original, (0, 0))
            composed.paste(edited, (original.width + gap, 0))
            draw = ImageDraw.Draw(composed)
            draw.line((original.width + gap // 2, 0, original.width + gap // 2, original.height), fill=(37, 37, 40), width=2)
            return composed
        return edited

    def _copy_masks(self, masks):
        return _copy_masks_data(masks)

    def _ref_masks(self, masks):
        # Shallow snapshot: a new dict that shares the mask array references. This
        # is safe because mask arrays are only ever *replaced* in the live state
        # (never mutated in place), so a stored reference keeps its historical
        # value. Any path that needs independent live arrays (_restore_face_profile)
        # deep-copies via _copy_masks. Avoids copying full-resolution mask pixels on
        # the interaction hot path (document-history capture on every sliderPressed).
        if masks is None:
            return None
        return dict(masks)

    def _copy_params(self, params):
        return {layer: dict(values) for layer, values in (params or {}).items()}

    def _copy_mask_adjustments(self, adjustments):
        return {layer: dict(values) for layer, values in normalize_mask_adjustments(adjustments).items()}

    def _adjusted_masks(self, masks):
        return apply_mask_adjustments(
            masks,
            self._mask_adjustments,
            acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
        )

    def _active_preview_mask(self):
        if self.preview_masks is None or self._active_layer not in self.preview_masks:
            return None
        adjusted = self._adjusted_masks({self._active_layer: self.preview_masks[self._active_layer]})
        if not adjusted:
            return None
        return adjusted.get(self._active_layer)

    def _stage_inputs_token(self, image):
        """Identity of the static (non-slider) inputs to the staged pipeline: which source
        image, mask revision/analysis, mask adjustments, and preview resolution. Slider params,
        layer options, color and runtime settings are intentionally excluded -- those are keyed
        per-stage inside process_all_layers. Any change here busts the whole stage chain, which
        is correct because these are exactly the inputs every stage depends on."""
        shape = tuple(int(v) for v in image.shape[:2]) if image is not None else None
        return {
            "source": self._source_file_signature(self.file_path),
            "revision": int(self._preview_state_revision),
            "analysis": self._preview_analysis_signature,
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "shape": list(shape) if shape else None,
            "deep_denoise": bool(self._deep_denoise_enabled),
        }

    def _current_preview_render_inputs(self):
        if self.preview_array is None:
            return None, None, None
        # When Deep Denoise is on, the pipeline base is the denoised proxy (live), not the noisy
        # source -- masks/guides still come from the noisy analysis, which is correct.
        denoise_on = self._deep_denoise_enabled and self._deep_denoise_preview is not None
        if self._slider_drag_active <= 0 or self._interactive_preview_array is None:
            base = self._deep_denoise_preview if denoise_on else self.preview_array
            return (
                base.copy(),
                self._adjusted_masks(self.preview_masks),
                self._copy_guides(self.preview_guides),
            )
        ratio = float(self._interactive_preview_ratio)
        interactive_base = (
            self._deep_denoise_interactive
            if denoise_on and self._deep_denoise_interactive is not None
            else self._interactive_preview_array
        )
        masks = self._scale_masks(self.preview_masks, interactive_base.shape[:2])
        masks = self._adjusted_masks(masks)
        guides = _scale_expression_guides(self.preview_guides, ratio, ratio)
        return interactive_base.copy(), masks, self._copy_guides(guides)

    def _hires_enabled_now(self) -> bool:
        """True full-res pixel zoom is only meaningful for the plain edited image -- every
        mode below already shows something other than that (source-space edit tools, the
        full straightened frame for crop-edit, a mask/guide overlay, or a before/after split),
        so the proxy is correct/expected there and hi-res tiles would either be wrong or
        pointless."""
        if self._mask_edit_enabled or self._wb_pick_enabled or self._click_mask_enabled or self._denoise_point_enabled:
            return False
        if self._crop_edit_enabled or self._show_mask or self._show_expression_guides:
            return False
        if self._show_sharpen_mask_preview:
            return False
        if self._compare_mode != "off":
            return False
        return True

    def _on_hires_requested(self, rect_norm):
        """Render one screen-resolution tile of the true full-resolution source for the
        normalized display-space rect `rect_norm`, requested by the canvas once it's zoomed
        in past where the editing proxy has real detail."""
        if self.full_array is None or self.full_masks is None:
            return
        if not self._hires_enabled_now():
            return
        dx0, dy0, dx1, dy1 = rect_norm
        full_h, full_w = self.full_array.shape[:2]
        display_framing = self._display_framing()

        corners = [(dx0, dy0), (dx1, dy0), (dx0, dy1), (dx1, dy1)]
        src_pts = [
            framing_ops.display_point_to_source(display_framing, full_w, full_h, dx, dy)
            for dx, dy in corners
        ]
        margin = 0.02
        sx0 = max(0.0, min(p[0] for p in src_pts) - margin)
        sy0 = max(0.0, min(p[1] for p in src_pts) - margin)
        sx1 = min(1.0, max(p[0] for p in src_pts) + margin)
        sy1 = min(1.0, max(p[1] for p in src_pts) + margin)

        ox0 = int(math.floor(sx0 * full_w))
        oy0 = int(math.floor(sy0 * full_h))
        ox1 = int(math.ceil(sx1 * full_w))
        oy1 = int(math.ceil(sy1 * full_h))
        ox1 = max(ox1, ox0 + 1)
        oy1 = max(oy1, oy0 + 1)

        max_edge = self._hires_max_edge
        if (ox1 - ox0) > max_edge:
            cx = (ox0 + ox1) // 2
            ox0, ox1 = cx - max_edge // 2, cx + max_edge // 2
        if (oy1 - oy0) > max_edge:
            cy = (oy0 + oy1) // 2
            oy0, oy1 = cy - max_edge // 2, cy + max_edge // 2
        ox0, oy0 = max(0, ox0), max(0, oy0)
        ox1, oy1 = min(full_w, ox1), min(full_h, oy1)
        if ox1 <= ox0 or oy1 <= oy0:
            return

        # Non-blocking: only use the deep-denoise full-res result if it's already computed --
        # must NOT call _export_base_full_array() here, since that can trigger a blocking,
        # multi-minute full-res NAFNet denoise synchronously if deep denoise is enabled but
        # not yet cached. A hi-res viewport request should never stall the GUI thread.
        denoise_on = self._deep_denoise_enabled and self._deep_denoise_full is not None
        base_full = self._deep_denoise_full if denoise_on else self.full_array
        crop = base_full[oy0:oy1, ox0:ox1].copy()
        masks_full = self._adjusted_masks(self.full_masks)
        crop_masks = {
            layer: np.asarray(mask)[oy0:oy1, ox0:ox1].copy()
            for layer, mask in (masks_full or {}).items()
            if mask is not None
        }
        crop_guides = _offset_expression_guides(self._copy_guides(self.full_guides), -ox0, -oy0)

        norm_f = normalize_framing(display_framing)
        origin_x, origin_y, crop_x1, crop_y1 = framing_ops.crop_box_px(norm_f, full_w, full_h)
        crop_w_px = crop_x1 - origin_x
        crop_h_px = crop_y1 - origin_y
        target_rect_px = (
            origin_x + round(dx0 * crop_w_px),
            origin_y + round(dy0 * crop_h_px),
            origin_x + round(dx1 * crop_w_px),
            origin_y + round(dy1 * crop_h_px),
        )

        self._hires_job_counter += 1
        job_id = self._hires_job_counter
        self._hires_active_job_id = job_id
        task = HiresTileTask(
            job_id=job_id,
            crop=crop,
            masks=crop_masks,
            geometry=crop_guides,
            params=self._copy_params(self._all_params()),
            layer_order=list(self._layer_order),
            layer_options={layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            color_settings=dict(self._color_settings),
            runtime_settings=dict(self._runtime_settings),
            crop_origin=(ox0, oy0),
            full_shape=(full_h, full_w),
            framing=display_framing,
            target_rect_px=target_rect_px,
            rect_norm=(dx0, dy0, dx1, dy1),
        )
        self._hires_task = task
        task.signals.finished.connect(self._on_hires_tile_finished)
        task.signals.failed.connect(self._on_hires_tile_failed)
        self._hires_pool.start(task)

    def _on_hires_tile_finished(self, job_id, tile, rect_norm):
        self._hires_task = None
        if int(job_id) != self._hires_active_job_id or tile is None:
            return
        pil = tile.convert("RGB")
        w, h = pil.size
        data = pil.tobytes("raw", "RGB")
        qimage = QImage(data, w, h, w * 3, QImage.Format_RGB888).copy()
        self.image_label.set_hires_tile(QPixmap.fromImage(qimage), rect_norm)

    def _on_hires_tile_failed(self, job_id, message: str):
        self._hires_task = None
        # Silent: this is a background quality upgrade triggered by panning/zooming, not a
        # user-initiated action -- the proxy stays on screen either way, so there's nothing
        # actionable to surface in the status bar.

    def _begin_mask_stroke(self):
        """Mask strokes now go through the same document-history timeline as everything
        else (sliders, presets, crop, ...) -- previously they had their own separate undo
        stack with its own buttons, so Ctrl+Z and the Masks panel's Undo button could disagree
        about what the "last action" even was."""
        if not self._mask_edit_enabled or self.preview_masks is None or self._active_layer not in self.preview_masks:
            return
        self._begin_document_change()

    def _commit_mask_stroke(self):
        if not self._mask_edit_enabled:
            return
        self._push_document_history()

    def _apply_mask_overlay(self, edited: Image.Image) -> Image.Image:
        if not self._show_mask or self.preview_masks is None or self._active_layer not in self.preview_masks:
            return edited
        mask = self._active_preview_mask()
        if mask is None:
            return edited
        mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        base = np.asarray(edited, dtype=np.float32) / 255.0
        if self._mask_debug_mode == "isolated":
            isolated = np.repeat(mask[:, :, None], 3, axis=2)
            return Image.fromarray((isolated * 255).astype(np.uint8))

        if self._mask_debug_mode == "heatmap":
            heat = np.zeros_like(base)
            heat[:, :, 0] = np.clip((mask - 0.35) / 0.65, 0.0, 1.0)
            heat[:, :, 1] = np.clip(1.0 - np.abs(mask - 0.5) / 0.5, 0.0, 1.0) * 0.85
            heat[:, :, 2] = np.clip((0.55 - mask) / 0.55, 0.0, 1.0)
            alpha = np.clip(0.15 + mask * 0.55, 0.0, 0.70)[..., None]
            blended = base * (1.0 - alpha) + heat * alpha
            return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8))

        color_hex = LAYER_COLORS.get(self._active_layer, "#ffffff")
        cr, cg, cb = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        alpha = mask[..., None] * 0.45
        color = np.zeros_like(base)
        color[:, :, 0] = cr / 255.0
        color[:, :, 1] = cg / 255.0
        color[:, :, 2] = cb / 255.0
        blended = base * (1.0 - alpha) + color * alpha
        return Image.fromarray((np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8))

    def _apply_sharpen_mask_preview_overlay(self, edited: Image.Image) -> Image.Image:
        """Grayscale preview of the Sharpen Masking edge mask: white = sharpening applied,
        black = protected/flat. Replaces the canvas wholesale (like the layer Mask View's
        isolated mode) rather than tinting, since the point is to read the threshold cleanly."""
        if not self._show_sharpen_mask_preview or self._sharpen_mask_preview is None:
            return edited
        mask = np.clip(self._sharpen_mask_preview.astype(np.float32), 0.0, 1.0)
        isolated = np.repeat(mask[:, :, None], 3, axis=2)
        return Image.fromarray((isolated * 255).astype(np.uint8))

    def _paint_active_mask_at(self, rel_x: float, rel_y: float):
        if not self._mask_edit_enabled or self.preview_masks is None or self._active_layer not in self.preview_masks:
            return
        mask = self.preview_masks[self._active_layer]
        h, w = mask.shape[:2]
        ix = int(np.clip(rel_x * w, 0, w - 1))
        iy = int(np.clip(rel_y * h, 0, h - 1))
        radius = max(1, int(round(self._mask_brush_size)))
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2).astype(np.float32)
        hard_ratio = max(0.0, min(1.0, self._mask_brush_hardness / 100.0))
        inner_radius = radius * hard_ratio
        feather_span = max(1.0, radius - inner_radius)
        brush = np.clip((radius - dist) / feather_span, 0.0, 1.0)
        if hard_ratio >= 0.999:
            brush = (dist <= radius).astype(np.float32)
        else:
            brush = np.where(dist <= inner_radius, 1.0, brush).astype(np.float32)
        if self._mask_paint_mode == "erase":
            mask = mask * (1.0 - brush)
        else:
            mask = np.maximum(mask, brush)
        self.preview_masks[self._active_layer] = np.clip(mask.astype(np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(self._active_layer)
        self._update_document_history_actions()
        self._update_mask_debug_label()
        self._update_mask_review_panel()
        self._schedule_render()

    def _sync_full_mask_from_preview(self, layer: str):
        if self.preview_masks is None or self.full_array is None or layer not in self.preview_masks:
            return
        self._mark_preview_pixels_changed()
        mask = self.preview_masks[layer]
        pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize((self.full_array.shape[1], self.full_array.shape[0]), Image.BILINEAR)
        if self.full_masks is None:
            self.full_masks = {}
        self.full_masks[layer] = np.asarray(pil, dtype=np.float32) / 255.0

    def _on_object_clicked(self, nx: float, ny: float):
        """AI Select: one click -> one SAM-prompted mask for the active layer. Runs off the GUI
        thread (the first click on a fresh image pays the SAM embedding cost, up to a couple
        seconds on CPU); every click after that on the same image reuses the cached embedding."""
        if self.preview_array is None or self._active_layer not in MASK_ORDER or self.preview_masks is None:
            return
        instance = getattr(self.segmenter, "_instance", None)
        if instance is None or not getattr(instance, "available", False):
            self.statusBar().showMessage(
                "AI Select needs the SAM model -- see System Check for setup, or use the brush instead."
            )
            return
        h, w = self.preview_array.shape[:2]
        point_xy = (float(np.clip(nx, 0.0, 1.0)) * (w - 1), float(np.clip(ny, 0.0, 1.0)) * (h - 1))

        self.statusBar().showMessage("AI Select: computing mask...")
        self._mask_click_job_id += 1
        job_id = self._mask_click_job_id
        task = MaskClickTask(job_id, self.preview_array, point_xy, self.segmenter)
        # Pinned on self for the same reason every other worker task here is -- an unpinned
        # local-only QRunnable isn't guaranteed to survive until its signal reaches the GUI thread.
        self._mask_click_task = task
        task.signals.finished.connect(self._on_mask_click_finished)
        task.signals.failed.connect(self._on_mask_click_failed)
        self._segmentation_pool.start(task)

    def _on_mask_click_finished(self, job_id: int, mask):
        self._mask_click_task = None
        if job_id != self._mask_click_job_id:
            return  # superseded by a newer click
        if mask is None or self.preview_masks is None or self._active_layer not in self.preview_masks:
            self.statusBar().showMessage("AI Select: couldn't produce a mask there -- try a different point.")
            return
        layer = self._active_layer
        self._begin_document_change()
        self.preview_masks[layer] = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(layer)
        self._update_mask_debug_label()
        self._update_mask_review_panel()
        self._schedule_render()
        self._push_document_history()
        self.statusBar().showMessage(f"AI Select: replaced the {layer} mask (Ctrl+Z to undo).")

    def _on_mask_click_failed(self, job_id: int, message: str):
        self._mask_click_task = None
        if job_id != self._mask_click_job_id:
            return
        self.statusBar().showMessage(f"AI Select failed: {self._friendly_error_message(RuntimeError(message))}")

    def _reset_active_mask(self):
        if self._active_layer not in MASK_ORDER or self._auto_preview_masks is None:
            return
        self._begin_document_change()
        if self._active_layer in self._auto_preview_masks:
            self.preview_masks[self._active_layer] = self._auto_preview_masks[self._active_layer].copy()
            self._mark_preview_pixels_changed()
        if self._auto_full_masks is not None and self._active_layer in self._auto_full_masks:
            if self.full_masks is None:
                self.full_masks = {}
            self.full_masks[self._active_layer] = self._auto_full_masks[self._active_layer].copy()
        self._update_mask_debug_label()
        self._update_mask_review_panel()
        self._schedule_render()
        self._push_document_history()

    def _recalculate_active_mask(self):
        """Force a genuinely fresh recomputation of the active layer's mask, bypassing any
        cached result (unlike Reset Mask, which only reverts to whatever was cached). Only the
        active layer is replaced -- other layers, including any manual edits there, are left
        untouched, since the user explicitly asked for single-layer scope even though the
        underlying segmentation pass computes every layer together."""
        if self.preview_array is None or self.full_array is None or self._active_layer not in MASK_ORDER:
            return
        if self._mask_recalc_task is not None:
            return  # one in flight at a time
        layer = self._active_layer
        self.statusBar().showMessage(f"Recalculating {layer} mask from scratch -- this may take a while...")
        self._drop_segmentation_result_cache(self.file_path)
        self._mask_recalc_job_id += 1
        job_id = self._mask_recalc_job_id
        task = SegmentationTask(
            job_id=job_id,
            file_path=self.file_path,
            preview_array=self.preview_array,
            full_shape=self.full_array.shape[:2],
            preview_scale=self.preview_scale,
            active_face_index=self._active_face_index,
            segmenter=self.segmenter,
            force_recompute=True,
        )
        task.layer = layer  # stashed for the finished/failed handlers -- not read by SegmentationTask itself
        # Pinned on self for the same reason every other worker task here is -- an unpinned
        # local-only QRunnable isn't guaranteed to survive until its signal reaches the GUI thread.
        self._mask_recalc_task = task
        task.signals.finished.connect(self._on_mask_recalculate_finished)
        task.signals.failed.connect(self._on_mask_recalculate_failed)
        self._segmentation_pool.start(task)
        self._refresh_mask_controls()

    def _on_mask_recalculate_finished(self, job_id: int, result: dict):
        task = self._mask_recalc_task
        self._mask_recalc_task = None
        layer = getattr(task, "layer", None)
        if job_id != self._mask_recalc_job_id or layer is None:
            self._refresh_mask_controls()
            return  # superseded by a newer recalculate request
        if self.preview_masks is None or result.get("mask_face_index") != self._mask_face_index:
            # The active face changed while this was computing -- applying it now would splice a
            # different person's mask into the current document. Silently drop it; the user can
            # just click Recalculate again on the face they actually want.
            self.statusBar().showMessage(f"Recalculate {layer}: discarded (active face changed while computing).")
            self._refresh_mask_controls()
            return
        fresh_preview = result.get("preview_masks", {}).get(layer)
        if fresh_preview is None:
            self.statusBar().showMessage(f"Recalculate {layer}: the model didn't return this layer.")
            self._refresh_mask_controls()
            return

        self._begin_document_change()
        self.preview_masks[layer] = np.clip(np.asarray(fresh_preview, dtype=np.float32), 0.0, 1.0)
        self._mark_preview_pixels_changed()
        fresh_full = result.get("full_masks", {}).get(layer)
        if fresh_full is not None and self.full_masks is not None:
            self.full_masks[layer] = np.clip(np.asarray(fresh_full, dtype=np.float32), 0.0, 1.0)
        # Update the auto-baseline too, so a later Reset Mask reverts to *this* fresh value
        # instead of the stale one it would otherwise fall back to.
        if self._auto_preview_masks is not None:
            self._auto_preview_masks[layer] = self.preview_masks[layer].copy()
        if fresh_full is not None and self._auto_full_masks is not None:
            self._auto_full_masks[layer] = self.full_masks[layer].copy()
        self._update_mask_debug_label()
        self._schedule_render()
        self._push_document_history()
        self.statusBar().showMessage(f"Recalculated {layer} mask from scratch (Ctrl+Z to undo).")
        self._refresh_mask_controls()

    def _on_mask_recalculate_failed(self, job_id: int, message: str):
        self._mask_recalc_task = None
        if job_id != self._mask_recalc_job_id:
            return
        self.statusBar().showMessage(f"Recalculate failed: {self._friendly_error_message(RuntimeError(message))}")
        self._refresh_mask_controls()

    def _feather_active_mask(self):
        if self._active_layer not in MASK_ORDER or self.preview_masks is None:
            return
        if self._active_layer not in self.preview_masks:
            return
        self._begin_document_change()
        sigma = max(1.0, float(self._mask_brush_size) / 6.0)
        feathered = smooth_mask(
            self.preview_masks[self._active_layer],
            sigma=sigma,
            acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
        )
        self.preview_masks[self._active_layer] = np.clip(feathered.astype(np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(self._active_layer)
        self._update_mask_debug_label()
        self._update_mask_review_panel()
        self._schedule_render()
        self._push_document_history()

    def export_image(self):
        if self.full_array is None:
            if self._hydrate_active_image_if_needed("export", on_complete=self.export_image):
                return
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        src = Path(self.file_path) if self.file_path else None
        default_dir = self._last_export_dest_dir or (str(src.parent) if src else os.path.expanduser("~"))
        stem = src.stem if src else "portrait"
        h, w = self.full_array.shape[:2]
        dialog = ExportDialog(
            self,
            output_dir=default_dir,
            stem=stem,
            output_format=self._last_export_format,
            quality=self._last_export_quality,
            has_metadata=bool(self._source_metadata),
            source_w=w,
            source_h=h,
            output_sharpening=self._last_export_output_sharpening,
            deep_denoise=self._last_export_deep_denoise,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        opts = dialog.options()
        if not opts["dest_dir"]:
            QMessageBox.warning(self, "Export", "Choose a destination folder.")
            return
        out_path = opts["out_path"]
        if os.path.exists(out_path):
            confirm = QMessageBox.question(
                self, "Export", f"{os.path.basename(out_path)} already exists. Overwrite?"
            )
            if confirm != QMessageBox.Yes:
                return
        self._remember_export_options(opts)
        self._dispatch_export_as_job(src, out_path, opts)

    def quick_export_image(self):
        """Zero-dialog export: writes immediately using whatever format/quality/destination/
        resize were used last (this image's own folder + jpeg/92 if nothing's been exported
        yet this session), instead of requiring a trip through the Export dialog every time.
        Still confirms before overwriting -- speed shouldn't come at the cost of silently
        clobbering an existing file."""
        if self.full_array is None:
            if self._hydrate_active_image_if_needed("quick export", on_complete=self.quick_export_image):
                return
            self.statusBar().showMessage("Open an image first")
            return
        src = Path(self.file_path) if self.file_path else None
        stem = src.stem if src else "portrait"
        dest_dir = self._last_export_dest_dir or (str(src.parent) if src else os.path.expanduser("~"))
        ext = {"jpeg": ".jpg", "png": ".png", "tiff": ".tiff"}.get(self._last_export_format, ".jpg")
        opts = {
            "format": self._last_export_format,
            "ext": ext,
            "quality": self._last_export_quality,
            "resize_long_edge": self._last_export_resize_long_edge,
            "keep_metadata": self._last_export_keep_metadata and bool(self._source_metadata),
            "output_sharpening": self._last_export_output_sharpening,
            "deep_denoise": self._last_export_deep_denoise,
            "dest_dir": dest_dir,
        }
        out_path = os.path.join(dest_dir, f"{stem}_export{ext}")
        if not os.path.isdir(dest_dir):
            self.statusBar().showMessage(f"Quick Export: destination folder no longer exists -- {dest_dir}")
            return
        if os.path.exists(out_path):
            confirm = QMessageBox.question(
                self, "Quick Export", f"{os.path.basename(out_path)} already exists. Overwrite?"
            )
            if confirm != QMessageBox.Yes:
                return
        self._dispatch_quick_export_job(src, dest_dir, opts)

    def _dispatch_quick_export_job(self, src: Path | None, dest_dir: str, opts: dict):
        if src is None or not self.file_path:
            self.statusBar().showMessage("Quick Export: no source image")
            return
        source_path = os.path.abspath(str(src))
        ext = opts.get("ext") or {"jpeg": ".jpg", "png": ".png", "tiff": ".tiff"}.get(opts.get("format"), ".jpg")
        output_format = opts.get("format", self._last_export_format)
        suffix = "_export"
        override = self._build_settings_payload()
        job_path = self._write_batch_job_file(
            {
                "mode": "collection_export",
                "quick_export": True,
                "output_dir": os.path.abspath(dest_dir),
                "suffix": suffix,
                "output_format": output_format,
                "output_quality": int(opts.get("quality", self._last_export_quality)),
                "resize_long_edge": int(opts.get("resize_long_edge") or 0),
                "keep_metadata": bool(opts.get("keep_metadata", False)),
                "skip_completed": False,
                "disable_segmentation": False,
                "output_sharpening": opts.get("output_sharpening", "standard"),
                "deep_denoise": bool(opts.get("deep_denoise", False)),
                "runtime_settings": dict(self._runtime_settings),
                "source_paths": [source_path],
                "image_overrides": {source_path: override},
            },
            dest_dir,
        )
        job_id = Path(job_path).stem
        self._launch_background_batch_job(job_path, dest_dir)
        self._set_filmstrip_export_status([source_path], "queued")
        self._start_export_monitor(
            job_id=job_id,
            output_dir=os.path.abspath(dest_dir),
            source_count=1,
            label=f"Quick Export {src.name}",
            source_paths=[source_path],
        )
        self.statusBar().showMessage(
            f"Quick Export running in background -> {os.path.join(dest_dir, src.stem + suffix + ext)}"
        )

    def _dispatch_export_as_job(self, src: Path | None, out_path: str, opts: dict):
        if src is None or not self.file_path:
            self.statusBar().showMessage("Export As: no source image")
            return
        source_path = os.path.abspath(str(src))
        dest_dir = os.path.abspath(opts.get("dest_dir") or os.path.dirname(out_path))
        override = self._build_settings_payload()
        job_path = self._write_batch_job_file(
            {
                "mode": "export_as",
                "output_dir": dest_dir,
                "suffix": "",
                "output_format": opts.get("format", self._last_export_format),
                "output_quality": int(opts.get("quality", self._last_export_quality)),
                "resize_long_edge": int(opts.get("resize_long_edge") or 0),
                "keep_metadata": bool(opts.get("keep_metadata", False)),
                "skip_completed": False,
                "disable_segmentation": False,
                "output_sharpening": opts.get("output_sharpening", "standard"),
                "deep_denoise": bool(opts.get("deep_denoise", False)),
                "runtime_settings": dict(self._runtime_settings),
                "source_paths": [source_path],
                "image_overrides": {source_path: override},
                "explicit_output_paths": {source_path: os.path.abspath(out_path)},
            },
            dest_dir,
        )
        job_id = Path(job_path).stem
        self._launch_background_batch_job(job_path, dest_dir)
        self._set_filmstrip_export_status([source_path], "queued")
        self._start_export_monitor(
            job_id=job_id,
            output_dir=dest_dir,
            source_count=1,
            label=f"Export As {os.path.basename(out_path)}",
            source_paths=[source_path],
        )
        self.statusBar().showMessage(f"Export As running in background -> {out_path}")

    def _remember_export_options(self, opts: dict):
        self._last_export_format = opts["format"]
        self._last_export_quality = opts["quality"]
        self._last_export_resize_long_edge = opts["resize_long_edge"]
        self._last_export_keep_metadata = opts["keep_metadata"]
        self._last_export_output_sharpening = opts.get("output_sharpening", "standard")
        self._last_export_deep_denoise = bool(opts.get("deep_denoise", False))
        self._last_export_dest_dir = opts["dest_dir"]

    def _export_base_full_array(self, force_deep_denoise: bool = False):
        """Full-resolution source for export. When Deep Denoise is on, this is the full-res
        NAFNet result -- preferably the background pass cached during editing; if that hasn't
        finished yet, it's computed inline here (export is a deliberate commit, so the wait is
        acceptable and expected -- 'full-resolution applied at export')."""
        deep_denoise_on = bool(force_deep_denoise or self._deep_denoise_enabled)
        if not deep_denoise_on or self.full_array is None:
            return self.full_array
        if self._deep_denoise_full is not None:
            return self._deep_denoise_full
        result = self._run_export_deep_denoise()
        self._deep_denoise_full = result  # cache for subsequent exports of the same image
        return result

    def _run_export_deep_denoise(self):
        """Return the full-resolution Deep Denoise result for export.

        A checked export denoise option must not silently save the original pixels: CoreML can
        fail at inference time on this model, so retry on CPU and then fail the export visibly if
        denoise still cannot run.
        """
        deep = getattr(self, "_deep_denoiser", None)
        failures = []
        if deep is not None and getattr(deep, "available", False):
            self.statusBar().showMessage("Export: applying full-resolution Deep Denoise (this can take a few minutes)...")
            QApplication.processEvents()
            result = deep.denoise(self.full_array)
            if result is not None:
                return result
            failures.append(f"{getattr(deep, 'execution_provider', 'default')} inference failed")
        else:
            reason = getattr(deep, "reason_unavailable", "") if deep is not None else "denoiser was not initialized"
            failures.append(reason or "denoiser unavailable")

        cpu_denoiser = DeepDenoiser(use_coreml=False)
        if not getattr(cpu_denoiser, "available", False):
            reason = getattr(cpu_denoiser, "reason_unavailable", "") or "CPU denoiser unavailable"
            failures.append(reason)
            raise RuntimeError("Deep Denoise was requested for export but is unavailable: " + "; ".join(failures))

        self.statusBar().showMessage("Export: Deep Denoise accelerator failed; retrying on CPU...")
        QApplication.processEvents()
        result = cpu_denoiser.denoise(self.full_array)
        if result is None:
            failures.append("CPU inference failed")
            raise RuntimeError("Deep Denoise was requested for export but failed: " + "; ".join(failures))

        self._deep_denoiser = cpu_denoiser
        return result

    def _render_and_save_export(self, out_path: str, opts: dict):
        result = process_all_layers(
            self._export_base_full_array(force_deep_denoise=bool(opts.get("deep_denoise", False))),
            self._all_params(),
            self._adjusted_masks(self.full_masks),
            geometry=self.full_guides,
            layer_order=self._layer_order,
            layer_options=self._layer_options,
            color_settings=self._color_settings,
            runtime_settings=self._runtime_settings,
        )
        result = apply_framing(result, self._framing)
        if opts["resize_long_edge"]:
            result = self._resize_long_edge(result, opts["resize_long_edge"])
        # Calibrated to the export's *actual* output size -- must run after resize, since a
        # downsized export needs a different sharpening radius than the working-resolution
        # render.
        result = apply_output_sharpening(result, opts.get("output_sharpening", "standard"))
        save_kwargs = self._export_metadata_kwargs(opts) if opts["keep_metadata"] else {}
        if opts["format"] == "jpeg":
            result.save(out_path, "JPEG", quality=opts["quality"], subsampling=0, **save_kwargs)
        elif opts["format"] == "png":
            result.save(out_path, "PNG", **save_kwargs)
        else:
            result.save(out_path, "TIFF", **save_kwargs)

    def _resize_long_edge(self, image, limit: int):
        w, h = image.size
        longest = max(w, h)
        if longest <= int(limit):
            return image
        scale = int(limit) / float(longest)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        return image.resize(new_size, Image.LANCZOS)

    def _export_metadata_kwargs(self, opts):
        """Re-embed EXIF/ICC/DPI from the source when the format supports it."""
        meta = self._source_metadata or {}
        kwargs = {}
        ext = opts["ext"].lower()
        if meta.get("exif") and ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            kwargs["exif"] = meta["exif"]
        if meta.get("icc_profile"):
            kwargs["icc_profile"] = meta["icc_profile"]
        if meta.get("dpi"):
            kwargs["dpi"] = meta["dpi"]
        return kwargs

    def reset_all(self):
        self._begin_document_change()
        for layer, sliders in ALL_LAYERS.items():
            defaults = {key: default for key, _label, _mn, _mx, default in sliders}
            for key, slider in self._sliders.get(layer, {}).items():
                slider.blockSignals(True)
                slider.setValue(int(defaults[key]))
                slider.blockSignals(False)
                self._slider_value_labels[layer][key].setText("0")
        if self._auto_preview_masks is not None:
            self.preview_masks = self._copy_masks(self._auto_preview_masks)
        if self._auto_full_masks is not None:
            self.full_masks = self._copy_masks(self._auto_full_masks)
        self._mask_adjustments = default_mask_adjustments()
        self._sync_mask_adjustment_controls()
        self._framing = default_framing()
        self._sync_framing_controls()
        self._sync_wb_controls()
        self._sync_tone_curve_controls()
        self._sync_hsl_controls()
        self._schedule_render()
        self._push_document_history()


def run_qt_app():
    app = QApplication.instance() or QApplication([])
    window = PortraitEnhancerQtWindow()
    window.show()
    QTimer.singleShot(0, window._fit_initial_window_to_screen)
    # Make the window the active/key window on launch. On macOS, trackpad pinch-to-zoom arrives
    # as a NativeGesture that's only delivered while the window is key -- without this, the app
    # can open un-activated (e.g. launched from a terminal/IDE) and the first pinch-zoom is
    # swallowed until you click into the window. raise_()+activateWindow() avoids that dead first
    # gesture; focusing the canvas also primes Ctrl/Cmd+scroll zoom.
    window.raise_()
    window.activateWindow()
    if getattr(window, "image_label", None) is not None:
        window.image_label.setFocus()
    return app.exec()
