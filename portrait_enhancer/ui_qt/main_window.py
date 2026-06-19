"""PySide6 frontend for Portrait Enhancer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import base64
import subprocess
import sys
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import rawpy

    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QSize, QEvent, QPointF, QRectF, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
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
    QMessageBox,
    QInputDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from portrait_enhancer.config import ALL_LAYERS, LAYER_COLORS, LAYER_NAMES, MASK_ORDER
from portrait_enhancer.core import framing as framing_ops
from portrait_enhancer.core.framing import apply_framing, default_framing, normalize_framing
from portrait_enhancer.core.histogram import compute_histogram
from portrait_enhancer.core import white_balance as wb_ops
from portrait_enhancer.core import tone_curve as tc_ops
from portrait_enhancer.core import color_mixer as cm_ops
from portrait_enhancer.core.masks import apply_mask_adjustments, default_mask_adjustments, normalize_mask_adjustments
from portrait_enhancer.core.processing import process_all_layers, _scale_expression_guides, expression_warp_mode
from portrait_enhancer.core.refine import get_face_refiner
from portrait_enhancer.core.segmentation import FaceSegmenter
from portrait_enhancer.core.utils import resize_image, smooth_mask, to_uint8


class BatchExportDialog(QDialog):
    SUPPORTED_IMAGE_EXTS = (".cr2", ".nef", ".arw", ".dng", ".raw", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, parent=None, preset_path="", input_dir="", output_dir="", suffix="_enhanced", output_format="jpeg", skip_completed=True, profiles=None, selected_profile=""):
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

    def __init__(self, parent=None, output_dir="", stem="export", output_format="jpeg",
                 quality=92, has_metadata=False, source_w=0, source_h=0):
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
            "out_path": os.path.join(self.dest_edit.text().strip(), f"{name}{ext}"),
            "dest_dir": self.dest_edit.text().strip(),
        }


class BatchJobsDialog(QDialog):
    def __init__(self, parent=None, output_dir="", load_jobs_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Jobs")
        self.setModal(True)
        self.resize(980, 620)
        self._jobs = []
        self._load_jobs_callback = load_jobs_callback
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
            summary = (
                f"{job.get('created_at', 'unknown')} | "
                f"{job.get('mode', 'batch')} | "
                f"{job.get('status', 'unknown')} | "
                f"ok={job.get('success', 0)} skip={job.get('skipped', 0)} err={job.get('error', 0)}"
            )
            self.jobs_list.addItem(summary)
            if selected_job_id and job.get("job_id") == selected_job_id:
                selected_index = self.jobs_list.count() - 1
        if self._jobs:
            self.jobs_list.setCurrentRow(selected_index if selected_index >= 0 else 0)
        else:
            self.details.setPlainText("No batch jobs found in the selected output folder.")

    def _on_job_selected(self, index):
        if index < 0 or index >= len(self._jobs):
            self.details.clear()
            return
        job = self._jobs[index]
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
            "",
            f"Job File: {job.get('job_path', '')}",
            f"Batch Log: {job.get('log_path', '')}",
            f"Runner Log: {job.get('runner_log_path', '')}",
        ]
        if job.get("latest_error"):
            lines.extend(["", "Latest Error:", str(job["latest_error"])])
        self.details.setPlainText("\n".join(lines))

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
        heading.setStyleSheet("font-size: 15px; font-weight: 700;")
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


class ImagePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._pixmap = None
        self._scaled_pixmap = None
        self._zoom = 1.0
        self._min_zoom = 1.0
        self._max_zoom = 8.0
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
        self._wb_pick_enabled = False
        self._wb_pick_callback = None
        self._pixmap_image = None
        self._wb_hover_src = None
        self._wb_hover_widget = None

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

    def set_compare_state(self, mode: str, split_callback=None):
        self._compare_mode = str(mode or "off")
        self._split_callback = split_callback

    def set_edit_state(self, enabled: bool, paint_callback=None, paint_start_callback=None):
        self._edit_enabled = bool(enabled)
        self._paint_callback = paint_callback
        self._paint_start_callback = paint_start_callback
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
        self.update()

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
        if not self._edit_enabled and self._crop_overlay is None and not self._wb_pick_enabled:
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
        if self._crop_overlay is not None:
            self._paint_crop_overlay()
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
            # No edit tool active and zoomed in: left-drag pans the canvas.
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
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
                self._crop_overlay = framing_ops.resize_crop(self._crop_overlay, self._crop_drag_handle, dx, dy)
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
                self._panning = False
                self._pan_last = None
                self.setCursor(Qt.CrossCursor if self._wb_pick_enabled else Qt.ArrowCursor)
            self._dragging_split = False
            self._dragging_paint = False
        super().mouseReleaseEvent(event)

    def _apply_scaled_pixmap(self):
        # The image is painted manually in paintEvent (so it can be zoomed/panned);
        # QLabel only shows the placeholder text when there is no image.
        if self._pixmap is None:
            self._scaled_pixmap = None
            self.setText("Open an image to begin")
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
        shadow = self._data.get("shadow_clip", {})
        highlight = self._data.get("highlight_clip", {})
        shadow_clipped = any(v > self._CLIP_THRESHOLD for v in shadow.values())
        highlight_clipped = any(v > self._CLIP_THRESHOLD for v in highlight.values())
        marker = 6
        if shadow_clipped:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(91, 155, 213))
            painter.drawRect(rect.left(), rect.top(), marker, marker)
        if highlight_clipped:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(212, 168, 83))
            painter.drawRect(rect.right() - marker + 1, rect.top(), marker, marker)
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
            )
        except Exception as ex:
            self.signals.failed.emit(self.job_id, str(ex))
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.signals.finished.emit(self.job_id, result, float(elapsed_ms))


class ImportDialog(QDialog):
    """Dialog to import images from a folder."""

    def __init__(self, parent=None, supported_exts=None):
        super().__init__(parent)
        self.setWindowTitle("Import Images")
        self.setModal(True)
        self.resize(600, 200)
        self._supported_exts = supported_exts or {".cr2", ".nef", ".arw", ".dng", ".raw", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self._selected_folder = None
        self._image_count = 0

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

        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_btn = buttons.button(QDialogButtonBox.Ok)
        self.ok_btn.setEnabled(False)
        layout.addWidget(buttons)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Import")
        if not folder:
            return
        self._selected_folder = folder
        self.folder_label.setText(folder)

        # Count images in the folder.
        try:
            folder_path = Path(folder)
            images = [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in self._supported_exts]
            self._image_count = len(images)
            self.count_label.setText(f"Found {self._image_count} image{'s' if self._image_count != 1 else ''}")
            self.ok_btn.setEnabled(self._image_count > 0)
        except Exception as e:
            self.count_label.setText(f"Error: {str(e)}")
            self.ok_btn.setEnabled(False)

    def selected_folder(self):
        return self._selected_folder

    def image_count(self):
        return self._image_count


class FilmstripThumbnail(QFrame):
    """Clickable thumbnail widget for the image filmstrip."""

    clicked = Signal(str)  # emits the image path

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = str(path)
        self._pixmap = None
        self._is_active = False

        self.setObjectName("FilmstripThumbnail")
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(80)
        self.setFixedHeight(98)

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
        self.name_label.setStyleSheet("font-size: 9px;")
        fname = Path(path).stem
        self.name_label.setText(fname[:12] + ("..." if len(fname) > 12 else ""))
        self.name_label.setToolTip(Path(path).name)
        layout.addWidget(self.name_label)
        layout.addStretch(1)

        self.set_active(False)

    def set_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None:
            self.thumb_label.setText("No preview")
            self._pixmap = None
            return
        # Scale to fit the label while keeping aspect ratio.
        scaled = pixmap.scaled(72, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.thumb_label.setPixmap(scaled)
        self._pixmap = scaled

    def set_active(self, active: bool):
        self._is_active = bool(active)
        if active:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 2px solid #d4a853; background: #1a1a1a; }"
            )
        else:
            self.setStyleSheet(
                "FilmstripThumbnail { border: 1px solid #3a3a3a; background: #141414; }"
            )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)
            event.accept()
        else:
            super().mousePressEvent(event)


class PortraitEnhancerQtWindow(QMainWindow):
    SUPPORTED_IMAGE_EXTS = (".cr2", ".nef", ".arw", ".dng", ".raw", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    BATCH_OUTPUT_FORMATS = {
        "jpeg": ".jpg",
        "png": ".png",
        "tiff": ".tiff",
    }
    ESSENTIAL_SLIDERS = {
        "global": {"exposure", "temperature", "tint", "highlights", "shadows", "clarity", "vibrance"},
        "subjects": {"exposure", "clarity", "saturation", "warmth"},
        "background": {"blur", "exposure", "saturation", "dehaze"},
        "face": {"exposure", "smile", "eye_open", "brow_lift", "refine", "refine_fidelity"},
        "skin": {"smooth", "blemish", "warmth", "clarity"},
        "eyes": {"brightness", "whites", "sharpen", "iris_pop"},
        "lips": {"brightness", "saturation", "gloss", "hue_shift"},
        "hair": {"brightness", "highlights", "shine", "clarity"},
    }
    BASIC_LAYERS = ("global", "subjects", "background", "face", "skin")
    BROWSER_STATE_FILE = ".preset_browser_state.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portrait Enhancer Qt")
        self.resize(1560, 940)
        self.setFocusPolicy(Qt.StrongFocus)

        self.segmenter = FaceSegmenter()
        self.file_path = ""
        self.full_array = None
        self._source_metadata = {}
        self._last_export_format = "jpeg"
        self._last_export_quality = 92
        self.preview_array = None
        self.full_masks = None
        self.preview_masks = None
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
        self._active_layer = "global"
        self._show_mask = False
        self._mask_debug_mode = "tint"
        self._show_expression_guides = False
        self._mask_edit_enabled = False
        self._mask_paint_mode = "paint"
        self._mask_brush_size = 24
        self._mask_brush_hardness = 100
        self._mask_adjustments = default_mask_adjustments()
        self._mask_adjustment_sliders = {}
        self._mask_adjustment_labels = {}
        self._mask_history = []
        self._mask_history_index = -1
        self._max_mask_history = 40
        self._document_history = []
        self._document_history_index = -1
        self._max_document_history = 30
        self._restoring_document_state = False
        self._preset_library_entries = []
        self._recent_preset_paths = []
        self._preset_category_filter = "all"
        self._detected_faces = []
        self._active_face_index = 0
        self._face_profiles = {}
        self._sliders = {}
        self._slider_value_labels = {}
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
        self._slider_drag_active = 0
        self._focus_mode = False
        self._focus_restore = None
        self._settings_clipboard = None
        self._imported_images = []
        self._import_queue = []
        self._import_worker_timer = QTimer(self)
        self._import_worker_timer.setSingleShot(False)
        self._import_worker_timer.setInterval(500)
        self._import_worker_timer.timeout.connect(self._process_import_queue)
        self._import_thread_pool = QThreadPool(self)
        self._import_thread_pool.setMaxThreadCount(1)

        self._color_settings = self._default_color_settings()
        self._runtime_settings = {"acceleration_mode": "auto"}
        self._layer_order = list(MASK_ORDER)
        self._layer_options = {layer: {"enabled": True, "opacity": 100.0, "blend_mode": "normal"} for layer in MASK_ORDER}
        self._perf_stats = {"detect_ms": None, "segment_ms": None, "render_ms": None}
        self._essentials_only = True
        self._slider_blocks = {}
        self._section_buttons = {}
        self._section_contents = {}
        self._section_wrappers = {}
        self._section_state = {}

        self._build_ui()
        self._load_browser_state()
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
                font-size: 13px;
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
            QWidget#InspectorPanel,
            QWidget#InspectorPanel QWidget {
                background: #15171d;
            }
            QScrollArea#InspectorScroll {
                border: 1px solid #292f3a;
                border-radius: 8px;
            }
            QLabel#AppTitle {
                color: #f2eadc;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#ImageStatus {
                color: #f2eadc;
                font-weight: 600;
            }
            QLabel#MutedLabel {
                color: #9ea4ad;
            }
            QLabel#GroupHeader {
                color: #6f7682;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
                padding: 10px 2px 2px 2px;
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
            QToolBar#MainToolbar {
                background: #15171d;
                border: 0;
                spacing: 6px;
                padding: 6px;
            }
            QToolButton,
            QPushButton {
                background: #232832;
                border: 1px solid #363d49;
                border-radius: 6px;
                padding: 6px 9px;
                color: #ece8df;
            }
            QToolButton:hover,
            QPushButton:hover {
                background: #2e3541;
                border-color: #4b5667;
            }
            QToolButton:checked,
            QPushButton:checked {
                background: #36475a;
                border-color: #5b9bd5;
            }
            QToolButton:disabled,
            QPushButton:disabled,
            QLineEdit:disabled,
            QComboBox:disabled {
                background: #181d25;
                border-color: #252b35;
                color: #69717d;
            }
            QLabel:disabled,
            QCheckBox:disabled {
                color: #69717d;
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
                background: #101217;
                border: 1px solid #2d3440;
                border-radius: 6px;
                padding: 5px;
                selection-background-color: #5b9bd5;
            }
            QTabWidget::pane {
                border: 1px solid #2d3440;
                border-radius: 6px;
                background: #101217;
            }
            QTabBar::tab {
                background: #202632;
                color: #cfd3d8;
                padding: 7px 9px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #d4a853;
                color: #15120a;
                font-weight: 700;
            }
            QScrollArea {
                background: transparent;
                border: 0;
            }
            QScrollBar:vertical {
                background: #101217;
                width: 10px;
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
            QSlider::handle:horizontal {
                background: #d4a853;
                border: 1px solid #f0d08c;
                width: 14px;
                height: 14px;
                margin: -6px 0;
                border-radius: 7px;
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
        nav_panel.setMinimumWidth(280)
        nav_panel.setMaximumWidth(340)
        nav_layout = QVBoxLayout(nav_panel)
        nav_layout.setContentsMargins(12, 12, 12, 12)
        nav_layout.setSpacing(10)

        title = QLabel("Portrait Enhancer", self)
        title.setObjectName("AppTitle")
        nav_layout.addWidget(title)
        nav_layout.addWidget(make_button("Open Image", self.open_image, primary=True))

        project_grid = QGridLayout()
        project_grid.setContentsMargins(0, 0, 0, 0)
        project_grid.setHorizontalSpacing(6)
        project_grid.setVerticalSpacing(6)
        project_grid.addWidget(make_button("Open Project", self.open_project), 0, 0)
        project_grid.addWidget(make_button("Save Project", self.save_project), 0, 1)
        project_grid.addWidget(make_button("Open Preset", self.open_preset), 1, 0)
        project_grid.addWidget(make_button("Save Preset", self.save_preset), 1, 1)
        project_grid.addWidget(make_button("Recipes", self.open_recipe_dialog), 2, 0)
        project_grid.addWidget(make_button("System Check", self.show_system_check), 2, 1)
        nav_layout.addLayout(project_grid)

        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.addWidget(QLabel("Face Target", self))
        self.face_combo = QComboBox(self)
        self.face_combo.addItem("Auto")
        self.face_combo.currentIndexChanged.connect(self._on_face_changed)
        face_row.addWidget(self.face_combo, 1)
        nav_layout.addLayout(face_row)

        batch_grid = QGridLayout()
        batch_grid.setContentsMargins(0, 0, 0, 0)
        batch_grid.setHorizontalSpacing(6)
        batch_grid.setVerticalSpacing(6)
        batch_grid.addWidget(make_button("Batch Export", self.batch_export), 0, 0)
        batch_grid.addWidget(make_button("Batch Jobs", self.view_batch_jobs), 0, 1)
        batch_grid.addWidget(make_button("Retry Failed", self.retry_failed_batch), 1, 0)
        batch_grid.addWidget(make_button("Export", self.export_image), 1, 1)
        nav_layout.addLayout(batch_grid)

        preset_content = QWidget(self)
        preset_layout = QVBoxLayout(preset_content)
        preset_layout.setContentsMargins(8, 4, 8, 4)
        preset_layout.setSpacing(8)
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

        self.preset_preview_label = QLabel("No preset preview", self)
        self.preset_preview_label.setAlignment(Qt.AlignCenter)
        self.preset_preview_label.setMinimumHeight(104)
        self.preset_preview_label.setMaximumHeight(128)
        self.preset_preview_label.setObjectName("PreviewSwatch")
        preset_layout.addWidget(self.preset_preview_label)

        self.preset_meta_label = QLabel("No preset selected", self)
        self.preset_meta_label.setWordWrap(True)
        self.preset_meta_label.setObjectName("MutedLabel")
        preset_layout.addWidget(self.preset_meta_label)

        preset_btn_grid = QGridLayout()
        preset_btn_grid.setContentsMargins(0, 0, 0, 0)
        preset_btn_grid.setHorizontalSpacing(6)
        preset_btn_grid.setVerticalSpacing(6)
        self.apply_browser_preset_btn = QPushButton("Apply", self)
        self.apply_browser_preset_btn.clicked.connect(self._apply_selected_browser_preset)
        preset_btn_grid.addWidget(self.apply_browser_preset_btn, 0, 0)
        self.save_browser_preset_btn = QPushButton("Save Here", self)
        self.save_browser_preset_btn.clicked.connect(self._save_preset_to_library)
        preset_btn_grid.addWidget(self.save_browser_preset_btn, 0, 1)
        self.rename_browser_preset_btn = QPushButton("Rename", self)
        self.rename_browser_preset_btn.clicked.connect(self._rename_selected_browser_preset)
        preset_btn_grid.addWidget(self.rename_browser_preset_btn, 1, 0)
        self.delete_browser_preset_btn = QPushButton("Delete", self)
        self.delete_browser_preset_btn.clicked.connect(self._delete_selected_browser_preset)
        preset_btn_grid.addWidget(self.delete_browser_preset_btn, 1, 1)
        self.refresh_browser_preset_btn = QPushButton("Refresh", self)
        self.refresh_browser_preset_btn.clicked.connect(self._refresh_preset_browser)
        preset_btn_grid.addWidget(self.refresh_browser_preset_btn, 2, 0, 1, 2)
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
        canvas_header_layout.setContentsMargins(12, 10, 12, 10)
        canvas_header_layout.setSpacing(8)

        self.info_label = QLabel("No image loaded", self)
        self.info_label.setObjectName("ImageStatus")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        canvas_header_layout.addWidget(self.info_label)

        preview_controls = QHBoxLayout()
        preview_controls.setContentsMargins(0, 0, 0, 0)
        preview_controls.setSpacing(8)
        self.compare_hint_label = QLabel("Before/after", self)
        self.compare_hint_label.setObjectName("MutedLabel")
        self.compare_hint_label.setToolTip("Hold Space to view the original")
        preview_controls.addWidget(self.compare_hint_label)
        preview_controls.addStretch(1)
        preview_controls.addWidget(QLabel("Compare", self))
        self.compare_combo = QComboBox(self)
        self.compare_combo.addItems(["off", "before", "split", "side_by_side"])
        self.compare_combo.currentTextChanged.connect(self._on_compare_mode_changed)
        preview_controls.addWidget(self.compare_combo)
        preview_controls.addWidget(QLabel("Split", self))
        self.split_slider = QSlider(Qt.Horizontal, self)
        self.split_slider.setRange(0, 100)
        self.split_slider.setMaximumWidth(180)
        self.split_slider.setValue(int(self._split_position * 100))
        self.split_slider.setEnabled(self._compare_mode == "split")
        self.split_slider.valueChanged.connect(self._on_split_slider_changed)
        preview_controls.addWidget(self.split_slider)
        preview_controls.addSpacing(12)

        self.copy_settings_btn = QPushButton("Copy Settings", self)
        self.copy_settings_btn.setToolTip("Copy global and selective adjustments (Ctrl+Alt+C)")
        self.copy_settings_btn.setMaximumWidth(120)
        self.copy_settings_btn.clicked.connect(self._copy_settings)
        preview_controls.addWidget(self.copy_settings_btn)

        self.paste_settings_btn = QPushButton("Paste Settings", self)
        self.paste_settings_btn.setToolTip("Paste settings to this image (Ctrl+Alt+V)")
        self.paste_settings_btn.setMaximumWidth(120)
        self.paste_settings_btn.setEnabled(False)
        self.paste_settings_btn.clicked.connect(self._paste_settings)
        self._paste_settings_btn = self.paste_settings_btn
        preview_controls.addWidget(self.paste_settings_btn)

        self.settings_indicator = QLabel("", self)
        self.settings_indicator.setObjectName("MutedLabel")
        self.settings_indicator.setMaximumWidth(80)
        preview_controls.addWidget(self.settings_indicator)

        canvas_header_layout.addLayout(preview_controls)
        center_layout.addWidget(canvas_header)

        canvas_frame = QFrame(self)
        canvas_frame.setObjectName("CanvasSurface")
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.setSpacing(0)
        self.image_label = ImagePreviewLabel(self)
        self.image_label.setObjectName("ImagePreview")
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.image_label.set_edit_state(False, self._paint_active_mask_at, self._begin_mask_stroke)
        self.image_label.set_crop_callbacks(
            self._on_crop_changed, self._on_crop_committed, self._on_crop_drag_start
        )
        self.image_label.set_wb_pick_state(False, self._on_wb_picked)
        canvas_layout.addWidget(self.image_label, 1)
        center_layout.addWidget(canvas_frame, 1)

        # Filmstrip for quick image navigation.
        filmstrip_frame = QFrame(self)
        self._filmstrip_frame = filmstrip_frame
        filmstrip_frame.setObjectName("FilmstripStrip")
        filmstrip_frame.setMaximumHeight(110)
        filmstrip_frame.setMinimumHeight(0)
        filmstrip_layout = QVBoxLayout(filmstrip_frame)
        filmstrip_layout.setContentsMargins(8, 6, 8, 6)
        filmstrip_layout.setSpacing(0)

        filmstrip_label = QLabel("Images in folder", self)
        filmstrip_label.setObjectName("MutedLabel")
        filmstrip_layout.addWidget(filmstrip_label)

        filmstrip_scroll = QScrollArea(self)
        filmstrip_scroll.setWidgetResizable(True)
        filmstrip_scroll.setFrameShape(QFrame.NoFrame)
        filmstrip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        filmstrip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._filmstrip_host = QWidget(self)
        self._filmstrip_layout = QHBoxLayout(self._filmstrip_host)
        self._filmstrip_layout.setContentsMargins(0, 0, 0, 0)
        self._filmstrip_layout.setSpacing(6)
        filmstrip_scroll.setWidget(self._filmstrip_host)
        filmstrip_layout.addWidget(filmstrip_scroll, 1)

        self._filmstrip_items = {}  # path -> thumbnail widget
        self._filmstrip_images = []  # sorted list of image paths
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
        hist_row = QHBoxLayout()
        hist_row.setContentsMargins(0, 0, 0, 0)
        hist_row.setSpacing(8)
        hist_row.addWidget(QLabel("Histogram", self))
        self.histogram_widget = HistogramWidget(self)
        hist_row.addWidget(self.histogram_widget, 1)
        status_layout.addLayout(hist_row)
        center_layout.addWidget(status_panel)

        inspector_panel = QWidget(self)
        inspector_panel.setObjectName("InspectorPanel")
        inspector_panel.setMinimumWidth(380)
        inspector_panel.setMaximumWidth(480)
        inspector_layout = QVBoxLayout(inspector_panel)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(8)

        workspace_content = QWidget(self)
        workspace_layout = QVBoxLayout(workspace_content)
        workspace_layout.setContentsMargins(8, 4, 8, 4)
        workspace_layout.setSpacing(8)

        quick_layer_row = QHBoxLayout()
        quick_layer_row.setContentsMargins(0, 0, 0, 0)
        quick_layer_row.setSpacing(4)
        for layer in self.BASIC_LAYERS:
            button = QPushButton(self)
            button.setText(layer.title())
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.clicked.connect(lambda _checked=False, layer_name=layer: self._activate_layer(layer_name))
            quick_layer_row.addWidget(button)
        workspace_layout.addLayout(quick_layer_row)

        org_row = QHBoxLayout()
        org_row.setContentsMargins(0, 0, 0, 0)
        self.essentials_only_check = QCheckBox("Essentials Only", self)
        self.essentials_only_check.setChecked(self._essentials_only)
        self.essentials_only_check.toggled.connect(self._on_essentials_only_toggled)
        org_row.addWidget(self.essentials_only_check)
        org_row.addStretch(1)
        self.reset_layer_btn = QPushButton("Reset Layer", self)
        self.reset_layer_btn.clicked.connect(self._reset_active_layer)
        org_row.addWidget(self.reset_layer_btn)
        workspace_layout.addLayout(org_row)
        workspace_section = self._make_collapsible_section("Workspace", workspace_content, expanded=True)

        # Global adjustments live in their own "Basic" section; the layer tabs hold
        # only the portrait feature layers (subjects, background, face, ...).
        self._tab_layers = [layer for layer in ALL_LAYERS if layer != "global"]

        global_content = QWidget(self)
        global_layout = QVBoxLayout(global_content)
        global_layout.setContentsMargins(8, 4, 8, 4)
        global_layout.setSpacing(6)
        global_layout.addWidget(self._build_slider_stack("global", ALL_LAYERS["global"], add_stretch=False))
        global_section = self._make_collapsible_section("Global", global_content, expanded=True)

        layers_content = QWidget(self)
        layers_layout = QVBoxLayout(layers_content)
        layers_layout.setContentsMargins(0, 4, 0, 4)
        layers_layout.setSpacing(6)
        self.layer_tabs = QTabWidget(self)
        self.layer_tabs.setDocumentMode(True)
        for layer in self._tab_layers:
            self.layer_tabs.addTab(self._build_layer_tab(layer, ALL_LAYERS[layer]), LAYER_NAMES[layer])
        self.layer_tabs.currentChanged.connect(self._on_layer_changed)
        layers_layout.addWidget(self.layer_tabs)
        layers_section = self._make_collapsible_section("Layers", layers_content, expanded=True)

        mask_content = QWidget(self)
        mask_layout = QVBoxLayout(mask_content)
        mask_layout.setContentsMargins(8, 4, 8, 4)
        mask_layout.setSpacing(8)
        mask_row = QHBoxLayout()
        self.mask_view_btn = QPushButton("Mask View", self)
        self.mask_view_btn.setCheckable(True)
        self.mask_view_btn.toggled.connect(self._on_mask_view_toggled)
        mask_row.addWidget(self.mask_view_btn)
        self.mask_edit_btn = QPushButton("Edit Mask", self)
        self.mask_edit_btn.setCheckable(True)
        self.mask_edit_btn.toggled.connect(self._on_mask_edit_toggled)
        mask_row.addWidget(self.mask_edit_btn)
        mask_layout.addLayout(mask_row)

        debug_row = QHBoxLayout()
        debug_row.addWidget(QLabel("View", self))
        self.mask_debug_combo = QComboBox(self)
        self.mask_debug_combo.addItems(["tint", "heatmap", "isolated"])
        self.mask_debug_combo.currentTextChanged.connect(self._on_mask_debug_mode_changed)
        debug_row.addWidget(self.mask_debug_combo, 1)
        self.guides_btn = QPushButton("Guides", self)
        self.guides_btn.setCheckable(True)
        self.guides_btn.toggled.connect(self._on_guides_toggled)
        debug_row.addWidget(self.guides_btn)
        mask_layout.addLayout(debug_row)

        self.mask_debug_label = QLabel("Mask: --", self)
        self.mask_debug_label.setObjectName("MutedLabel")
        self.mask_debug_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_debug_label)

        mask_layout.addWidget(QLabel("Mask Settings", self))
        for key, label, mn, mx in (
            ("strength", "Strength", 0, 200),
            ("feather", "Feather", 0, 40),
            ("expand", "Expand", -40, 40),
        ):
            mask_layout.addLayout(self._build_mask_adjustment_row(key, label, mn, mx))

        reset_settings_row = QHBoxLayout()
        reset_settings_row.addStretch(1)
        self.reset_mask_settings_btn = QPushButton("Reset Settings", self)
        self.reset_mask_settings_btn.clicked.connect(self._reset_active_mask_settings)
        reset_settings_row.addWidget(self.reset_mask_settings_btn)
        mask_layout.addLayout(reset_settings_row)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush", self))
        self.brush_slider = QSlider(Qt.Horizontal, self)
        self.brush_slider.setRange(2, 80)
        self.brush_slider.setValue(self._mask_brush_size)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        brush_row.addWidget(self.brush_slider, 1)
        self.brush_value_label = QLabel(str(self._mask_brush_size), self)
        brush_row.addWidget(self.brush_value_label)
        mask_layout.addLayout(brush_row)

        hardness_row = QHBoxLayout()
        hardness_row.addWidget(QLabel("Hardness", self))
        self.hardness_slider = QSlider(Qt.Horizontal, self)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(self._mask_brush_hardness)
        self.hardness_slider.valueChanged.connect(self._on_brush_hardness_changed)
        hardness_row.addWidget(self.hardness_slider, 1)
        self.hardness_value_label = QLabel(f"{self._mask_brush_hardness}%", self)
        hardness_row.addWidget(self.hardness_value_label)
        mask_layout.addLayout(hardness_row)

        mask_mode_row = QHBoxLayout()
        mask_mode_row.addWidget(QLabel("Mode", self))
        self.mask_mode_combo = QComboBox(self)
        self.mask_mode_combo.addItems(["paint", "erase"])
        self.mask_mode_combo.currentTextChanged.connect(self._on_mask_mode_changed)
        mask_mode_row.addWidget(self.mask_mode_combo, 1)
        self.reset_mask_btn = QPushButton("Reset Mask", self)
        self.reset_mask_btn.clicked.connect(self._reset_active_mask)
        mask_mode_row.addWidget(self.reset_mask_btn)
        self.feather_mask_btn = QPushButton("Feather", self)
        self.feather_mask_btn.clicked.connect(self._feather_active_mask)
        mask_mode_row.addWidget(self.feather_mask_btn)
        mask_layout.addLayout(mask_mode_row)

        history_row = QHBoxLayout()
        self.undo_mask_btn = QPushButton("Undo", self)
        self.undo_mask_btn.clicked.connect(self._undo_mask_edit)
        history_row.addWidget(self.undo_mask_btn)
        self.redo_mask_btn = QPushButton("Redo", self)
        self.redo_mask_btn.clicked.connect(self._redo_mask_edit)
        history_row.addWidget(self.redo_mask_btn)
        mask_layout.addLayout(history_row)
        masks_section = self._make_collapsible_section("Masks", mask_content, expanded=False)

        geometry_content = QWidget(self)
        geometry_layout = QVBoxLayout(geometry_content)
        geometry_layout.setContentsMargins(8, 4, 8, 4)
        geometry_layout.setSpacing(8)

        self.crop_edit_btn = QPushButton("Crop", self)
        self.crop_edit_btn.setCheckable(True)
        self.crop_edit_btn.toggled.connect(self._on_crop_edit_toggled)
        geometry_layout.addWidget(self.crop_edit_btn)

        aspect_row = QHBoxLayout()
        aspect_row.addWidget(QLabel("Aspect", self))
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
        geometry_layout.addLayout(aspect_row)

        straighten_row = QHBoxLayout()
        straighten_row.addWidget(QLabel("Straighten", self))
        self.straighten_slider = QSlider(Qt.Horizontal, self)
        self.straighten_slider.setRange(-45, 45)
        self.straighten_slider.setValue(0)
        self.straighten_slider.valueChanged.connect(self._on_straighten_changed)
        straighten_row.addWidget(self.straighten_slider, 1)
        self.straighten_value_label = QLabel("0°", self)
        straighten_row.addWidget(self.straighten_value_label)
        geometry_layout.addLayout(straighten_row)

        flip_row = QHBoxLayout()
        self.flip_h_btn = QPushButton("Flip H", self)
        self.flip_h_btn.clicked.connect(lambda: self._on_flip("flip_h"))
        flip_row.addWidget(self.flip_h_btn)
        self.flip_v_btn = QPushButton("Flip V", self)
        self.flip_v_btn.clicked.connect(lambda: self._on_flip("flip_v"))
        flip_row.addWidget(self.flip_v_btn)
        self.reset_framing_btn = QPushButton("Reset", self)
        self.reset_framing_btn.clicked.connect(self._reset_framing)
        flip_row.addWidget(self.reset_framing_btn)
        geometry_layout.addLayout(flip_row)
        geometry_section = self._make_collapsible_section("Geometry", geometry_content, expanded=False)

        wb_content = QWidget(self)
        wb_layout = QVBoxLayout(wb_content)
        wb_layout.setContentsMargins(8, 4, 8, 4)
        wb_layout.setSpacing(8)
        self.wb_pick_btn = QPushButton("Pick Neutral", self)
        self.wb_pick_btn.setCheckable(True)
        self.wb_pick_btn.setToolTip("Click a neutral area in the image")
        self.wb_pick_btn.toggled.connect(self._on_wb_pick_toggled)
        wb_layout.addWidget(self.wb_pick_btn)
        wb_auto_row = QHBoxLayout()
        self.wb_gray_btn = QPushButton("Auto Gray", self)
        self.wb_gray_btn.clicked.connect(self._wb_auto_gray_world)
        wb_auto_row.addWidget(self.wb_gray_btn)
        self.wb_white_btn = QPushButton("Auto White", self)
        self.wb_white_btn.clicked.connect(self._wb_auto_white_patch)
        wb_auto_row.addWidget(self.wb_white_btn)
        self.wb_reset_btn = QPushButton("Reset", self)
        self.wb_reset_btn.clicked.connect(self._reset_wb)
        wb_auto_row.addWidget(self.wb_reset_btn)
        wb_layout.addLayout(wb_auto_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset", self))
        self.wb_preset_combo = QComboBox(self)
        self.wb_preset_combo.addItem("Custom")
        for name, _temp, _tint in wb_ops.PRESETS:
            self.wb_preset_combo.addItem(name)
        self.wb_preset_combo.activated.connect(self._on_wb_preset_chosen)
        preset_row.addWidget(self.wb_preset_combo, 1)
        wb_layout.addLayout(preset_row)

        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temp", self))
        self.wb_temp_slider = QSlider(Qt.Horizontal, self)
        self.wb_temp_slider.setRange(wb_ops.MIN_K, wb_ops.MAX_K)
        self.wb_temp_slider.setValue(wb_ops.NEUTRAL_K)
        self.wb_temp_slider.setToolTip("Color temperature in Kelvin")
        self.wb_temp_slider.sliderPressed.connect(self._begin_document_change)
        self.wb_temp_slider.valueChanged.connect(self._on_wb_temp_changed)
        self.wb_temp_slider.sliderReleased.connect(self._push_document_history)
        temp_row.addWidget(self.wb_temp_slider, 1)
        self.wb_temp_value_label = QLabel(f"{wb_ops.NEUTRAL_K}K", self)
        temp_row.addWidget(self.wb_temp_value_label)
        wb_layout.addLayout(temp_row)

        tint_row = QHBoxLayout()
        tint_row.addWidget(QLabel("Tint", self))
        self.wb_tint_slider = QSlider(Qt.Horizontal, self)
        self.wb_tint_slider.setRange(wb_ops.TINT_MIN, wb_ops.TINT_MAX)
        self.wb_tint_slider.setValue(0)
        self.wb_tint_slider.sliderPressed.connect(self._begin_document_change)
        self.wb_tint_slider.valueChanged.connect(self._on_wb_tint_changed)
        self.wb_tint_slider.sliderReleased.connect(self._push_document_history)
        tint_row.addWidget(self.wb_tint_slider, 1)
        self.wb_tint_value_label = QLabel("0", self)
        tint_row.addWidget(self.wb_tint_value_label)
        wb_layout.addLayout(tint_row)

        self.wb_status_label = QLabel("White balance: neutral", self)
        self.wb_status_label.setObjectName("MutedLabel")
        wb_layout.addWidget(self.wb_status_label)
        wb_section = self._make_collapsible_section("White Balance", wb_content, expanded=False)

        curve_content = QWidget(self)
        curve_layout = QVBoxLayout(curve_content)
        curve_layout.setContentsMargins(8, 4, 8, 4)
        curve_layout.setSpacing(6)
        self.tone_curve_widget = ToneCurveWidget(self)
        self.tone_curve_widget.set_callbacks(
            self._on_tone_curve_changed, self._on_tone_curve_commit, self._on_tone_curve_start
        )
        curve_layout.addWidget(self.tone_curve_widget)
        curve_reset_row = QHBoxLayout()
        curve_reset_row.addStretch(1)
        self.tone_curve_reset_btn = QPushButton("Reset Curve", self)
        self.tone_curve_reset_btn.clicked.connect(self._reset_tone_curve)
        curve_reset_row.addWidget(self.tone_curve_reset_btn)
        curve_layout.addLayout(curve_reset_row)
        curve_section = self._make_collapsible_section("Tone Curve", curve_content, expanded=False)

        hsl_content = QWidget(self)
        hsl_layout = QVBoxLayout(hsl_content)
        hsl_layout.setContentsMargins(8, 4, 8, 4)
        hsl_layout.setSpacing(8)
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("Color", self))
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
            row.addWidget(QLabel(label, self))
            slider = QSlider(Qt.Horizontal, self)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.sliderPressed.connect(self._begin_document_change)
            slider.valueChanged.connect(lambda value, k=key: self._on_hsl_slider_changed(k, value))
            slider.sliderReleased.connect(self._push_document_history)
            row.addWidget(slider, 1)
            value_label = QLabel("0", self)
            row.addWidget(value_label)
            hsl_layout.addLayout(row)
            self.hsl_sliders[key] = slider
            self.hsl_value_labels[key] = value_label

        hsl_reset_row = QHBoxLayout()
        hsl_reset_row.addStretch(1)
        self.hsl_reset_band_btn = QPushButton("Reset Color", self)
        self.hsl_reset_band_btn.clicked.connect(self._reset_hsl_band)
        hsl_reset_row.addWidget(self.hsl_reset_band_btn)
        self.hsl_reset_all_btn = QPushButton("Reset All", self)
        self.hsl_reset_all_btn.clicked.connect(self._reset_hsl_all)
        hsl_reset_row.addWidget(self.hsl_reset_all_btn)
        hsl_layout.addLayout(hsl_reset_row)
        hsl_section = self._make_collapsible_section("Color Mixer (HSL)", hsl_content, expanded=False)

        inspector_layout.addWidget(self._make_group_header("Portrait"))
        inspector_layout.addWidget(workspace_section)
        inspector_layout.addWidget(layers_section)
        inspector_layout.addWidget(self._make_group_header("Masks"))
        inspector_layout.addWidget(masks_section)
        inspector_layout.addWidget(self._make_group_header("Basic"))
        inspector_layout.addWidget(global_section)
        inspector_layout.addWidget(geometry_section)
        inspector_layout.addWidget(wb_section)
        inspector_layout.addWidget(self._make_group_header("Color"))
        inspector_layout.addWidget(curve_section)
        inspector_layout.addWidget(hsl_section)
        inspector_layout.addStretch(1)

        inspector_scroll = QScrollArea(self)
        self._inspector_scroll = inspector_scroll
        inspector_scroll.setObjectName("InspectorScroll")
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFrameShape(QFrame.NoFrame)
        inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inspector_scroll.setWidget(inspector_panel)
        inspector_scroll.viewport().setStyleSheet("background: #15171d;")

        splitter.addWidget(nav_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(inspector_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 860, 420])
        self._splitter = splitter

        self._apply_window_style()
        self._refresh_mask_controls()
        self._refresh_preset_browser()

    def _build_ui(self):
        self._build_editor_shell()

    def _build_mask_adjustment_row(self, key: str, label: str, mn: int, mx: int):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(mn), int(mx))
        slider.setValue(100 if key == "strength" else 0)
        slider.sliderPressed.connect(self._begin_document_change)
        slider.sliderPressed.connect(self._on_slider_drag_started)
        slider.sliderReleased.connect(self._on_slider_drag_finished)
        slider.valueChanged.connect(lambda value, name=key: self._on_mask_adjustment_changed(name, value))
        row.addWidget(slider, 1)
        value_label = QLabel(self._format_mask_adjustment_value(key, slider.value()))
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

    def _build_slider_stack(self, layer: str, sliders, add_stretch: bool = True):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        for key, label, mn, mx, default in sliders:
            block = QWidget()
            block_layout = QVBoxLayout(block)
            block_layout.setContentsMargins(0, 0, 0, 0)
            block_layout.setSpacing(4)

            top = QHBoxLayout()
            title = QLabel(label)
            value_label = QLabel("0")
            top.addWidget(title)
            top.addStretch(1)
            top.addWidget(value_label)
            block_layout.addLayout(top)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(int(mn), int(mx))
            slider.setValue(int(default))
            slider.sliderPressed.connect(self._begin_document_change)
            slider.sliderPressed.connect(self._on_slider_drag_started)
            slider.sliderReleased.connect(self._on_slider_drag_finished)
            slider.valueChanged.connect(self._make_slider_handler(layer, key, value_label))
            block_layout.addWidget(slider)

            self._sliders.setdefault(layer, {})[key] = slider
            self._slider_value_labels.setdefault(layer, {})[key] = value_label
            self._slider_blocks.setdefault(layer, {})[key] = block
            layout.addWidget(block)

        if add_stretch:
            layout.addStretch(1)
        return content

    def _make_group_header(self, text: str) -> QLabel:
        label = QLabel(text.upper(), self)
        label.setObjectName("GroupHeader")
        return label

    def _section_button_text(self, title: str, expanded: bool) -> str:
        marker = "v" if expanded else ">"
        return f"{marker} {title}"

    def _make_collapsible_section(self, title: str, content: QWidget, expanded: bool = True):
        expanded = bool(self._section_state.get(title, expanded))
        wrapper = QWidget(self)
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QPushButton(self)
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
            "working_space": "srgb",
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
        }

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
        return normalized

    @staticmethod
    def _humanize_reason(reason: str) -> str:
        text = str(reason or "").strip()
        low = text.lower()
        if "onnxruntime" in low:
            return "ONNX Runtime is not installed — advanced portrait masks use fallback mode."
        if "mediapipe" in low:
            return "MediaPipe is not installed — using fallback mode."
        if "no module named" in low:
            return "A required Python package is missing — using fallback mode."
        if any(token in low for token in ("not found", "missing", "no model", "does not exist", "unavailable")):
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
        subject_segmenter = getattr(self.segmenter, "_subjects", None)
        facial_hair_segmenter = getattr(self.segmenter, "_facial_hair", None)
        refiner = get_face_refiner()
        refiner._ensure_session()

        def reason_of(obj):
            return str(getattr(obj, "reason_unavailable", "") or "") if obj is not None else ""

        # name, component-reason, degraded-status when unavailable, ready description
        specs = [
            ("Face parsing", reason_of(parser), "fallback",
             "Segments facial regions so Skin / Eyes / Lips / Hair edits stay local."),
            ("Face detector", reason_of(detector), "fallback",
             "Finds faces to target for selective edits."),
            ("Subject selection", reason_of(subject_segmenter), "fallback",
             "Separates subject from background for Subjects / Background layers."),
            ("Facial hair", reason_of(facial_hair_segmenter), "off",
             "Optional model that excludes facial hair from skin smoothing."),
        ]

        items = []
        for name, reason, degraded, description in specs:
            if reason:
                items.append({"name": name, "status": degraded,
                              "message": self._humanize_reason(reason), "detail": reason})
            else:
                items.append({"name": name, "status": "ready", "message": description, "detail": ""})

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
        if isinstance(section_state, dict):
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
                "section_state": dict(self._section_state),
                "readiness_seen": True,
            }
        )
        self._write_browser_state(payload)

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
        }

    def _copy_settings(self):
        """Copy global + selective adjustments, layer settings, and color management to clipboard."""
        if self.preview_array is None:
            self.statusBar().showMessage("No image loaded — nothing to copy")
            return
        self._store_active_face_profile()
        params = self._all_params()
        selective_by_face = []
        for face_idx in sorted(self._face_profiles.keys()):
            profile = self._face_profiles.get(face_idx)
            if profile:
                selective = {layer: dict(profile.get("selective_params", {}).get(layer, {})) for layer in MASK_ORDER}
                selective_by_face.append({"face_index": face_idx, "params": selective})
        self._settings_clipboard = {
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "selective_by_face": selective_by_face,
        }
        self._paste_settings_action.setEnabled(True)
        if hasattr(self, '_paste_settings_btn'):
            self._paste_settings_btn.setEnabled(True)
        if hasattr(self, 'settings_indicator'):
            self.settings_indicator.setText("copied ✓")
        self.statusBar().showMessage("Settings copied ✓")

    def _paste_settings(self):
        """Paste copied settings to the current image."""
        if self._settings_clipboard is None:
            self.statusBar().showMessage("No settings in clipboard")
            return
        if self.preview_array is None:
            self.statusBar().showMessage("No image loaded — nothing to paste to")
            return
        self._begin_document_change()
        clipboard = self._settings_clipboard
        # Apply global params.
        global_params = clipboard.get("global_params", {})
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
        color_settings = clipboard.get("color_settings", {})
        if isinstance(color_settings, dict):
            merged = self._default_color_settings()
            merged.update(color_settings)
            self._color_settings = merged
        # Sync tone curve UI if visible.
        if hasattr(self, '_sync_tone_curve_controls'):
            self._sync_tone_curve_controls()
        # Apply layer options and order.
        layer_options = clipboard.get("layer_options", {})
        if isinstance(layer_options, dict):
            for layer in MASK_ORDER:
                cfg = dict(self._layer_options.get(layer, {}))
                cfg.update(layer_options.get(layer, {}))
                self._layer_options[layer] = cfg
        layer_order = [layer for layer in clipboard.get("layer_order", []) if layer in MASK_ORDER]
        for layer in MASK_ORDER:
            if layer not in layer_order:
                layer_order.append(layer)
        self._layer_order = layer_order
        # Sync mask adjustment controls.
        if hasattr(self, '_sync_mask_adjustment_controls'):
            self._sync_mask_adjustment_controls()
        # Apply selective params to detected faces.
        selective_by_face = clipboard.get("selective_by_face", [])
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
        self._schedule_render()
        self._push_document_history()
        if hasattr(self, 'settings_indicator'):
            self.settings_indicator.setText("")
        self.statusBar().showMessage("Settings pasted ✓")

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
            self._sync_wb_controls()
            self._sync_tone_curve_controls()
            self._sync_hsl_controls()

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
        self._schedule_render()

    def _update_document_history_actions(self):
        can_undo = self._document_history_index > 0
        can_redo = self._document_history_index >= 0 and self._document_history_index < len(self._document_history) - 1
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(can_undo)
        if hasattr(self, "redo_action"):
            self.redo_action.setEnabled(can_redo)

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
        if guides is None:
            return None
        if isinstance(guides, list):
            return [self._copy_guides(item) for item in guides]
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
        self.preview_masks = self._copy_masks(profile.get("preview_masks"))
        self.full_masks = self._copy_masks(profile.get("full_masks"))
        self._auto_preview_masks = self._copy_masks(profile.get("auto_preview_masks"))
        self._auto_full_masks = self._copy_masks(profile.get("auto_full_masks"))
        self.preview_guides = self._copy_guides(profile.get("preview_guides"))
        self.full_guides = self._copy_guides(profile.get("full_guides"))
        self._clear_mask_history()
        self._push_mask_history()
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
        return {
            "version": 1,
            "global_params": dict(params.get("global", {})),
            "color_settings": dict(self._color_settings),
            "selective_params": {layer: dict(params.get(layer, {})) for layer in MASK_ORDER},
            "layer_options": {layer: dict(cfg) for layer, cfg in self._layer_options.items()},
            "layer_order": list(self._layer_order),
            "mask_adjustments": self._copy_mask_adjustments(self._mask_adjustments),
            "meta": self._default_preset_meta(),
        }

    def _apply_preset_state(self, preset: dict):
        self._begin_document_change()
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
        self._schedule_render()

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
            self._apply_preset_state(preset)
            self._remember_recent_preset(path)
            self.statusBar().showMessage(f"Preset loaded -> {os.path.basename(path)}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", str(ex))

    def save_preset(self):
        default_name = "portrait_enhancer_qt_preset.pepreset"
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
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self._save_preset_thumbnail(out_path)
            self._remember_recent_preset(out_path)
            self.statusBar().showMessage(f"Preset saved -> {os.path.basename(out_path)}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Save Error", str(ex))

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

    def _save_preset_thumbnail(self, preset_path):
        if self.preview_image is None:
            return
        try:
            thumb = self.preview_image.convert("RGB").copy()
            thumb.thumbnail((320, 220), Image.LANCZOS)
            thumb.save(self._preset_thumbnail_path(preset_path), quality=90)
        except Exception:
            return

    def _set_preset_preview(self, preset_path):
        if not hasattr(self, "preset_preview_label"):
            return
        thumb_path = self._preset_thumbnail_path(preset_path)
        if thumb_path is None or not thumb_path.exists():
            self.preset_preview_label.setPixmap(QPixmap())
            self.preset_preview_label.setText("No preset preview")
            return
        pixmap = QPixmap(str(thumb_path))
        if pixmap.isNull():
            self.preset_preview_label.setPixmap(QPixmap())
            self.preset_preview_label.setText("No preset preview")
            return
        scaled = pixmap.scaled(320, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preset_preview_label.setText("")
        self.preset_preview_label.setPixmap(scaled)

    def _preset_list_icon(self, preset_path):
        thumb_path = self._preset_thumbnail_path(preset_path)
        pixmap = QPixmap(str(thumb_path)) if thumb_path is not None and thumb_path.exists() else QPixmap()
        if pixmap.isNull():
            pixmap = QPixmap(140, 96)
            pixmap.fill(QColor("#1a1a1f"))
            painter = QPainter(pixmap)
            painter.setPen(QColor("#5a5650"))
            painter.drawRect(0, 0, pixmap.width() - 1, pixmap.height() - 1)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "No Preview")
            painter.end()
        else:
            pixmap = pixmap.scaled(140, 96, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
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
        self.preset_list.clear()
        for entry in self._filtered_preset_entries():
            meta = entry["meta"]
            is_recent = str(entry["path"].resolve()) in self._recent_preset_paths
            prefix = "★ " if is_recent else ""
            item = QListWidgetItem(
                self._preset_list_icon(entry["path"]),
                f"{prefix}{meta.get('name', entry['path'].stem)}\n{meta.get('category', 'general')}",
            )
            tags = ", ".join(meta.get("tags", [])) or "none"
            item.setToolTip(
                f"{meta.get('name', entry['path'].stem)}\n"
                f"Category: {meta.get('category', 'general')}\n"
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
            self.preset_meta_label.setText(
                f"Name: {meta.get('name', Path(recent_path).stem)}\n"
                f"Category: {meta.get('category', 'general')}\n"
                f"Affects: {self._preset_scope_label(preset)}\n"
                f"Tags: {', '.join(meta.get('tags', [])) or 'none'}\n"
                f"Saved: {meta.get('saved_at') or 'unknown'}\n"
                f"Path: {recent_path}"
            )
            self._set_preset_preview(recent_path)
            return
        if entry is None:
            self.preset_meta_label.setText("No preset selected")
            self._set_preset_preview("")
            return
        meta = entry["meta"]
        tags = ", ".join(meta.get("tags", [])) or "none"
        saved_at = meta.get("saved_at") or "unknown"
        scope = self._preset_scope_label(entry.get("preset") or {})
        self.preset_meta_label.setText(
            f"Name: {meta.get('name', entry['path'].stem)}\n"
            f"Category: {meta.get('category', 'general')}\n"
            f"Affects: {scope}\n"
            f"Tags: {tags}\n"
            f"Saved: {saved_at}\n"
            f"Path: {entry['path']}"
        )
        self._set_preset_preview(entry["path"])

    def _apply_selected_browser_preset(self):
        entry = self._selected_browser_entry()
        if entry is None:
            QMessageBox.information(self, "Preset Browser", "Select a preset from the library first.")
            return
        try:
            preset = entry.get("preset") or self._load_preset_file(entry["path"])
            self._apply_preset_state(preset)
            self._remember_recent_preset(str(entry["path"]))
            self.statusBar().showMessage(f"Preset loaded -> {entry['path'].name}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", str(ex))

    def _apply_selected_recent_preset(self):
        path = self._selected_recent_path()
        if not path:
            QMessageBox.information(self, "Recent Presets", "Select a recent preset first.")
            return
        try:
            preset = self._load_preset_file(path)
            self._apply_preset_state(preset)
            self._remember_recent_preset(path)
            self.statusBar().showMessage(f"Preset loaded -> {Path(path).name}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Load Error", str(ex))

    def _save_preset_to_library(self):
        preset_dir = self._preset_library_dir()
        name = f"{Path(self.file_path).stem}_preset" if self.file_path else "portrait_recipe"
        out_path = preset_dir / f"{name}.pepreset"
        try:
            preset = self._serialize_preset_state()
            preset["meta"]["name"] = Path(out_path).stem
            preset["meta"]["saved_at"] = datetime.now(timezone.utc).isoformat()
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(preset, fh)
            self._save_preset_thumbnail(out_path)
            self._remember_recent_preset(str(out_path))
            self._refresh_preset_browser()
            self.statusBar().showMessage(f"Preset saved -> {out_path.name}")
        except Exception as ex:
            QMessageBox.critical(self, "Preset Save Error", str(ex))

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
            QMessageBox.critical(self, "Preset Rename Error", str(ex))

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
            QMessageBox.critical(self, "Preset Delete Error", str(ex))

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

    def open_recipe_dialog(self):
        dialog = RecipeDialog(self._guided_recipes(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        recipe = dialog.selected_recipe()
        if not recipe:
            return
        self._begin_document_change()
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
        self._apply_preset_state(preset)
        self.statusBar().showMessage(f"Recipe applied -> {recipe.get('name', 'Recipe')}")

    def show_system_check(self):
        items, _has_issues, models_dir = self._readiness_items()
        ReadinessDialog(
            items, self, title="System Check",
            recheck_callback=self._readiness_items, models_dir=models_dir,
        ).exec()

    def _maybe_show_startup_readiness(self):
        state = self._browser_state_payload()
        items, has_issues, models_dir = self._readiness_items()
        if not has_issues and state.get("readiness_seen"):
            return
        ReadinessDialog(
            items, self, title="Startup Readiness",
            recheck_callback=self._readiness_items, models_dir=models_dir,
        ).exec()
        state["readiness_seen"] = True
        self._write_browser_state(state)

    def _on_left_panel_toggled(self, visible: bool):
        self._nav_panel.setVisible(bool(visible))

    def _on_right_panel_toggled(self, visible: bool):
        self._inspector_scroll.setVisible(bool(visible))

    def _on_focus_mode_toggled(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self._focus_mode:
            return
        self._focus_mode = enabled
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

    def _activate_layer(self, layer: str):
        if layer == "global":
            # Global adjustments are their own section now, not a layer tab.
            self._active_layer = "global"
            self._set_section_expanded("Global", True)
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
        self._refresh_mask_controls()
        self._update_preview_label()

    def _on_essentials_only_toggled(self, checked: bool):
        if bool(checked) != self._essentials_only:
            self._begin_document_change()
        self._essentials_only = bool(checked)
        self._apply_essentials_filter()

    def _apply_essentials_filter(self):
        for layer, blocks in self._slider_blocks.items():
            essentials = self.ESSENTIAL_SLIDERS.get(layer, set())
            for key, block in blocks.items():
                block.setVisible((not self._essentials_only) or key in essentials)

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
        if self.preview_array is None:
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
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
        return str(job_path)

    def _launch_background_batch_job(self, job_path, output_dir):
        runner_log = os.path.join(output_dir, "batch_runner_stdout.log")
        with open(runner_log, "ab") as fh:
            subprocess.Popen(
                [sys.executable, "-m", "portrait_enhancer.batch_runner", "--job", job_path],
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                cwd=os.getcwd(),
            )
        return runner_log

    def _read_batch_job_payload(self, job_path):
        try:
            with open(job_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _load_batch_jobs(self, output_dir):
        jobs_dir = Path(output_dir) / ".batch_jobs"
        log_path = os.path.join(output_dir, "batch_export_log.jsonl")
        runner_log_path = os.path.join(output_dir, "batch_runner_stdout.log")
        records = self._load_batch_log_records(log_path)
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
                source_paths = payload.get("source_paths") or []
                if source_paths:
                    source_count = len(source_paths)
                else:
                    source_count = len(self._supported_image_paths(payload.get("input_dir", "")))
                status = "queued"
                if job_records:
                    if error:
                        status = "error"
                    elif success + skipped >= source_count and source_count > 0:
                        status = "complete"
                    else:
                        status = "running"
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
                        "status": status,
                        "latest_error": next((r.get("error") for r in reversed(job_records) if r.get("status") == "error"), ""),
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
        if masks is None:
            return None
        h, w = shape_hw
        scaled = {}
        for key, mask in masks.items():
            pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
            pil = pil.resize((w, h), Image.BILINEAR)
            scaled[key] = np.asarray(pil, dtype=np.float32) / 255.0
        return scaled

    def _on_face_changed(self, index: int):
        if self.preview_array is None:
            return
        if max(0, index) != self._active_face_index:
            self._begin_document_change()
        self._store_active_face_profile()
        self._active_face_index = max(0, index)
        self._run_segmentation()

    def _on_slider_drag_started(self):
        self._slider_drag_active += 1

    def _on_slider_drag_finished(self):
        self._slider_drag_active = max(0, self._slider_drag_active - 1)
        self._schedule_render()

    def _on_layer_changed(self, index: int):
        self._active_layer = self._tab_layers[max(0, min(index, len(self._tab_layers) - 1))]
        self._refresh_mask_controls()
        self._update_preview_label()

    def _on_mask_view_toggled(self, checked: bool):
        if bool(checked) != self._show_mask:
            self._begin_document_change()
        self._show_mask = bool(checked)
        self._update_mask_debug_label()
        self._update_preview_label()

    def _on_mask_edit_toggled(self, checked: bool):
        self._mask_edit_enabled = bool(checked)
        if self._mask_edit_enabled and self._mask_history_index < 0:
            self._push_mask_history()
        if self._mask_edit_enabled:
            if self._crop_edit_enabled and hasattr(self, "crop_edit_btn"):
                self.crop_edit_btn.setChecked(False)  # mutually exclusive edit modes
            if self._wb_pick_enabled and hasattr(self, "wb_pick_btn"):
                self.wb_pick_btn.setChecked(False)
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
        self._update_preview_label()
        self._push_document_history()

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
            self._set_section_expanded("White Balance", True)
        self.image_label.set_wb_pick_state(self._wb_pick_enabled, self._on_wb_picked)
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

    def _refresh_mask_controls(self):
        editable = self._active_layer in MASK_ORDER and self.preview_masks is not None
        self.mask_view_btn.setEnabled(editable)
        self.mask_edit_btn.setEnabled(editable)
        self.mask_debug_combo.setEnabled(editable)
        self.guides_btn.setEnabled(self.preview_guides is not None)
        self.mask_mode_combo.setEnabled(editable)
        self.brush_slider.setEnabled(editable)
        self.hardness_slider.setEnabled(editable)
        self.reset_mask_btn.setEnabled(editable)
        self.feather_mask_btn.setEnabled(editable)
        self._sync_mask_adjustment_controls()
        if not editable and self._mask_edit_enabled:
            self.mask_edit_btn.blockSignals(True)
            self.mask_edit_btn.setChecked(False)
            self.mask_edit_btn.blockSignals(False)
            self._mask_edit_enabled = False
        self.image_label.set_edit_state(editable and self._mask_edit_enabled, self._paint_active_mask_at, self._begin_mask_stroke)
        source_size = None
        if self.preview_array is not None:
            source_size = (self.preview_array.shape[1], self.preview_array.shape[0])
        self.image_label.set_brush_preview(
            self._mask_brush_size,
            source_image_size=source_size,
            hardness=self._mask_brush_hardness,
        )
        self._update_mask_debug_label()
        self._update_mask_history_buttons()

    def _on_compare_mode_changed(self, mode: str):
        if str(mode or "off") != self._compare_mode:
            self._begin_document_change()
        self._compare_mode = str(mode or "off")
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.split_slider.setEnabled(self._compare_mode == "split")
        self._update_preview_label()

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
        super().keyPressEvent(event)

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
            QMessageBox.critical(self, "Load Error", str(ex))

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
            QMessageBox.critical(self, "Project Load Error", str(ex))

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
            QMessageBox.critical(self, "Project Save Error", str(ex))

    def _load_image_path(self, path: str, project_state=None):
        self.file_path = path
        self.statusBar().showMessage(f"Loading {os.path.basename(path)} ...")
        self.image_label.reset_view()
        self._populate_filmstrip()
        full = self._read_image_file(path)
        preview, scale = self._build_preview_proxy(full)
        interactive_preview, interactive_ratio = self._build_interactive_preview_proxy(preview)
        self.full_array = full
        self.preview_array = preview
        self.preview_scale = scale
        self._interactive_preview_array = interactive_preview
        self._interactive_preview_ratio = interactive_ratio
        self._face_profiles = {}
        self._mask_adjustments = default_mask_adjustments()
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self.preview_guides = None
        self.full_guides = None
        if project_state is not None:
            self._apply_project_state(project_state)
        else:
            self._framing = default_framing()
            self._sync_framing_controls()
            self._sync_wb_controls()
            self._sync_tone_curve_controls()
            self._sync_hsl_controls()
            self._run_segmentation()
            self._clear_document_history()
            self._push_document_history()

    def _populate_filmstrip(self, folder: str = None):
        """Populate the filmstrip with imported images."""
        # Clear old filmstrip.
        for widget in self._filmstrip_items.values():
            widget.deleteLater()
        self._filmstrip_items.clear()
        self._filmstrip_images = list(self._imported_images)

        # Create thumbnails for imported images.
        for img_path in self._filmstrip_images:
            if not Path(img_path).is_file():
                continue
            thumb = FilmstripThumbnail(img_path, self)
            thumb.clicked.connect(self._on_filmstrip_image_clicked)
            self._filmstrip_layout.addWidget(thumb)
            self._filmstrip_items[img_path] = thumb

            # Use cached thumbnail if available, otherwise load.
            cached_thumb = self._get_cached_thumbnail(img_path)
            if cached_thumb is not None:
                thumb.set_pixmap(cached_thumb)
            else:
                # Check if thumbnail is in queue; if not, queue it.
                if img_path not in self._import_queue:
                    self._import_queue.append(img_path)

        self._filmstrip_layout.addStretch(1)
        self._update_filmstrip_active()

        # Start the import worker if not running.
        if self._import_queue and not self._import_worker_timer.isActive():
            self._import_worker_timer.start()

    def _import_folder(self):
        """Open import dialog and start importing images from selected folder."""
        dialog = ImportDialog(self, self.SUPPORTED_IMAGE_EXTS)
        if dialog.exec() != QDialog.Accepted:
            return

        folder = dialog.selected_folder()
        if not folder:
            return

        # Scan folder for images.
        folder_path = Path(folder)
        supported = self.SUPPORTED_IMAGE_EXTS
        image_files = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        )
        new_images = [str(p) for p in image_files if str(p) not in self._imported_images]

        if not new_images:
            self.statusBar().showMessage(f"No new images to import from {folder}")
            return

        # Add to imported images and queue for thumbnail generation.
        self._imported_images.extend(new_images)
        self._import_queue.extend(new_images)

        # Refresh filmstrip.
        self._populate_filmstrip()

        # Start import worker.
        if not self._import_worker_timer.isActive():
            self._import_worker_timer.start()

        self.statusBar().showMessage(f"Importing {len(new_images)} images from {Path(folder).name}...")

    def _get_cached_thumbnail(self, img_path: str):
        """Get cached thumbnail if available, None otherwise."""
        # For now, just return None - we'll implement caching if needed.
        return None

    def _process_import_queue(self):
        """Process next image in import queue (called by timer every 500ms)."""
        if not self._import_queue:
            self._import_worker_timer.stop()
            if self._imported_images:
                self.statusBar().showMessage(f"Imported {len(self._imported_images)} images")
            return

        img_path = self._import_queue.pop(0)
        if img_path not in self._filmstrip_items:
            # Image not in filmstrip yet (shouldn't happen, but handle gracefully).
            return

        thumb_widget = self._filmstrip_items[img_path]

        # Load thumbnail in background thread.
        def load_and_set():
            pixmap = self._load_thumbnail_blocking(img_path)
            if pixmap is not None and img_path in self._filmstrip_items:
                self._filmstrip_items[img_path].set_pixmap(pixmap)

        worker = lambda: load_and_set()
        import threading
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # Update status bar with progress.
        remaining = len(self._import_queue)
        total = len(self._imported_images)
        processed = total - remaining
        self.statusBar().showMessage(f"Importing: {processed}/{total} thumbnails...")

    def _load_thumbnail_blocking(self, img_path: str):
        """Load a single thumbnail (blocking, meant for worker thread)."""
        try:
            ext = Path(img_path).suffix.lower()
            if ext in {".cr2", ".nef", ".arw", ".dng", ".raw"}:
                if not HAS_RAWPY:
                    return None
                with rawpy.imread(img_path) as raw:
                    rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
                img_array = rgb16.astype(np.float32) / 65535.0
            else:
                pil = Image.open(img_path).convert("RGB")
                img_array = np.asarray(pil, dtype=np.float32) / 255.0

            # Scale to fit thumbnail (max 200x200 for speed).
            h, w = img_array.shape[:2]
            max_dim = 200
            if w > max_dim or h > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert to QPixmap.
            img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
            h, w = img_uint8.shape[:2]
            if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
                qimg = QImage(img_uint8.data, w, h, 3 * w, QImage.Format_RGB888)
            else:
                # Grayscale, convert to RGB.
                img_uint8_rgb = np.stack([img_uint8] * 3, axis=-1)
                qimg = QImage(img_uint8_rgb.data, w, h, 3 * w, QImage.Format_RGB888)

            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def _load_thumbnail_async(self, img_path: str, thumb_widget: FilmstripThumbnail):
        """Load a thumbnail image asynchronously (worker thread)."""
        def load_thumb():
            try:
                ext = Path(img_path).suffix.lower()
                if ext in {".cr2", ".nef", ".arw", ".dng", ".raw"}:
                    if not HAS_RAWPY:
                        return None
                    with rawpy.imread(img_path) as raw:
                        rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
                    img_array = rgb16.astype(np.float32) / 65535.0
                else:
                    pil = Image.open(img_path).convert("RGB")
                    img_array = np.asarray(pil, dtype=np.float32) / 255.0

                # Scale to fit thumbnail (max 200x200 for speed).
                h, w = img_array.shape[:2]
                max_dim = 200
                if w > max_dim or h > max_dim:
                    scale = max_dim / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Convert to QPixmap.
                img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
                h, w = img_uint8.shape[:2]
                if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
                    qimg = QImage(img_uint8.data, w, h, 3 * w, QImage.Format_RGB888)
                else:
                    # Grayscale, convert to RGB.
                    img_uint8_rgb = np.stack([img_uint8] * 3, axis=-1)
                    qimg = QImage(img_uint8_rgb.data, w, h, 3 * w, QImage.Format_RGB888)

                return QPixmap.fromImage(qimg)
            except Exception:
                return None

        def set_thumb(pixmap):
            if pixmap is not None and img_path in self._filmstrip_items:
                self._filmstrip_items[img_path].set_pixmap(pixmap)

        # Run in thread pool to avoid blocking UI.
        worker = lambda: set_thumb(load_thumb())
        import threading
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _update_filmstrip_active(self):
        """Highlight the current image in the filmstrip."""
        for path, thumb in self._filmstrip_items.items():
            thumb.set_active(path == self.file_path)
        self._current_filmstrip_path = self.file_path

    def _on_filmstrip_image_clicked(self, path: str):
        """Load the clicked image from the filmstrip."""
        if path == self.file_path:
            return
        if not Path(path).is_file():
            self.statusBar().showMessage(f"File not found: {path}")
            return
        self.open_image(path)

    def _read_image_file(self, path: str):
        ext = Path(path).suffix.lower()
        self._source_metadata = {}
        if ext in {".cr2", ".nef", ".arw", ".dng", ".raw"}:
            if not HAS_RAWPY:
                raise RuntimeError("rawpy is not installed. Install base dependencies first.")
            with rawpy.imread(path) as raw:
                rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
            return rgb16.astype(np.float32) / 65535.0

        pil = Image.open(path)
        # Capture embeddable metadata before exif_transpose/convert drop it.
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
        self._source_metadata = metadata
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        return np.asarray(pil, dtype=np.float32) / 255.0

    def batch_export(self):
        selected_entry = self._selected_browser_entry()
        last_options = self._last_batch_options()
        profiles = self._batch_profiles()
        selected_profile = str(self._browser_state_payload().get("last_batch_profile", ""))
        dialog = BatchExportDialog(
            self,
            preset_path=last_options.get("preset_path") or (str(selected_entry["path"]) if selected_entry is not None else ""),
            input_dir=last_options.get("input_dir", ""),
            output_dir=last_options.get("output_dir", ""),
            suffix=last_options.get("suffix", "_enhanced"),
            output_format=last_options.get("format", "jpeg"),
            skip_completed=bool(last_options.get("skip_completed", True)),
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
            }
        )
        preset_path = options["preset_path"]
        input_dir = options["input_dir"]
        output_dir = options["output_dir"]
        suffix = options["suffix"]
        output_format = options["format"]
        skip_completed = bool(options["skip_completed"])

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
            QMessageBox.critical(self, "Preset Load Error", str(ex))
            return

        image_paths = self._supported_image_paths(input_dir)
        if not image_paths:
            QMessageBox.information(self, "Batch Export", "No supported images found in the selected input folder.")
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
            QMessageBox.critical(self, "Preset Load Error", str(ex))
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
        start_dir = ""
        if self.file_path:
            start_dir = os.path.dirname(self.file_path)
        dialog = BatchJobsDialog(self, output_dir=start_dir, load_jobs_callback=self._load_batch_jobs)
        dialog.exec()

    def _run_segmentation(self):
        if self.preview_array is None:
            return

        t0 = time.perf_counter()
        self._detected_faces = self.segmenter.list_faces(self.preview_array)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        self.face_combo.blockSignals(True)
        self.face_combo.clear()
        if not self._detected_faces:
            self.face_combo.addItem("Auto")
            self.face_combo.setCurrentIndex(0)
            self._active_face_index = 0
        else:
            for idx, (x, y, w, h) in enumerate(self._detected_faces):
                self.face_combo.addItem(f"Face {idx + 1} ({w}x{h} @ {x},{y})")
            self._active_face_index = min(self._active_face_index, len(self._detected_faces) - 1)
            self.face_combo.setCurrentIndex(self._active_face_index)
        self.face_combo.blockSignals(False)

        face_index = self._active_face_index if self._detected_faces else 0
        t1 = time.perf_counter()
        self.preview_masks, self.preview_guides = self.segmenter.segment_with_guides(self.preview_array, face_index=face_index)
        segment_ms = (time.perf_counter() - t1) * 1000.0
        auto_preview_masks = self._copy_masks(self.preview_masks)
        auto_full_masks = self._scale_masks(self.preview_masks, self.full_array.shape[:2])
        scale = 1.0 / max(self.preview_scale, 1e-6)
        auto_full_guides = _scale_expression_guides(self.preview_guides, scale, scale)
        self._auto_preview_masks = auto_preview_masks
        self._auto_full_masks = self._copy_masks(auto_full_masks)
        self.full_masks = self._copy_masks(auto_full_masks)
        self.full_guides = self._copy_guides(auto_full_guides)
        restored = self._restore_face_profile(self._active_face_index)
        if not restored:
            self.preview_masks = self._copy_masks(auto_preview_masks)
            self.full_masks = self._copy_masks(auto_full_masks)
            self.preview_guides = self._copy_guides(self.preview_guides)
            self.full_guides = self._copy_guides(auto_full_guides)
            self._clear_mask_history()
            self._push_mask_history()
        self._perf_stats["detect_ms"] = detect_ms
        self._perf_stats["segment_ms"] = segment_ms
        status = (
            f"Loaded {os.path.basename(self.file_path)} | "
            f"seg={self.segmenter.backend_label} | "
            f"{self.segmenter.detector_backend_label} | "
            f"{getattr(self.segmenter, 'subject_backend_label', 'subjects=heuristic')} | "
            f"{getattr(self.segmenter, 'facial_hair_backend_label', 'f_hair=fallback')} | "
            f"faces={len(self._detected_faces)}"
        )
        self.statusBar().showMessage(status)
        self.info_label.setText(status)
        self._update_perf_label()
        self._refresh_mask_controls()
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
        )
        task.signals.finished.connect(self._on_preview_render_finished)
        task.signals.failed.connect(self._on_preview_render_failed)
        self._render_pool.start(task)

    def _on_preview_render_finished(self, job_id: int, result, elapsed_ms: float):
        self._render_in_flight = False
        if int(job_id) != self._active_render_job_id:
            if self._render_pending:
                self._update_perf_label()
                self._schedule_render()
            return
        self._last_completed_render_job_id = int(job_id)
        self._perf_stats["render_ms"] = float(elapsed_ms)
        self.preview_image = result
        self._update_perf_label()
        self._update_preview_label()
        self._update_histogram()
        if self._render_pending:
            self._schedule_render()

    def _on_preview_render_failed(self, job_id: int, message: str):
        self._render_in_flight = False
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
        if self._mask_edit_enabled or self._wb_pick_enabled:
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
        data = image.tobytes("raw", "RGB")
        qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format_RGB888).copy()
        self.image_label.set_preview_pixmap(QPixmap.fromImage(qimage))

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
        if masks is None:
            return None
        return {key: value.copy() for key, value in masks.items()}

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

    def _current_preview_render_inputs(self):
        if self.preview_array is None:
            return None, None, None
        if self._slider_drag_active <= 0 or self._interactive_preview_array is None:
            return (
                self.preview_array.copy(),
                self._adjusted_masks(self.preview_masks),
                self._copy_guides(self.preview_guides),
            )
        ratio = float(self._interactive_preview_ratio)
        masks = self._scale_masks(self.preview_masks, self._interactive_preview_array.shape[:2])
        masks = self._adjusted_masks(masks)
        guides = _scale_expression_guides(self.preview_guides, ratio, ratio)
        return self._interactive_preview_array.copy(), masks, self._copy_guides(guides)

    def _capture_mask_snapshot(self):
        return self._copy_masks(self.preview_masks)

    def _restore_mask_snapshot(self, snapshot):
        if snapshot is None:
            return
        self.preview_masks = self._copy_masks(snapshot)
        if self.preview_masks is not None:
            for layer in list(self.preview_masks.keys()):
                self._sync_full_mask_from_preview(layer)
        self._update_mask_debug_label()

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
        can_edit = self._active_layer in MASK_ORDER and self.preview_masks is not None
        can_undo = can_edit and self._mask_history_index > 0
        can_redo = can_edit and self._mask_history_index >= 0 and self._mask_history_index < len(self._mask_history) - 1
        if hasattr(self, "undo_mask_btn"):
            self.undo_mask_btn.setEnabled(can_undo)
        if hasattr(self, "redo_mask_btn"):
            self.redo_mask_btn.setEnabled(can_redo)

    def _begin_mask_stroke(self):
        if not self._mask_edit_enabled or self.preview_masks is None or self._active_layer not in self.preview_masks:
            return
        self._push_mask_history()

    def _undo_mask_edit(self):
        if self._mask_history_index <= 0:
            return
        self._mask_history_index -= 1
        self._restore_mask_snapshot(self._mask_history[self._mask_history_index])
        self._update_mask_history_buttons()
        self._schedule_render()

    def _redo_mask_edit(self):
        if self._mask_history_index < 0 or self._mask_history_index >= len(self._mask_history) - 1:
            return
        self._mask_history_index += 1
        self._restore_mask_snapshot(self._mask_history[self._mask_history_index])
        self._update_mask_history_buttons()
        self._schedule_render()

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
        self._update_mask_history_buttons()
        self._update_mask_debug_label()
        self._schedule_render()

    def _sync_full_mask_from_preview(self, layer: str):
        if self.preview_masks is None or self.full_array is None or layer not in self.preview_masks:
            return
        mask = self.preview_masks[layer]
        pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
        pil = pil.resize((self.full_array.shape[1], self.full_array.shape[0]), Image.BILINEAR)
        if self.full_masks is None:
            self.full_masks = {}
        self.full_masks[layer] = np.asarray(pil, dtype=np.float32) / 255.0

    def _reset_active_mask(self):
        if self._active_layer not in MASK_ORDER or self._auto_preview_masks is None:
            return
        self._begin_document_change()
        self._push_mask_history()
        if self._active_layer in self._auto_preview_masks:
            self.preview_masks[self._active_layer] = self._auto_preview_masks[self._active_layer].copy()
        if self._auto_full_masks is not None and self._active_layer in self._auto_full_masks:
            if self.full_masks is None:
                self.full_masks = {}
            self.full_masks[self._active_layer] = self._auto_full_masks[self._active_layer].copy()
        self._update_mask_history_buttons()
        self._update_mask_debug_label()
        self._schedule_render()

    def _feather_active_mask(self):
        if self._active_layer not in MASK_ORDER or self.preview_masks is None:
            return
        if self._active_layer not in self.preview_masks:
            return
        self._begin_document_change()
        self._push_mask_history()
        sigma = max(1.0, float(self._mask_brush_size) / 6.0)
        feathered = smooth_mask(
            self.preview_masks[self._active_layer],
            sigma=sigma,
            acceleration=self._runtime_settings.get("acceleration_mode", "auto"),
        )
        self.preview_masks[self._active_layer] = np.clip(feathered.astype(np.float32), 0.0, 1.0)
        self._sync_full_mask_from_preview(self._active_layer)
        self._update_mask_history_buttons()
        self._update_mask_debug_label()
        self._schedule_render()

    def export_image(self):
        if self.full_array is None:
            QMessageBox.information(self, "Export", "Open an image first.")
            return
        src = Path(self.file_path) if self.file_path else None
        default_dir = str(src.parent) if src else os.path.expanduser("~")
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
        self._last_export_format = opts["format"]
        self._last_export_quality = opts["quality"]
        try:
            result = process_all_layers(
                self.full_array,
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
            save_kwargs = self._export_metadata_kwargs(opts) if opts["keep_metadata"] else {}
            if opts["format"] == "jpeg":
                result.save(out_path, "JPEG", quality=opts["quality"], subsampling=0, **save_kwargs)
            elif opts["format"] == "png":
                result.save(out_path, "PNG", **save_kwargs)
            else:
                result.save(out_path, "TIFF", **save_kwargs)
            self.statusBar().showMessage(f"Exported -> {out_path}")
            QMessageBox.information(self, "Exported", f"Saved:\n{out_path}")
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))

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
        self._clear_mask_history()
        if self.preview_masks is not None:
            self._push_mask_history()
        self._schedule_render()


def run_qt_app():
    app = QApplication.instance() or QApplication([])
    window = PortraitEnhancerQtWindow()
    window.show()
    return app.exec()
