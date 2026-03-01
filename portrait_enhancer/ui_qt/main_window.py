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

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QSize, QEvent, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap
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
    QToolButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from portrait_enhancer.config import ALL_LAYERS, LAYER_COLORS, LAYER_NAMES, MASK_ORDER
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
    def __init__(self, report_text: str, parent=None, title: str = "System Check"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        label = QLabel("Runtime readiness and model status")
        layout.addWidget(label)

        details = QTextEdit(self)
        details.setReadOnly(True)
        details.setPlainText(report_text)
        layout.addWidget(details, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.reject)
        layout.addWidget(buttons)


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

    def set_preview_pixmap(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self._apply_scaled_pixmap()

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def leaveEvent(self, event):
        self._hover_rel = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._edit_enabled or self._hover_rel is None or self._scaled_pixmap is None:
            return
        if not self._source_image_size or self._source_image_size[0] <= 0 or self._source_image_size[1] <= 0:
            return

        rect = self.contentsRect()
        x0 = rect.x() + (rect.width() - self._scaled_pixmap.width()) / 2.0
        y0 = rect.y() + (rect.height() - self._scaled_pixmap.height()) / 2.0
        cx = x0 + self._hover_rel[0] * self._scaled_pixmap.width()
        cy = y0 + self._hover_rel[1] * self._scaled_pixmap.height()
        scale_x = self._scaled_pixmap.width() / float(self._source_image_size[0])
        scale_y = self._scaled_pixmap.height() / float(self._source_image_size[1])
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
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
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
            self._dragging_split = False
            self._dragging_paint = False
        super().mouseReleaseEvent(event)

    def _apply_scaled_pixmap(self):
        if self._pixmap is None:
            self._scaled_pixmap = None
            self.clear()
            self.setText("Open an image to begin")
            return
        self._scaled_pixmap = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self._scaled_pixmap)
        self.update()

    def _update_hover_rel(self, widget_x: float, widget_y: float):
        pix = self.pixmap()
        if pix is None or pix.width() <= 0 or pix.height() <= 0:
            self._hover_rel = None
            self.update()
            return
        x0 = (self.width() - pix.width()) / 2.0
        y0 = (self.height() - pix.height()) / 2.0
        rel_x = (float(widget_x) - x0) / float(pix.width())
        rel_y = (float(widget_y) - y0) / float(pix.height())
        if 0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0:
            self._hover_rel = (rel_x, rel_y)
        else:
            self._hover_rel = None
        self.update()

    def _emit_split_position(self, widget_x: float):
        if self._split_callback is None:
            return
        pix = self.pixmap()
        if pix is None or pix.width() <= 0:
            return
        x0 = (self.width() - pix.width()) / 2.0
        rel_x = (float(widget_x) - x0) / float(pix.width())
        self._split_callback(max(0.0, min(1.0, rel_x)))

    def _emit_paint_point(self, widget_x: float, widget_y: float):
        if self._paint_callback is None:
            return
        pix = self.pixmap()
        if pix is None or pix.width() <= 0 or pix.height() <= 0:
            return
        x0 = (self.width() - pix.width()) / 2.0
        y0 = (self.height() - pix.height()) / 2.0
        rel_x = (float(widget_x) - x0) / float(pix.width())
        rel_y = (float(widget_y) - y0) / float(pix.height())
        if 0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0:
            self._paint_callback(rel_x, rel_y)


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
        self._active_layer = "global"
        self._show_mask = False
        self._mask_debug_mode = "tint"
        self._show_expression_guides = False
        self._mask_edit_enabled = False
        self._mask_paint_mode = "paint"
        self._mask_brush_size = 24
        self._mask_brush_hardness = 100
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

    def _build_ui(self):
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.open_image)
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

        toolbar = self.addToolBar("Main")
        toolbar.addAction(open_action)
        toolbar.addAction(undo_action)
        toolbar.addAction(redo_action)
        toolbar.addAction(open_project_action)
        toolbar.addAction(save_project_action)
        toolbar.addAction(recipes_action)
        toolbar.addAction(check_action)
        toolbar.addAction(open_preset_action)
        toolbar.addAction(save_preset_action)
        toolbar.addAction(batch_export_action)
        toolbar.addAction(batch_jobs_action)
        toolbar.addAction(retry_failed_action)
        toolbar.addAction(export_action)
        toolbar.addAction(reset_action)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left_panel = QWidget(self)
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        workspace_content = QWidget(self)
        workspace_layout = QVBoxLayout(workspace_content)
        workspace_layout.setContentsMargins(8, 4, 8, 4)
        workspace_layout.setSpacing(8)

        session_card = QFrame(self)
        session_card.setFrameShape(QFrame.StyledPanel)
        session_layout = QVBoxLayout(session_card)
        session_layout.setContentsMargins(8, 8, 8, 8)
        session_layout.setSpacing(8)
        session_layout.addWidget(QLabel("Session"))

        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.addWidget(QLabel("Face Target"))
        self.face_combo = QComboBox()
        self.face_combo.addItem("Auto")
        self.face_combo.currentIndexChanged.connect(self._on_face_changed)
        face_row.addWidget(self.face_combo, 1)
        session_layout.addLayout(face_row)

        quick_actions = QGridLayout()
        quick_actions.setContentsMargins(0, 0, 0, 0)
        quick_actions.setHorizontalSpacing(6)
        quick_actions.setVerticalSpacing(6)
        recipes_btn = QPushButton("Recipes")
        recipes_btn.clicked.connect(self.open_recipe_dialog)
        quick_actions.addWidget(recipes_btn, 0, 0)
        check_btn = QPushButton("System Check")
        check_btn.clicked.connect(self.show_system_check)
        quick_actions.addWidget(check_btn, 0, 1)
        batch_btn = QPushButton("Batch Export")
        batch_btn.clicked.connect(self.batch_export)
        quick_actions.addWidget(batch_btn, 1, 0)
        jobs_btn = QPushButton("Batch Jobs")
        jobs_btn.clicked.connect(self.view_batch_jobs)
        quick_actions.addWidget(jobs_btn, 1, 1)
        session_layout.addLayout(quick_actions)
        workspace_layout.addWidget(session_card)

        basics_card = QFrame(self)
        basics_card.setFrameShape(QFrame.StyledPanel)
        basics_layout = QVBoxLayout(basics_card)
        basics_layout.setContentsMargins(8, 8, 8, 8)
        basics_layout.setSpacing(6)
        basics_layout.addWidget(QLabel("Portrait Basics"))
        quick_layer_row = QHBoxLayout()
        quick_layer_row.setContentsMargins(0, 0, 0, 0)
        for layer in self.BASIC_LAYERS:
            button = QToolButton(self)
            button.setText(layer.title())
            button.clicked.connect(lambda _checked=False, layer_name=layer: self._activate_layer(layer_name))
            quick_layer_row.addWidget(button)
        basics_layout.addLayout(quick_layer_row)

        org_row = QHBoxLayout()
        org_row.setContentsMargins(0, 0, 0, 0)
        self.essentials_only_check = QCheckBox("Essentials Only")
        self.essentials_only_check.setChecked(self._essentials_only)
        self.essentials_only_check.toggled.connect(self._on_essentials_only_toggled)
        org_row.addWidget(self.essentials_only_check)
        self.reset_layer_btn = QPushButton("Reset Layer")
        self.reset_layer_btn.clicked.connect(self._reset_active_layer)
        org_row.addWidget(self.reset_layer_btn)
        basics_layout.addLayout(org_row)
        workspace_layout.addWidget(basics_card)

        preview_card = QFrame(self)
        preview_card.setFrameShape(QFrame.StyledPanel)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        preview_layout.addWidget(QLabel("Preview"))
        compare_row = QHBoxLayout()
        compare_row.setContentsMargins(0, 0, 0, 0)
        compare_row.addWidget(QLabel("Compare"))
        self.compare_combo = QComboBox()
        self.compare_combo.addItems(["off", "before", "split", "side_by_side"])
        self.compare_combo.currentTextChanged.connect(self._on_compare_mode_changed)
        compare_row.addWidget(self.compare_combo, 1)
        preview_layout.addLayout(compare_row)

        split_row = QHBoxLayout()
        split_row.setContentsMargins(0, 0, 0, 0)
        split_row.addWidget(QLabel("Split"))
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(int(self._split_position * 100))
        self.split_slider.valueChanged.connect(self._on_split_slider_changed)
        split_row.addWidget(self.split_slider, 1)
        preview_layout.addLayout(split_row)

        self.compare_hint_label = QLabel("Hold Space: original")
        preview_layout.addWidget(self.compare_hint_label)
        workspace_layout.addWidget(preview_card)
        left_layout.addWidget(self._make_collapsible_section("Workspace", workspace_content, expanded=True))

        mask_content = QWidget(self)
        mask_layout = QVBoxLayout(mask_content)
        mask_layout.setContentsMargins(8, 4, 8, 4)
        mask_layout.setSpacing(8)
        mask_row = QHBoxLayout()
        self.mask_view_btn = QPushButton("Mask View")
        self.mask_view_btn.setCheckable(True)
        self.mask_view_btn.toggled.connect(self._on_mask_view_toggled)
        mask_row.addWidget(self.mask_view_btn)
        self.mask_edit_btn = QPushButton("Edit Mask")
        self.mask_edit_btn.setCheckable(True)
        self.mask_edit_btn.toggled.connect(self._on_mask_edit_toggled)
        mask_row.addWidget(self.mask_edit_btn)
        mask_layout.addLayout(mask_row)

        debug_row = QHBoxLayout()
        debug_row.addWidget(QLabel("Debug"))
        self.mask_debug_combo = QComboBox()
        self.mask_debug_combo.addItems(["tint", "heatmap", "isolated"])
        self.mask_debug_combo.currentTextChanged.connect(self._on_mask_debug_mode_changed)
        debug_row.addWidget(self.mask_debug_combo, 1)
        self.guides_btn = QPushButton("Guides")
        self.guides_btn.setCheckable(True)
        self.guides_btn.toggled.connect(self._on_guides_toggled)
        debug_row.addWidget(self.guides_btn)
        mask_layout.addLayout(debug_row)

        self.mask_debug_label = QLabel("Mask: --")
        self.mask_debug_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_debug_label)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(2, 80)
        self.brush_slider.setValue(self._mask_brush_size)
        self.brush_slider.valueChanged.connect(self._on_brush_size_changed)
        brush_row.addWidget(self.brush_slider, 1)
        self.brush_value_label = QLabel(str(self._mask_brush_size))
        brush_row.addWidget(self.brush_value_label)
        mask_layout.addLayout(brush_row)

        hardness_row = QHBoxLayout()
        hardness_row.addWidget(QLabel("Hardness"))
        self.hardness_slider = QSlider(Qt.Horizontal)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(self._mask_brush_hardness)
        self.hardness_slider.valueChanged.connect(self._on_brush_hardness_changed)
        hardness_row.addWidget(self.hardness_slider, 1)
        self.hardness_value_label = QLabel(f"{self._mask_brush_hardness}%")
        hardness_row.addWidget(self.hardness_value_label)
        mask_layout.addLayout(hardness_row)

        mask_mode_row = QHBoxLayout()
        mask_mode_row.addWidget(QLabel("Mode"))
        self.mask_mode_combo = QComboBox()
        self.mask_mode_combo.addItems(["paint", "erase"])
        self.mask_mode_combo.currentTextChanged.connect(self._on_mask_mode_changed)
        mask_mode_row.addWidget(self.mask_mode_combo, 1)
        self.reset_mask_btn = QPushButton("Reset Mask")
        self.reset_mask_btn.clicked.connect(self._reset_active_mask)
        mask_mode_row.addWidget(self.reset_mask_btn)
        self.feather_mask_btn = QPushButton("Feather")
        self.feather_mask_btn.clicked.connect(self._feather_active_mask)
        mask_mode_row.addWidget(self.feather_mask_btn)
        mask_layout.addLayout(mask_mode_row)

        history_row = QHBoxLayout()
        self.undo_mask_btn = QPushButton("Undo")
        self.undo_mask_btn.clicked.connect(self._undo_mask_edit)
        history_row.addWidget(self.undo_mask_btn)
        self.redo_mask_btn = QPushButton("Redo")
        self.redo_mask_btn.clicked.connect(self._redo_mask_edit)
        history_row.addWidget(self.redo_mask_btn)
        mask_layout.addLayout(history_row)
        left_layout.addWidget(self._make_collapsible_section("Mask Tools", mask_content, expanded=False))

        layers_content = QWidget(self)
        layers_layout = QVBoxLayout(layers_content)
        layers_layout.setContentsMargins(0, 4, 0, 4)
        layers_layout.setSpacing(6)
        self.layer_tabs = QTabWidget()
        for layer, sliders in ALL_LAYERS.items():
            self.layer_tabs.addTab(self._build_layer_tab(layer, sliders), LAYER_NAMES[layer])
        self.layer_tabs.currentChanged.connect(self._on_layer_changed)
        layers_layout.addWidget(self.layer_tabs)
        left_layout.addWidget(self._make_collapsible_section("Layers", layers_content, expanded=True))

        preset_content = QWidget(self)
        preset_layout = QVBoxLayout(preset_content)
        preset_layout.setContentsMargins(8, 4, 8, 4)
        preset_layout.setSpacing(8)
        self.preset_search = QLineEdit()
        self.preset_search.setPlaceholderText("Search presets")
        self.preset_search.textChanged.connect(self._refresh_preset_browser)
        self.preset_search.installEventFilter(self)
        preset_layout.addWidget(self.preset_search)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(QLabel("Category"))
        self.preset_category_combo = QComboBox()
        self.preset_category_combo.addItem("all")
        self.preset_category_combo.currentTextChanged.connect(self._on_preset_category_changed)
        self.preset_category_combo.installEventFilter(self)
        filter_row.addWidget(self.preset_category_combo, 1)
        preset_layout.addLayout(filter_row)

        recent_label = QLabel("Recent Presets")
        preset_layout.addWidget(recent_label)
        self.recent_preset_list = QListWidget()
        self.recent_preset_list.setMaximumHeight(92)
        self.recent_preset_list.setMouseTracking(True)
        self.recent_preset_list.installEventFilter(self)
        self.recent_preset_list.itemDoubleClicked.connect(lambda _item: self._apply_selected_recent_preset())
        self.recent_preset_list.currentRowChanged.connect(lambda _row: self._update_selected_preset_meta())
        self.recent_preset_list.itemEntered.connect(lambda _item: self._update_selected_preset_meta())
        preset_layout.addWidget(self.recent_preset_list)

        self.preset_list = QListWidget()
        self.preset_list.setMouseTracking(True)
        self.preset_list.installEventFilter(self)
        self.preset_list.setViewMode(QListView.IconMode)
        self.preset_list.setResizeMode(QListView.Adjust)
        self.preset_list.setMovement(QListView.Static)
        self.preset_list.setIconSize(QSize(140, 96))
        self.preset_list.setGridSize(QSize(164, 138))
        self.preset_list.setWordWrap(True)
        self.preset_list.setSpacing(8)
        self.preset_list.itemDoubleClicked.connect(lambda _item: self._apply_selected_browser_preset())
        self.preset_list.currentRowChanged.connect(lambda _row: self._update_selected_preset_meta())
        self.preset_list.itemEntered.connect(lambda _item: self._update_selected_preset_meta())
        preset_layout.addWidget(self.preset_list)

        self.preset_preview_label = QLabel("No preset preview")
        self.preset_preview_label.setAlignment(Qt.AlignCenter)
        self.preset_preview_label.setMinimumHeight(120)
        self.preset_preview_label.setStyleSheet("border: 1px solid #2a2a2f; background: #141418;")
        preset_layout.addWidget(self.preset_preview_label)

        self.preset_meta_label = QLabel("No preset selected")
        self.preset_meta_label.setWordWrap(True)
        preset_layout.addWidget(self.preset_meta_label)

        preset_btn_row = QHBoxLayout()
        self.apply_browser_preset_btn = QPushButton("Apply")
        self.apply_browser_preset_btn.clicked.connect(self._apply_selected_browser_preset)
        preset_btn_row.addWidget(self.apply_browser_preset_btn)
        self.save_browser_preset_btn = QPushButton("Save Here")
        self.save_browser_preset_btn.clicked.connect(self._save_preset_to_library)
        preset_btn_row.addWidget(self.save_browser_preset_btn)
        self.rename_browser_preset_btn = QPushButton("Rename")
        self.rename_browser_preset_btn.clicked.connect(self._rename_selected_browser_preset)
        preset_btn_row.addWidget(self.rename_browser_preset_btn)
        self.delete_browser_preset_btn = QPushButton("Delete")
        self.delete_browser_preset_btn.clicked.connect(self._delete_selected_browser_preset)
        preset_btn_row.addWidget(self.delete_browser_preset_btn)
        self.refresh_browser_preset_btn = QPushButton("Refresh")
        self.refresh_browser_preset_btn.clicked.connect(self._refresh_preset_browser)
        preset_btn_row.addWidget(self.refresh_browser_preset_btn)
        preset_layout.addLayout(preset_btn_row)
        left_layout.addWidget(self._make_collapsible_section("Preset Browser", preset_content, expanded=False))
        left_layout.addStretch(1)

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        root.addWidget(left_scroll, 0)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.info_label = QLabel("No image loaded")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_layout.addWidget(self.info_label)
        self.perf_label = QLabel("Perf: detect=-- ms | segment=-- ms | render=-- ms | preview=idle | expr=off | refine=off")
        self.perf_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right_layout.addWidget(self.perf_label)

        self.image_label = ImagePreviewLabel()
        self.image_label.set_compare_state(self._compare_mode, self._set_split_position)
        self.image_label.set_edit_state(False, self._paint_active_mask_at, self._begin_mask_stroke)
        right_layout.addWidget(self.image_label, 1)
        root.addWidget(right_panel, 1)
        self._refresh_mask_controls()
        self._refresh_preset_browser()

    def _build_layer_tab(self, layer: str, sliders):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
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

        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _make_collapsible_section(self, title: str, content: QWidget, expanded: bool = True):
        expanded = bool(self._section_state.get(title, expanded))
        wrapper = QWidget(self)
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QToolButton(self)
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(bool(expanded))
        header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        header.clicked.connect(lambda checked, widget=content, button=header: self._toggle_section(widget, button, checked))
        outer.addWidget(header)
        outer.addWidget(content)
        content.setVisible(bool(expanded))
        self._section_buttons[title] = header
        self._section_contents[title] = content
        self._section_wrappers[title] = wrapper
        self._section_state[title] = bool(expanded)
        return wrapper

    def _toggle_section(self, content: QWidget, button: QToolButton, expanded: bool):
        content.setVisible(bool(expanded))
        button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        title = button.text()
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
        button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
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

    def _model_readiness_summary(self):
        lines = []
        root = Path(__file__).resolve().parents[2]
        models_dir = root / "models"
        lines.append(f"Workspace: {root}")
        lines.append(f"Models Dir: {models_dir}")
        lines.append("")

        parser = getattr(self.segmenter, "_model", None)
        detector = getattr(getattr(self.segmenter, "_heuristic", None), "_face_detector", None)
        subject_segmenter = getattr(self.segmenter, "_subjects", None)
        facial_hair_segmenter = getattr(self.segmenter, "_facial_hair", None)
        refiner = get_face_refiner()
        refiner._ensure_session()

        def exists_line(label, path):
            if not path:
                return f"{label}: missing"
            return f"{label}: {'found' if Path(path).exists() else 'missing'} -> {path}"

        parser_path = parser._resolve_model_path(getattr(parser, "_explicit_model_path", None)) if parser is not None else None
        detector_path = detector._resolve_model_path(None) if detector is not None else None
        subject_path = subject_segmenter._resolve_model_path(None) if subject_segmenter is not None else None
        facial_hair_path = facial_hair_segmenter._resolve_model_path(None) if facial_hair_segmenter is not None else None
        refiner_path = refiner._resolve_model_path() if refiner is not None else None

        lines.append(f"Segmentation: {self.segmenter.backend_label}")
        lines.append(f"Detector: {self.segmenter.detector_backend_label}")
        lines.append(f"Subjects: {getattr(self.segmenter, 'subject_backend_label', 'subjects=heuristic')}")
        lines.append(f"Facial Hair: {getattr(self.segmenter, 'facial_hair_backend_label', 'f_hair=fallback')}")
        lines.append(f"Refiner: {refiner.backend_label if refiner.available else 'codeformer-unavailable'}")
        lines.append("")
        lines.append(exists_line("Face Parsing ONNX", parser_path))
        lines.append(exists_line("YuNet Detector", detector_path))
        lines.append(exists_line("Subject Segmenter", subject_path))
        lines.append(exists_line("Facial Hair ONNX", facial_hair_path))
        lines.append(exists_line("CodeFormer ONNX", refiner_path))
        lines.append(exists_line("Landmarker Task", os.getenv("PORTRAIT_FACE_LANDMARKER_TASK") or (models_dir / "face_landmarker.task")))
        lines.append("")

        issues = []
        parser_reason = getattr(parser, "reason_unavailable", "") if parser is not None else ""
        if parser_reason:
            issues.append(f"Segmentation: {parser_reason}")
        subject_reason = getattr(subject_segmenter, "reason_unavailable", "") if subject_segmenter is not None else ""
        if subject_reason:
            issues.append(f"Subjects: {subject_reason}")
        fh_reason = getattr(facial_hair_segmenter, "reason_unavailable", "") if facial_hair_segmenter is not None else ""
        if fh_reason:
            issues.append(f"Facial hair: {fh_reason}")
        refiner_reason = getattr(refiner, "reason_unavailable", "") if refiner is not None else ""
        if refiner_reason:
            issues.append(f"Refiner: {refiner_reason}")
        detector_reason = getattr(detector, "reason_unavailable", "") if detector is not None else ""
        if detector_reason:
            issues.append(f"Detector: {detector_reason}")

        if issues:
            lines.append("Issues:")
            lines.extend(f"- {item}" for item in issues)
        else:
            lines.append("Issues:")
            lines.append("- none")
        return "\n".join(lines), bool(issues)

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
                "preview_masks": self._copy_masks(profile.get("preview_masks")),
                "full_masks": self._copy_masks(profile.get("full_masks")),
                "auto_preview_masks": self._copy_masks(profile.get("auto_preview_masks")),
                "auto_full_masks": self._copy_masks(profile.get("auto_full_masks")),
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
            "show_mask": snapshot.get("show_mask", False),
            "mask_debug_mode": snapshot.get("mask_debug_mode", "tint"),
            "show_expression_guides": snapshot.get("show_expression_guides", False),
            "active_layer": snapshot.get("active_layer", "global"),
            "essentials_only": snapshot.get("essentials_only", True),
            "runtime_settings": snapshot.get("runtime_settings", {}),
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
            "show_mask": bool(self._show_mask),
            "mask_debug_mode": self._mask_debug_mode,
            "show_expression_guides": bool(self._show_expression_guides),
            "active_layer": self._active_layer,
            "essentials_only": bool(self._essentials_only),
            "runtime_settings": dict(self._runtime_settings),
        }

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

            self._face_profiles = self._copy_face_profiles(snapshot.get("face_profiles", {}))
            if self._detected_faces:
                self._active_face_index = int(np.clip(int(snapshot.get("active_face_index", 0)), 0, len(self._detected_faces) - 1))
            else:
                self._active_face_index = 0

            self._compare_mode = str(snapshot.get("compare_mode", "off") or "off")
            self._split_position = max(0.0, min(1.0, float(snapshot.get("split_position", 0.5))))
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
            "preview_masks": self._copy_masks(self.preview_masks),
            "full_masks": self._copy_masks(self.full_masks),
            "auto_preview_masks": self._copy_masks(self._auto_preview_masks),
            "auto_full_masks": self._copy_masks(self._auto_full_masks),
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
            "show_mask": bool(self._show_mask),
            "mask_debug_mode": self._mask_debug_mode,
            "show_expression_guides": bool(self._show_expression_guides),
            "active_layer": self._active_layer,
            "essentials_only": bool(self._essentials_only),
            "runtime_settings": dict(self._runtime_settings),
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

        self._detected_faces = [tuple(int(v) for v in face) for face in project.get("detected_faces", [])]
        self._active_face_index = int(project.get("active_face_index", 0))
        if self._detected_faces:
            self._active_face_index = int(np.clip(self._active_face_index, 0, len(self._detected_faces) - 1))
        else:
            self._active_face_index = 0
        self._face_profiles = self._deserialize_face_profiles(project.get("face_profiles", {}))

        self._compare_mode = str(project.get("compare_mode", "off") or "off")
        self._split_position = max(0.0, min(1.0, float(project.get("split_position", 0.5))))
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
        self.preset_meta_label.setText(
            f"Name: {meta.get('name', entry['path'].stem)}\n"
            f"Category: {meta.get('category', 'general')}\n"
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
        report, _has_issues = self._model_readiness_summary()
        ReadinessDialog(report, self, title="System Check").exec()

    def _maybe_show_startup_readiness(self):
        state = self._browser_state_payload()
        report, has_issues = self._model_readiness_summary()
        if not has_issues and state.get("readiness_seen"):
            return
        ReadinessDialog(report, self, title="Startup Readiness").exec()
        state["readiness_seen"] = True
        self._write_browser_state(state)

    def _activate_layer(self, layer: str):
        layers = list(ALL_LAYERS.keys())
        if layer not in layers:
            return
        self._set_section_expanded("Layers", True)
        self.layer_tabs.setCurrentIndex(layers.index(layer))

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
        self._schedule_render()

    def _schedule_render(self):
        if self.preview_array is None:
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
        self._active_layer = list(ALL_LAYERS.keys())[max(0, index)]
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
            self._set_section_expanded("Mask Tools", True)
        self._refresh_mask_controls()

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
        full = self._read_image_file(path)
        preview, scale = self._build_preview_proxy(full)
        interactive_preview, interactive_ratio = self._build_interactive_preview_proxy(preview)
        self.full_array = full
        self.preview_array = preview
        self.preview_scale = scale
        self._interactive_preview_array = interactive_preview
        self._interactive_preview_ratio = interactive_ratio
        self._face_profiles = {}
        self.preview_masks = None
        self.full_masks = None
        self._auto_preview_masks = None
        self._auto_full_masks = None
        self.preview_guides = None
        self.full_guides = None
        if project_state is not None:
            self._apply_project_state(project_state)
        else:
            self._run_segmentation()
            self._clear_document_history()
            self._push_document_history()

    def _read_image_file(self, path: str):
        ext = Path(path).suffix.lower()
        if ext in {".cr2", ".nef", ".arw", ".dng", ".raw"}:
            if not HAS_RAWPY:
                raise RuntimeError("rawpy is not installed. Install base dependencies first.")
            with rawpy.imread(path) as raw:
                rgb16 = raw.postprocess(use_camera_wb=True, output_bps=16)
            return rgb16.astype(np.float32) / 65535.0

        pil = Image.open(path)
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

    def _update_preview_label(self):
        if self.preview_image is None:
            return
        edited = self.preview_image.convert("RGB")
        edited = self._apply_mask_overlay(edited)
        edited = self._apply_expression_guide_overlay(edited)
        if self.preview_array is None:
            image = edited
        else:
            original = Image.fromarray(to_uint8(self.preview_array)).convert("RGB")
            image = self._compose_compare_image(original, edited)
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

        mask = self.preview_masks[self._active_layer]
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

    def _copy_params(self, params):
        return {layer: dict(values) for layer, values in (params or {}).items()}

    def _current_preview_render_inputs(self):
        if self.preview_array is None:
            return None, None, None
        if self._slider_drag_active <= 0 or self._interactive_preview_array is None:
            return (
                self.preview_array.copy(),
                self._copy_masks(self.preview_masks),
                self._copy_guides(self.preview_guides),
            )
        ratio = float(self._interactive_preview_ratio)
        masks = self._scale_masks(self.preview_masks, self._interactive_preview_array.shape[:2])
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
        mask = self.preview_masks[self._active_layer]
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
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(Path(self.file_path).with_name(f"{Path(self.file_path).stem}_qt_export.jpg")),
            "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not out_path:
            return
        try:
            result = process_all_layers(
                self.full_array,
                self._all_params(),
                self.full_masks,
                geometry=self.full_guides,
                layer_order=self._layer_order,
                layer_options=self._layer_options,
                color_settings=self._color_settings,
                runtime_settings=self._runtime_settings,
            )
            result.save(out_path)
            self.statusBar().showMessage(f"Exported {out_path}")
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))

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
        self._clear_mask_history()
        if self.preview_masks is not None:
            self._push_mask_history()
        self._schedule_render()


def run_qt_app():
    app = QApplication.instance() or QApplication([])
    window = PortraitEnhancerQtWindow()
    window.show()
    return app.exec()
