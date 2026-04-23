import sys
import os
import time
import cv2
import math
import numpy as np
from PIL import ImageFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QSlider, QFileDialog,
                               QComboBox, QSplitter, QGroupBox, QLineEdit,
                               QFrame, QScrollArea, QCheckBox, QProgressBar,
                               QStatusBar, QSpinBox, QColorDialog)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QEvent
from PySide6.QtGui import QFont, QImage, QPixmap, QFontDatabase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.font_manager import (FontManager, find_monospace_font, CHARSET_REGISTRY,
                                CHARSET_STANDARD)
from core.image_proc import process_image
from core.temporal import TemporalFilter
from core.renderer import Renderer
from core.video_proc import VideoProcessor
from core.ascii_image_renderer import render_ascii_image
from exporters.html_export import export_html as html_export_func
from exporters.ansi_export import export_ansi as ansi_export_func
from exporters.image_export import export_image as image_export_func
from exporters.video_export import export_video as video_export_func
from exporters.txt_export import export_txt as txt_export_func


# ── Presets ──
PRESETS = {
    "Default": {"cols": 150, "gamma": 10, "brightness": 0, "clahe": 0,
                "sharpen": 0, "denoise": 0, "dither": "none", "edge": "none"},
    "High Quality": {"cols": 250, "gamma": 10, "brightness": 0, "clahe": 15,
                     "sharpen": 5, "denoise": 2, "dither": "floyd-steinberg", "edge": "none"},
    "Fast": {"cols": 80, "gamma": 10, "brightness": 0, "clahe": 0,
             "sharpen": 0, "denoise": 0, "dither": "none", "edge": "none"},
    "Edge Heavy": {"cols": 150, "gamma": 10, "brightness": 0, "clahe": 10,
                   "sharpen": 8, "denoise": 0, "dither": "none", "edge": "hybrid"},
    "High Contrast": {"cols": 150, "gamma": 8, "brightness": 10, "clahe": 25,
                      "sharpen": 3, "denoise": 0, "dither": "ordered", "edge": "none"},
}


class VideoExportThread(QThread):
    """Background thread for video export to avoid freezing the UI."""
    progress = Signal(int, int)
    finished_signal = Signal()

    def __init__(self, video_proc, output_path, font_path, font_size, cols,
                 renderer, gamma, clahe_clip, brightness, sharpen, denoise,
                 dither_mode, edge_mode, invert, use_color, drop_color, drop_tolerance,
                 temporal_smooth, preserve_audio, custom_text=""):
        super().__init__()
        self.video_proc = video_proc
        self.output_path = output_path
        self.font_path = font_path
        self.font_size = font_size
        self.cols = cols
        self.renderer = renderer
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.brightness = brightness
        self.sharpen = sharpen
        self.denoise = denoise
        self.dither_mode = dither_mode
        self.edge_mode = edge_mode
        self.invert = invert
        self.use_color = use_color
        self.drop_color = drop_color
        self.drop_tolerance = drop_tolerance
        self.temporal_smooth = temporal_smooth
        self.preserve_audio = preserve_audio
        self.custom_text = custom_text

    def run(self):
        video_export_func(
            self.video_proc, self.output_path,
            self.font_path, self.font_size,
            self.cols, self.renderer,
            self.gamma, self.clahe_clip, process_image,
            brightness=self.brightness, sharpen=self.sharpen,
            denoise=self.denoise, dither_mode=self.dither_mode,
            edge_mode=self.edge_mode, invert=self.invert,
            use_color=self.use_color, drop_color=self.drop_color,
            drop_tolerance=self.drop_tolerance, temporal_smooth=self.temporal_smooth,
            preserve_audio=self.preserve_audio,
            progress_callback=self._on_progress,
            custom_text=self.custom_text
        )
        self.finished_signal.emit()

    def _on_progress(self, current, total):
        self.progress.emit(current, total)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASKEE: ASCII System & Encoding Engine")
        self.resize(1400, 900)

        # Core components
        font_path = find_monospace_font()
        self.font_manager = FontManager(font_path, font_size=14)
        self.renderer = Renderer(self.font_manager)
        self.renderer.set_ramp(CHARSET_REGISTRY["Standard"])

        # Preview rendering font (smaller for display)
        self.preview_font_size = 8
        if self.font_manager.font_path:
            self.preview_pil_font = ImageFont.truetype(self.font_manager.font_path, self.preview_font_size)
        else:
            self.preview_pil_font = ImageFont.load_default()
        self._update_preview_metrics()

        self.video_processor = None
        self.current_image = None
        self.is_playing = False
        self.export_thread = None
        
        self.system_fonts = FontManager.get_system_fonts()
        self.drop_color = None
        self.temporal_filter = TemporalFilter()
        self.picking_color_mode = False

        self.export_cols_manual = False
        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._do_render)

    def _update_preview_metrics(self):
        """Recalculate preview character cell dimensions."""
        self.preview_char_w = self.preview_pil_font.getlength("M")
        asc, desc = self.preview_pil_font.getmetrics()
        self.preview_char_h = asc + desc
        if self.preview_char_w == 0:
            self.preview_char_w = 6
        if self.preview_char_h == 0:
            self.preview_char_h = 12

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        outer_layout = QVBoxLayout(main_widget)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ─── Left Panel ───
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        control_scroll.setFixedWidth(300)
        control_scroll.setFrameShape(QFrame.NoFrame)

        control_panel = QWidget()
        cl = QVBoxLayout(control_panel)
        cl.setSpacing(6)

        # ASKEE Logo
        logo_text = r"""    ___   _____ __ __  ___________ 
   /   | / ___// //_/ / ____/ ____/
  / /| | \__ \/ ,<   / __/ / __/   
 / ___ |___/ / /| | / /___/ /___   
/_/  |_/____/_/ |_|/_____/_____/   """
        logo_label = QLabel(logo_text)
        logo_label.setTextFormat(Qt.PlainText)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(10)
        font.setBold(True)
        logo_label.setFont(font)
        logo_label.setAlignment(Qt.AlignLeft)
        logo_label.setStyleSheet("padding-bottom: 5px;")
        
        logo_layout = QHBoxLayout()
        logo_layout.addStretch()
        logo_layout.addWidget(logo_label)
        logo_layout.addStretch()
        cl.addLayout(logo_layout)

        # File
        g = QGroupBox("File")
        gl = QVBoxLayout(g)
        btn = QPushButton("Load Image / Video")
        btn.clicked.connect(self.load_file)
        gl.addWidget(btn)
        self.btn_play = QPushButton("▶  Play / Pause")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        gl.addWidget(self.btn_play)
        cl.addWidget(g)

        # Presets
        g = QGroupBox("Presets")
        gl = QVBoxLayout(g)
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS.keys()))
        self.combo_preset.currentTextChanged.connect(self.apply_preset)
        gl.addWidget(self.combo_preset)
        cl.addWidget(g)

        # Image Adjustments
        g_img = QGroupBox("Image Adjustments")
        gl_img = QVBoxLayout(g_img)
        self.slider_gamma, self.lbl_gamma = self._add_slider(gl_img, "Gamma:", 1, 30, 10)
        self.slider_brightness, self.lbl_brightness = self._add_slider(gl_img, "Brightness:", -50, 50, 0)
        self.slider_clahe, self.lbl_clahe = self._add_slider(gl_img, "Contrast (CLAHE):", 0, 50, 0)
        self.slider_sharpen, self.lbl_sharpen = self._add_slider(gl_img, "Sharpen:", 0, 20, 0)
        self.slider_denoise, self.lbl_denoise = self._add_slider(gl_img, "Denoise:", 0, 10, 0)
        cl.addWidget(g_img)

        # ASCII Rendering
        g_asc = QGroupBox("ASCII Rendering")
        gl_asc = QVBoxLayout(g_asc)
        
        row_custom = QHBoxLayout()
        row_custom.addWidget(QLabel("Custom Text:"))
        self.txt_custom_text = QLineEdit()
        self.txt_custom_text.setPlaceholderText("e.g. apple")
        self.txt_custom_text.textChanged.connect(self.request_render)
        row_custom.addWidget(self.txt_custom_text)
        gl_asc.addLayout(row_custom)
        
        self.slider_cols, self.lbl_cols = self._add_slider(gl_asc, "Columns:", 20, 400, 150)
        self.slider_cols.valueChanged.connect(self._sync_export_cols)
        
        row_color = QHBoxLayout()
        self.btn_drop_color = QPushButton("Pick Drop Color...")
        self.btn_drop_color.clicked.connect(self.pick_drop_color)
        row_color.addWidget(self.btn_drop_color)
        
        self.btn_eyedropper = QPushButton("💉")
        self.btn_eyedropper.setToolTip("Eyedropper Tool")
        self.btn_eyedropper.setFixedSize(24, 24)
        self.btn_eyedropper.clicked.connect(self.toggle_eyedropper_mode)
        row_color.addWidget(self.btn_eyedropper)
        
        self.btn_color_swatch = QPushButton("")
        self.btn_color_swatch.setFixedSize(24, 24)
        self.btn_color_swatch.setStyleSheet("background-color: transparent; border: 1px solid #555;")
        self.btn_color_swatch.clicked.connect(self.pick_drop_color)
        row_color.addWidget(self.btn_color_swatch)
        gl_asc.addLayout(row_color)
        
        self.slider_drop_tol, self.lbl_drop_tol = self._add_slider(gl_asc, "Tolerance:", 0, 100, 0)

        gl_asc.addWidget(QLabel("Dither:"))
        self.combo_dither = QComboBox()
        self.combo_dither.addItems(["none", "ordered", "floyd-steinberg"])
        self.combo_dither.currentTextChanged.connect(self.request_render)
        gl_asc.addWidget(self.combo_dither)

        gl_asc.addWidget(QLabel("Edge:"))
        self.combo_edge = QComboBox()
        self.combo_edge.addItems(["none", "hybrid", "edge_only"])
        self.combo_edge.currentTextChanged.connect(self.request_render)
        gl_asc.addWidget(self.combo_edge)

        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.stateChanged.connect(self.request_render)
        gl_asc.addWidget(self.chk_invert)

        self.chk_color = QCheckBox("Color Preview")
        self.chk_color.setChecked(True)
        self.chk_color.stateChanged.connect(self.request_render)
        gl_asc.addWidget(self.chk_color)

        # Character Set (Moved here)
        gl_asc.addWidget(QLabel("Character Set:"))
        self.combo_charset = QComboBox()
        charset_options = list(CHARSET_REGISTRY.keys()) + ["Custom"]
        self.combo_charset.addItems(charset_options)
        self.combo_charset.currentTextChanged.connect(self.change_charset)
        gl_asc.addWidget(self.combo_charset)
        
        self.txt_custom = QLineEdit()
        self.txt_custom.setPlaceholderText("Custom: .:-=+*#%@")
        self.txt_custom.setText(CHARSET_REGISTRY["Standard"])
        self.txt_custom.setEnabled(False)
        self.txt_custom.editingFinished.connect(self.apply_custom_charset)
        gl_asc.addWidget(self.txt_custom)

        cl.addWidget(g_asc)

        # Video Processing
        g_vid = QGroupBox("Video Processing")
        gl_vid = QVBoxLayout(g_vid)
        self.slider_temporal, self.lbl_temporal = self._add_slider(gl_vid, "Temporal Smooth:", 0, 99, 0)
        cl.addWidget(g_vid)



        # Preview font size and selection
        g = QGroupBox("Preview & Font")
        gl = QVBoxLayout(g)
        
        self.combo_sys_fonts = QComboBox()
        self.combo_sys_fonts.addItems(list(self.system_fonts.keys()))
        self.combo_sys_fonts.currentTextChanged.connect(self.change_system_font)
        gl.addWidget(self.combo_sys_fonts)

        self.slider_preview_size, self.lbl_preview_size = self._add_slider(
            gl, "Font Size:", 4, 20, self.preview_font_size, connect=False)
        self.slider_preview_size.valueChanged.connect(self._change_preview_font_size)
        cl.addWidget(g)

        cl.addStretch()
        control_scroll.setWidget(control_panel)

        # ─── Right Panel (Preview) ───
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setStyleSheet("QScrollArea { background-color: #0a0a0a; border: none; }")

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.preview_label.setStyleSheet("background-color: #0a0a0a;")
        self.preview_label.installEventFilter(self)
        self.preview_scroll.setWidget(self.preview_label)

        splitter.addWidget(control_scroll)
        splitter.addWidget(self.preview_scroll)
        splitter.setSizes([300, 1100])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer_layout.addWidget(splitter, stretch=1)

        # ── Bottom Export Bar ──
        export_frame = QFrame()
        export_frame.setFrameShape(QFrame.StyledPanel)
        el = QHBoxLayout(export_frame)
        el.setContentsMargins(8, 4, 8, 4)

        el.addWidget(QLabel("Export Cols:"))
        self.spin_export_cols = QSpinBox()
        self.spin_export_cols.setRange(20, 5000)
        self.spin_export_cols.setValue(150)
        self.spin_export_cols.valueChanged.connect(self._mark_export_cols_manual)
        el.addWidget(self.spin_export_cols)

        el.addWidget(QLabel("Font Px:"))
        self.spin_export_font_size = QSpinBox()
        self.spin_export_font_size.setRange(4, 200)
        self.spin_export_font_size.setValue(14)
        el.addWidget(self.spin_export_font_size)

        self.chk_export_audio = QCheckBox("Preserve Audio")
        self.chk_export_audio.setChecked(True)
        el.addWidget(self.chk_export_audio)

        el.addSpacing(15)
        el.addWidget(QLabel("Export:"))
        for label, slot in [("TXT", self.export_txt), ("HTML", self.export_html),
                            ("ANSI", self.export_ansi), ("Image", self.export_image),
                            ("Video", self.export_video)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            el.addWidget(b)

        self.chk_color_export = QCheckBox("Color")
        self.chk_color_export.setChecked(True)
        el.addWidget(self.chk_color_export)

        el.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        el.addWidget(self.progress_bar)

        outer_layout.addWidget(export_frame, stretch=0)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — Load an image or video to begin")

    def _add_slider(self, layout, label, min_val, max_val, default, connect=True):
        """Helper to add a labeled slider with value display."""
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        lbl = QLabel(str(default))
        lbl.setFixedWidth(35)
        row.addWidget(slider)
        row.addWidget(lbl)
        layout.addLayout(row)
        if connect:
            slider.valueChanged.connect(self.request_render)
        return slider, lbl

    # ── File I/O ──

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "Images/Videos (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.mp4 *.avi *.mkv *.mov)")
        if not path:
            return

        if path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.gif')):
            if self.video_processor:
                self.video_processor.release()
            self.video_processor = VideoProcessor(path)
            self.current_image = self.video_processor.get_frame()
            self.btn_play.setEnabled(True)
            w, h = self.video_processor.get_resolution()
            self.status_bar.showMessage(
                f"Video: {w}×{h} | {self.video_processor.frame_count} frames | "
                f"{self.video_processor.fps:.1f} fps | {self.video_processor.get_duration():.1f}s")
        else:
            if self.video_processor:
                self.video_processor.release()
                self.video_processor = None
            self.current_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            self.btn_play.setEnabled(False)
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.status_bar.showMessage(f"Image: {w}×{h}")

        self.temporal_filter.reset()
        self.request_render()

    # ── Playback ──

    def toggle_playback(self):
        if not self.video_processor:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            current_frame = self.video_processor.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame >= self.video_processor.frame_count:
                self.video_processor.set_position(0)
            fps = self.video_processor.fps if self.video_processor else 30
            self.timer.start(int(1000 / fps))
            self.btn_play.setText("⏸  Pause")
        else:
            self.timer.stop()
            self.btn_play.setText("▶  Play")

    def update_frame(self):
        if self.video_processor:
            frame = self.video_processor.get_frame()
            if frame is None:
                self.timer.stop()
                self.is_playing = False
                self.btn_play.setText("▶  Play")
                return
            
            smoothing_factor = self.slider_temporal.value() / 100.0
            frame = self.temporal_filter.apply(frame, smoothing_factor)
            
            self.current_image = frame
            self.request_render()

    # ── Charset ──

    def change_charset(self, text):
        if text in CHARSET_REGISTRY:
            self.renderer.set_ramp(CHARSET_REGISTRY[text])
            self.txt_custom.setEnabled(False)
        elif text == "Custom":
            self.txt_custom.setEnabled(True)
            self.apply_custom_charset()
            return
        self.request_render()

    def apply_custom_charset(self):
        text = self.txt_custom.text().strip()
        if len(text) >= 2:
            self.renderer.set_ramp(text)
            self.request_render()

    # ── Presets ──

    def pick_drop_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.drop_color = (color.blue(), color.green(), color.red())
            self.btn_color_swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #555;")
            self.request_render()

    def toggle_eyedropper_mode(self):
        self.picking_color_mode = not self.picking_color_mode
        if self.picking_color_mode:
            self.btn_eyedropper.setStyleSheet("background-color: #555;")
            self.preview_label.setCursor(Qt.CrossCursor)
        else:
            self.btn_eyedropper.setStyleSheet("")
            self.preview_label.unsetCursor()

    def eventFilter(self, source, event):
        if source == self.preview_label and event.type() == QEvent.MouseButtonPress:
            if self.picking_color_mode and self.current_image is not None:
                pos = event.position()
                x, y = int(pos.x()), int(pos.y())
                pixmap = self.preview_label.pixmap()
                if pixmap:
                    orig_h, orig_w = self.current_image.shape[:2]
                    scaled_w, scaled_h = pixmap.width(), pixmap.height()
                    
                    # Map to original image coordinates
                    orig_x = int((x / scaled_w) * orig_w)
                    orig_y = int((y / scaled_h) * orig_h)
                    
                    # Ensure coordinates are within bounds
                    orig_x = max(0, min(orig_w - 1, orig_x))
                    orig_y = max(0, min(orig_h - 1, orig_y))
                    
                    pixel = self.current_image[orig_y, orig_x]
                    if hasattr(pixel, "__len__"):
                        b, g, r = pixel[0], pixel[1], pixel[2]
                    else:
                        b = g = r = pixel
                        
                    self.drop_color = (int(b), int(g), int(r))
                    
                    # Update UI
                    hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                    self.btn_color_swatch.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #555;")
                    
                    self.toggle_eyedropper_mode()
                    self.request_render()
                return True
        return super().eventFilter(source, event)

    def _sync_export_cols(self, val):
        if not self.export_cols_manual:
            self.spin_export_cols.blockSignals(True)
            self.spin_export_cols.setValue(val)
            self.spin_export_cols.blockSignals(False)

    def _mark_export_cols_manual(self, val):
        self.export_cols_manual = True

    def change_system_font(self, font_name):
        path = self.system_fonts.get(font_name)
        if path:
            self.font_manager.set_font(path)
            self.preview_pil_font = ImageFont.truetype(path, self.preview_font_size)
            self._update_preview_metrics()
            self.change_charset(self.combo_charset.currentText())
            self.request_render()

    def apply_preset(self, name):
        if name not in PRESETS:
            return
        p = PRESETS[name]
        # Block signals during batch update
        for s in [self.slider_cols, self.slider_gamma, self.slider_brightness,
                  self.slider_clahe, self.slider_sharpen, self.slider_denoise]:
            s.blockSignals(True)
        self.combo_dither.blockSignals(True)
        self.combo_edge.blockSignals(True)

        self.slider_cols.setValue(p["cols"])
        self.slider_gamma.setValue(p["gamma"])
        self.slider_brightness.setValue(p["brightness"])
        self.slider_clahe.setValue(p["clahe"])
        self.slider_sharpen.setValue(p["sharpen"])
        self.slider_denoise.setValue(p["denoise"])
        self.combo_dither.setCurrentText(p["dither"])
        self.combo_edge.setCurrentText(p["edge"])

        for s in [self.slider_cols, self.slider_gamma, self.slider_brightness,
                  self.slider_clahe, self.slider_sharpen, self.slider_denoise]:
            s.blockSignals(False)
        self.combo_dither.blockSignals(False)
        self.combo_edge.blockSignals(False)

        self.request_render()

    # ── Preview font size ──

    def _change_preview_font_size(self, val):
        self.lbl_preview_size.setText(str(val))
        self.preview_font_size = val
        if self.font_manager.font_path:
            self.preview_pil_font = ImageFont.truetype(self.font_manager.font_path, val)
        self._update_preview_metrics()
        self.request_render()

    # ── Core Render ──

    def _get_params(self):
        """Collect all current parameter values."""
        return {
            "cols": self.slider_cols.value(),
            "gamma": self.slider_gamma.value() / 10.0,
            "brightness": self.slider_brightness.value(),
            "clahe_clip": self.slider_clahe.value() / 10.0,
            "sharpen": self.slider_sharpen.value() / 10.0,
            "denoise": self.slider_denoise.value() / 10.0,
            "dither_mode": self.combo_dither.currentText(),
            "edge_mode": self.combo_edge.currentText(),
            "invert": self.chk_invert.isChecked(),
            "color": self.chk_color.isChecked(),
            "drop_color": self.drop_color,
            "drop_tolerance": self.slider_drop_tol.value(),
            "temporal_smooth": self.slider_temporal.value(),
            "custom_text": self.txt_custom_text.text(),
        }

    def request_render(self, *_args):
        if self.current_image is None:
            return

        p = self._get_params()

        # Update slider labels immediately for responsive UI
        self.lbl_cols.setText(str(p["cols"]))
        self.lbl_gamma.setText(f'{p["gamma"]:.1f}')
        self.lbl_brightness.setText(str(p["brightness"]))
        self.lbl_clahe.setText(f'{p["clahe_clip"]:.1f}')
        self.lbl_sharpen.setText(f'{p["sharpen"]:.1f}')
        self.lbl_denoise.setText(f'{p["denoise"]:.1f}')
        self.lbl_drop_tol.setText(str(p["drop_tolerance"]))
        self.lbl_temporal.setText(str(p["temporal_smooth"]))

        # Debounce the heavy rendering
        self.render_timer.start(50)

    def _do_render(self):
        if self.current_image is None:
            return

        t0 = time.perf_counter()
        p = self._get_params()

        processed = process_image(
            self.current_image, gamma=p["gamma"], brightness=p["brightness"],
            clahe_clip=p["clahe_clip"], sharpen=p["sharpen"], denoise=p["denoise"])

        ascii_text, color_data = self.renderer.render_frame(
            processed, p["cols"], dither_mode=p["dither_mode"],
            edge_mode=p["edge_mode"], invert=p["invert"],
            drop_color=p["drop_color"], drop_tolerance=p["drop_tolerance"],
            raw_image=self.current_image, custom_text=p["custom_text"])

        self.last_ascii = ascii_text
        self.last_color = color_data

        # Render preview image using shared renderer
        color_arg = color_data if p["color"] else None
        bg = (255, 255, 255) if p["invert"] else (10, 10, 10)
        fg = (0, 0, 0) if p["invert"] else (224, 224, 224)

        pil_img = render_ascii_image(
            ascii_text, self.preview_pil_font,
            self.preview_char_w, self.preview_char_h,
            color_image=color_arg, bg_color=bg, fg_color=fg, invert=False
        )

        # Convert PIL → QPixmap
        img_array = np.array(pil_img)
        h, w, ch = img_array.shape
        qimg = QImage(img_array.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        self.preview_label.setPixmap(pixmap)
        self.preview_label.resize(pixmap.size())

        dt = time.perf_counter() - t0
        lines = ascii_text.split('\n')
        rows = len(lines)
        cols_actual = len(lines[0]) if lines else 0
        self.status_bar.showMessage(
            f"Grid: {cols_actual}×{rows} | Render: {dt*1000:.0f}ms | "
            f"Font: {self.preview_font_size}px")

    # ── Exports ──

    def _render_for_export(self):
        p = self._get_params()
        export_cols = self.spin_export_cols.value()
        
        processed = process_image(
            self.current_image, gamma=p["gamma"], brightness=p["brightness"],
            clahe_clip=p["clahe_clip"], sharpen=p["sharpen"], denoise=p["denoise"])

        return self.renderer.render_frame(
            processed, export_cols, dither_mode=p["dither_mode"],
            edge_mode=p["edge_mode"], invert=p["invert"], 
            drop_color=p["drop_color"], drop_tolerance=p["drop_tolerance"],
            raw_image=self.current_image, custom_text=p["custom_text"])

    def export_txt(self):
        if self.current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save TXT", "", "Text Files (*.txt)")
        if path:
            ascii_text, _ = self._render_for_export()
            txt_export_func(ascii_text, path)
            self.status_bar.showMessage(f"Exported TXT → {path}")

    def export_html(self):
        if self.current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save HTML", "", "HTML Files (*.html)")
        if path:
            ascii_text, color_data = self._render_for_export()
            color = color_data if self.chk_color_export.isChecked() else None
            html_export_func(ascii_text, path, color_image=color,
                             invert=self.chk_invert.isChecked())
            self.status_bar.showMessage(f"Exported HTML → {path}")

    def export_ansi(self):
        if self.current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ANSI", "", "Text Files (*.txt *.ans)")
        if path:
            ascii_text, color_data = self._render_for_export()
            color = color_data if self.chk_color_export.isChecked() else None
            ansi_text = ansi_export_func(ascii_text, color,
                                         invert=self.chk_invert.isChecked())
            with open(path, 'w', encoding='utf-8') as f:
                f.write(ansi_text)
            self.status_bar.showMessage(f"Exported ANSI → {path}")

    def export_image(self):
        if self.current_image is None or self.font_manager.font_path is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if path:
            self.status_bar.showMessage("Rendering high-res image...")
            QApplication.processEvents()
            ascii_text, color_data = self._render_for_export()
            color = color_data if self.chk_color_export.isChecked() else None
            image_export_func(ascii_text, path, self.font_manager.font_path,
                              self.spin_export_font_size.value(), color_image=color,
                              invert=self.chk_invert.isChecked())
            self.status_bar.showMessage(f"Exported Image → {path}")

    def export_video(self):
        if not self.video_processor:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Video/GIF Files (*.mp4 *.gif)")
        if not path or self.font_manager.font_path is None:
            return

        p = self._get_params()
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.export_thread = VideoExportThread(
            self.video_processor, path,
            self.font_manager.font_path, self.spin_export_font_size.value(),
            self.spin_export_cols.value(), self.renderer, p["gamma"], p["clahe_clip"],
            p["brightness"], p["sharpen"], p["denoise"],
            p["dither_mode"], p["edge_mode"], p["invert"],
            self.chk_color_export.isChecked(), p["drop_color"], p["drop_tolerance"],
            p["temporal_smooth"], self.chk_export_audio.isChecked(), p["custom_text"]
        )
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.finished_signal.connect(self._on_export_done)
        self.status_bar.showMessage("Exporting video...")
        self.export_thread.start()

    def _on_export_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
            self.status_bar.showMessage(f"Exporting video: {current}/{total} frames")

    def _on_export_done(self):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Video export complete!")
