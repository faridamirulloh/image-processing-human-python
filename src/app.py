"""
Aplikasi Utama
Deteksi dan penghitungan orang secara real-time menggunakan model YOLO.
Menangani tata letak UI, manajemen kamera, perekaman, dan proses deteksi.
"""

import os
import time
import traceback
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QFrame,
    QSplitter, QStatusBar, QMessageBox, QFileDialog, QMenu, QAction,
    QDialog, QSlider, QSpinBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from services import CameraService, VideoService, DetectorService, RecordingService
from widgets import VideoWidget, StatsWidget
from utils.constants import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, YOLO_MODELS,
    DEFAULT_CAPTURE_FPS, MIN_FPS, MAX_FPS
)
from utils import styles


# =============================================================================
# Background Threads
# =============================================================================

class ModelLoaderThread(QThread):
    """Thread latar belakang untuk memuat model YOLO tanpa memblokir UI."""
    
    # Signal: (success: bool, model_name: str, error: str)
    model_loaded = pyqtSignal(bool, str, str)
    
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self.detector_service = None
    
    def run(self):
        """Muat model di thread terpisah."""
        try:
            self.detector_service = DetectorService(self._model_name, use_gpu=False)
            if self.detector_service.init_error:
                self.model_loaded.emit(False, self._model_name, self.detector_service.init_error)
            else:
                self.model_loaded.emit(True, self._model_name, "")
        except Exception as e:
            self.model_loaded.emit(False, self._model_name, str(e))


# =============================================================================
# Settings Dialog
# =============================================================================

class SettingsDialog(QDialog):
    """Modal settings dialog for performance tuning and FPS control."""
    
    def __init__(self, parent=None, video_service=None, detector_service=None, video_widget=None):
        super().__init__(parent)
        self._video_service = video_service
        self._detector_service = detector_service
        self._video_widget = video_widget
        self.setWindowTitle("⚙️ Performance Settings")
        self.setFixedWidth(420)
        self.setStyleSheet(styles.get_settings_dialog_style())
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("⚙️ Performance Settings")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #00d9ff;")
        layout.addWidget(title)
        
        # === FPS Group ===
        fps_group = QGroupBox("Frame Rate")
        fps_layout = QVBoxLayout(fps_group)
        fps_layout.setSpacing(8)
        
        # FPS Slider
        slider_row = QHBoxLayout()
        slider_label = QLabel("Target FPS:")
        slider_row.addWidget(slider_label)
        
        self._fps_slider = QSlider(Qt.Horizontal)
        self._fps_slider.setMinimum(MIN_FPS)
        self._fps_slider.setMaximum(MAX_FPS)
        current_fps = self._video_service.get_target_fps() if self._video_service else DEFAULT_CAPTURE_FPS
        self._fps_slider.setValue(current_fps)
        self._fps_slider.setStyleSheet(styles.get_slider_style())
        slider_row.addWidget(self._fps_slider)
        
        self._fps_value_label = QLabel(str(current_fps))
        self._fps_value_label.setFixedWidth(30)
        self._fps_value_label.setStyleSheet("color: #00d9ff; font-weight: bold;")
        slider_row.addWidget(self._fps_value_label)
        
        fps_layout.addLayout(slider_row)
        
        # FPS Presets
        preset_row = QHBoxLayout()
        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet("color: #8b8b8b; font-size: 11px;")
        preset_row.addWidget(preset_label)
        
        for name, value in [("5", 5), ("10", 10), ("15", 15), ("20", 20), ("30", 30)]:
            btn = QPushButton(name)
            btn.setFixedSize(36, 26)
            btn.setStyleSheet(styles.get_icon_button_style("#2d2d44", "#00d9ff"))
            btn.clicked.connect(lambda checked, v=value: self._set_fps_preset(v))
            preset_row.addWidget(btn)
        
        preset_row.addStretch()
        fps_layout.addLayout(preset_row)
        layout.addWidget(fps_group)
        
        # === Detection Group ===
        det_group = QGroupBox("Detection Performance")
        det_layout = QVBoxLayout(det_group)
        det_layout.setSpacing(8)
        
        # Inference Scale
        scale_row = QHBoxLayout()
        scale_label = QLabel("Inference Resolution:")
        scale_row.addWidget(scale_label)
        
        self._scale_combo = QComboBox()
        self._scale_combo.setStyleSheet(styles.get_combo_style())
        self._scale_combo.addItem("Full (1.0x)", 1.0)
        self._scale_combo.addItem("¾ (0.75x)", 0.75)
        self._scale_combo.addItem("Half (0.5x)", 0.5)
        self._scale_combo.addItem("Quarter (0.25x)", 0.25)
        
        # Set current value
        if self._detector_service:
            current_scale = self._detector_service.get_inference_scale()
            for i in range(self._scale_combo.count()):
                if self._scale_combo.itemData(i) == current_scale:
                    self._scale_combo.setCurrentIndex(i)
                    break
        
        scale_row.addWidget(self._scale_combo)
        det_layout.addLayout(scale_row)
        
        # Skip Frames
        skip_row = QHBoxLayout()
        skip_label = QLabel("Skip Frames:")
        skip_row.addWidget(skip_label)
        
        self._skip_spin = QSpinBox()
        self._skip_spin.setMinimum(1)
        self._skip_spin.setMaximum(10)
        if self._detector_service:
            self._skip_spin.setValue(self._detector_service.get_skip_frames())
        skip_row.addWidget(self._skip_spin)
        
        skip_desc = QLabel("(1 = every frame)")
        skip_desc.setStyleSheet("color: #8b8b8b; font-size: 11px;")
        skip_row.addWidget(skip_desc)
        skip_row.addStretch()
        det_layout.addLayout(skip_row)
        
        layout.addWidget(det_group)
        
        # === Display Group ===
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        self._fast_scaling_cb = QCheckBox("Fast scaling (lower quality, saves CPU)")
        self._fast_scaling_cb.setChecked(
            self._video_widget.get_fast_scaling() if self._video_widget else True
        )
        display_layout.addWidget(self._fast_scaling_cb)
        
        layout.addWidget(display_group)
        
        # === Low Spec Mode ===
        low_spec_group = QGroupBox("Quick Presets")
        low_spec_layout = QVBoxLayout(low_spec_group)
        
        low_spec_btn = QPushButton("⚡ Low Spec Mode")
        low_spec_btn.setToolTip(
            "Optimizes for slow CPUs:\n"
            "• FPS → 15\n"
            "• Inference → Half resolution\n"
            "• Skip frames → 2\n"
            "• Fast scaling → ON"
        )
        low_spec_btn.setStyleSheet(styles.get_button_style("#ffa502", "#e69500"))
        low_spec_btn.setFixedHeight(36)
        low_spec_btn.clicked.connect(self._apply_low_spec_mode)
        low_spec_layout.addWidget(low_spec_btn)
        
        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#5a5a7a"))
        reset_btn.setFixedHeight(36)
        reset_btn.clicked.connect(self._reset_defaults)
        low_spec_layout.addWidget(reset_btn)
        
        layout.addWidget(low_spec_group)
        
        # === Apply / Close ===
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        apply_btn = QPushButton("✓ Apply")
        apply_btn.setStyleSheet(styles.get_button_style("#00d9ff", "#00b8d9"))
        apply_btn.setFixedSize(100, 36)
        apply_btn.clicked.connect(self._apply_settings)
        btn_row.addWidget(apply_btn)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#5a5a7a"))
        close_btn.setFixedSize(100, 36)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        
        layout.addLayout(btn_row)
        
        # Connect live slider update
        self._fps_slider.valueChanged.connect(
            lambda v: self._fps_value_label.setText(str(v))
        )
    
    def _set_fps_preset(self, fps: int):
        """Set FPS slider to a preset value."""
        self._fps_slider.setValue(fps)
    
    def _apply_low_spec_mode(self):
        """Apply all low-spec optimizations."""
        self._fps_slider.setValue(15)
        self._scale_combo.setCurrentIndex(2)  # Half (0.5x)
        self._skip_spin.setValue(2)
        self._fast_scaling_cb.setChecked(True)
        self._apply_settings()
    
    def _reset_defaults(self):
        """Reset all settings to defaults."""
        self._fps_slider.setValue(DEFAULT_CAPTURE_FPS)
        self._scale_combo.setCurrentIndex(0)  # Full (1.0x)
        self._skip_spin.setValue(1)
        self._fast_scaling_cb.setChecked(True)
        self._apply_settings()
    
    def _apply_settings(self):
        """Push all settings to the services."""
        fps = self._fps_slider.value()
        scale = self._scale_combo.currentData()
        skip = self._skip_spin.value()
        fast_scaling = self._fast_scaling_cb.isChecked()
        
        if self._video_service:
            self._video_service.set_target_fps(fps)
        
        if self._detector_service:
            self._detector_service.set_inference_scale(scale)
            self._detector_service.set_skip_frames(skip)
        
        if self._video_widget:
            self._video_widget.set_fast_scaling(fast_scaling)
        
        # Notify parent to update stats display
        parent = self.parent()
        if parent and hasattr(parent, '_on_settings_applied'):
            parent._on_settings_applied(fps)


# =============================================================================
# Main Application Window
# =============================================================================

class MainWindow(QMainWindow):
    """Aplikasi utama dengan pratinjau kamera, deteksi AI, dan perekaman."""
    
    def __init__(self):
        super().__init__()
        
        # Layanan inti
        self._camera_service = CameraService()
        self._video_service = VideoService()
        self._recording_service = RecordingService()
        self._detector_service = None
        
        # Thread latar belakang (simpan referensi agar tidak di-garbage-collect)
        self._model_loader_thread = None
        self._camera_scan_thread = None
        
        # Bendera status
        self._is_running = False
        self._is_previewing = False
        self._is_loading_model = False
        self._current_camera = 0
        self._compact_mode = False
        
        # Processing mode state
        self._processing_mode = 'streaming'  # 'streaming' atau 'low-specs'
        self._lowspec_ready = True
        self._last_annotated_frame = None
        
        # Timer untuk low-specs cooldown
        self._lowspec_timer = QTimer(self)
        self._lowspec_timer.setSingleShot(True)
        self._lowspec_timer.timeout.connect(self._lowspec_mark_ready)
        
        # FPS tracking (measures actual detection rate)
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = 0
        self._last_fps_update = 0
        
        # Detection throttling: video streams at camera rate,
        # YOLO runs at target FPS, cached boxes redrawn on other frames
        self._last_detection_time = 0.0
        self._cached_detections = []    # Last YOLO results for redraw
        self._cached_person_count = 0
        
        # Inisiasi UI, hubungkan sinyal, pindai kamera, dan load model AI
        self._init_ui()
        self._connect_signals()
        self._refresh_cameras()
        QTimer.singleShot(500, self._preload_model)
    
    def _init_ui(self):
        """Tata letak window utama."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(640, 480)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Icon aplikasi
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.setStyleSheet(styles.get_main_theme())
        
        # Tata letak root
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Bilah kontrol atas
        control_bar = self._create_control_bar()
        main_layout.addWidget(control_bar)
        
        # Pemisah: video (kiri) + statistik (kanan)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(styles.get_splitter_handle_style())
        
        # Kiri: umpan video langsung
        self._video_widget = VideoWidget()
        splitter.addWidget(self._video_widget)
        
        # Kanan: statistik deteksi
        self._stats_widget = StatsWidget()
        splitter.addWidget(self._stats_widget)
        
        # Pemisahan default: 75% video, 25% statistik
        splitter.setSizes([750, 250])
        main_layout.addWidget(splitter)
        
        # Status bar bawah
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet(styles.get_status_bar_style())
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready - Select a camera and click Start")

    def _create_control_bar(self) -> QWidget:
        """Kontrol bar atas: pemilih kamera, pemilih model, dan tombol tindakan."""
        control_bar = QFrame()
        control_bar.setStyleSheet(styles.get_control_bar_style())
        control_bar.setMaximumHeight(55)
        
        layout = QHBoxLayout(control_bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(12)
        
        # 1. Tombol Refresh
        self._refresh_btn = QPushButton("R")
        self._refresh_btn.setToolTip("Refresh camera list")
        self._refresh_btn.setFixedSize(32, 32)
        self._refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a6a;
                border: none;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #00d9ff;
                color: #000000;
            }
            QPushButton:pressed {
                background-color: #0099cc;
            }
        """)
        layout.addWidget(self._refresh_btn)
        
        # 2. Pemilih Kamera & Model
        self._create_camera_controls(layout)
        
        layout.addStretch()
        
        # 3. Kontrol Perekaman
        self._create_recording_controls(layout)
        
        # 4. Settings Button
        self._settings_btn = QPushButton("⚙️")
        self._settings_btn.setToolTip("Performance Settings")
        self._settings_btn.setFixedSize(32, 32)
        self._settings_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#00d9ff"))
        layout.addWidget(self._settings_btn)
        
        # Vertical separator
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #2d2d44;")
        separator.setFixedHeight(24)
        layout.addWidget(separator)
        
        # 5. Start/Stop Buttons
        self._start_btn = QPushButton("▶ Start")
        self._start_btn.setMinimumSize(32, 32)
        self._start_btn.setStyleSheet(styles.get_button_style("#00d9ff", "#00b8d9"))
        layout.addWidget(self._start_btn)
        
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setMinimumSize(32, 32)
        self._stop_btn.setStyleSheet(styles.get_button_style("#ff4757", "#ff3344"))
        self._stop_btn.setEnabled(False)
        layout.addWidget(self._stop_btn)
        
        return control_bar

    def _create_camera_controls(self, parent_layout: QHBoxLayout):
        """Tambahkan dropdown kamera dan model ke tata letak."""
        # Dropdown kamera
        camera_layout = QHBoxLayout()
        camera_layout.setSpacing(5)
        camera_label = QLabel("📷")
        camera_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumWidth(140)
        self._camera_combo.setStyleSheet(styles.get_combo_style())
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self._camera_combo)
        parent_layout.addLayout(camera_layout)
        
        # Dropdown model AI
        model_layout = QHBoxLayout()
        model_layout.setSpacing(5)
        model_label = QLabel("🧠")
        model_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(150)
        self._model_combo.setStyleSheet(styles.get_combo_style())
        
        # Isi dengan model YOLO yang tersedia
        for model_name, info in YOLO_MODELS.items():
            self._model_combo.addItem(model_name)
            self._model_combo.setItemData(
                self._model_combo.count() - 1, 
                info['description'], 
                Qt.ToolTipRole
            )
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self._model_combo)
        parent_layout.addLayout(model_layout)
        
        # Dropdown mode pemrosesan
        # mode_layout = QHBoxLayout()
        # mode_layout.setSpacing(5)
        # mode_label = QLabel("⚡")
        # mode_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        # self._mode_combo = QComboBox()
        # self._mode_combo.setMinimumWidth(120)
        # self._mode_combo.setStyleSheet(styles.get_combo_style())
        # self._mode_combo.addItem("Streaming")
        # self._mode_combo.addItem("Low-Specs")
        # self._mode_combo.setToolTip(
        #     "Streaming: Process every frame (high CPU usage)\n"
        #     "Low-Specs: Capture → Detect → Display → Repeat (low CPU usage)"
        # )
        
        # mode_layout.addWidget(mode_label)
        # mode_layout.addWidget(self._mode_combo)
        # parent_layout.addLayout(mode_layout)

    def _create_recording_controls(self, parent_layout: QHBoxLayout):
        """Tambahkan tombol folder, tangkapan layar, dan rekam ke tata letak."""
        # Tombol folder
        self._folder_btn = QPushButton("📂")
        self._folder_btn.setFixedSize(32, 32)
        self._folder_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#5a5a7a"))
        self._folder_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self._update_folder_tooltip()
        parent_layout.addWidget(self._folder_btn)
        
        # Tombol screenshot
        self._capture_btn = QPushButton("📸")
        self._capture_btn.setToolTip("Capture screenshot")
        self._capture_btn.setFixedSize(32, 32)
        self._capture_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#5a5a7a"))
        self._capture_btn.setEnabled(False)
        parent_layout.addWidget(self._capture_btn)
        
        # Tombol rekam
        self._record_btn = QPushButton("⏺ Record")
        self._record_btn.setToolTip("Start / stop video recording")
        self._record_btn.setMinimumSize(32, 32)
        self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
        self._record_btn.setEnabled(False)
        parent_layout.addWidget(self._record_btn)

    
    def _connect_signals(self):
        """Hubungkan sinyal widget ke metode penangan."""
        # Tombol kontrol bar
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._refresh_btn.clicked.connect(self._refresh_cameras)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        # self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._settings_btn.clicked.connect(self._on_settings_open)
        
        # Tombol rekam & screenshot
        self._folder_btn.clicked.connect(self._on_folder_open)
        self._folder_btn.customContextMenuRequested.connect(self._on_folder_select)
        self._capture_btn.clicked.connect(self._on_capture)
        self._record_btn.clicked.connect(self._on_record_toggle)
        
        # Callback penangkapan video
        self._video_service.frame_ready.connect(self._on_frame_ready)
        self._video_service.error_occurred.connect(self._on_video_error)
    
    # =========================================================================
    # Pemindaian Kamera (non-blocking via thread)
    # =========================================================================
    
    def _refresh_cameras(self):
        """Pindai kamera yang tersedia di thread latar belakang."""
        self._status_bar.showMessage("⏳ Scanning for cameras...")
        self._refresh_btn.setEnabled(False)
        
        self._camera_scan_thread = self._camera_service.scan_async()
        self._camera_scan_thread.cameras_found.connect(self._on_cameras_found)
        self._camera_scan_thread.scan_error.connect(self._on_camera_scan_error)
        self._camera_scan_thread.start()
    
    def _on_cameras_found(self, cameras: list):
        """Tangani hasil pemindaian kamera."""
        self._refresh_btn.setEnabled(True)
        self._camera_combo.clear()
        
        if not cameras:
            self._camera_combo.addItem("No cameras found")
            self._start_btn.setEnabled(False)
            self._video_widget.show_no_camera()
            self._status_bar.showMessage("⚠️ No cameras detected")
        else:
            for camera in cameras:
                display_text = f"{camera['name']} ({camera['resolution'][0]}x{camera['resolution'][1]})"
                self._camera_combo.addItem(display_text, camera['index'])
            
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"✓ Found {len(cameras)} camera(s)")
            
            # Buka preview kamera secara otomatis
            QTimer.singleShot(100, self._start_preview)
    
    def _on_camera_scan_error(self, error: str):
        """Tangani error pemindaian kamera."""
        self._refresh_btn.setEnabled(True)
        self._camera_combo.clear()
        self._camera_combo.addItem("No cameras found")
        self._start_btn.setEnabled(False)
        self._video_widget.show_error(error)
        self._status_bar.showMessage(f"⚠️ Camera scan failed: {error}")
        
        QMessageBox.warning(
            self,
            "Peringatan Kamera",
            f"Gagal memindai kamera:\n\n{error}\n\n"
            "Silakan periksa koneksi kamera dan coba klik Refresh."
        )
    
    def _on_camera_changed(self, index: int):
        """Beralih ke kamera yang baru dipilih."""
        if index >= 0:
            camera_index = self._camera_combo.itemData(index)
            if camera_index is not None:
                # Hentikan deteksi saat ini, lalu mulai ulang dalam mode yang sama
                was_detecting = self._is_running
                if self._is_running:
                    self._on_stop()
                elif self._is_previewing:
                    self._stop_preview()
                
                self._current_camera = camera_index
                
                # Restart in same mode after brief delay
                if was_detecting:
                    QTimer.singleShot(200, lambda: self._on_start())
                else:
                    QTimer.singleShot(200, self._start_preview)
    
    # =========================================================================
    # Pemuatan Model (non-blocking via thread)
    # =========================================================================
    
    def _on_model_changed(self, model_name: str):
        """Ganti model AI. Peringatkan pengguna jika memilih varian yang lebih berat."""
        model_lower = model_name.lower()
        is_heavy = '- balanced' in model_lower or 's -' in model_lower
        
        if is_heavy:
            QMessageBox.warning(
                self,
                "Peringatan Performance",
                f"⚠️ {model_name} adalah model yang lebih berat.\n\n"
                "Ini dapat mengakibatkan kinerja yang lebih lambat pada CPU.\n"
                "Untuk hasil terbaik, gunakan model 'Fast' (nano)."
            )
        
        # Tukar model di thread latar belakang jika detektor sudah dimuat
        if self._detector_service is not None:
            self._load_model_async(model_name)
    
    def _preload_model(self):
        """Load model AI di thread latar belakang saat startup."""
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._load_model_async(model_name)
    
    def _load_model_async(self, model_name: str):
        """Muat model YOLO di thread latar belakang (tidak memblokir UI)."""
        if self._is_loading_model:
            self._status_bar.showMessage("⏳ Model masih dimuat, harap tunggu...")
            return
        
        self._is_loading_model = True
        self._start_btn.setEnabled(False)
        self._model_combo.setEnabled(False)
        self._status_bar.showMessage(f"⏳ Loading AI model: {model_name}...")
        
        self._model_loader_thread = ModelLoaderThread(model_name, self)
        self._model_loader_thread.model_loaded.connect(self._on_model_loaded)
        self._model_loader_thread.start()
    
    def _on_model_loaded(self, success: bool, model_name: str, error: str):
        """Tangani hasil pemuatan model dari thread latar belakang."""
        self._is_loading_model = False
        self._model_combo.setEnabled(True)
        
        if success and self._model_loader_thread:
            self._detector_service = self._model_loader_thread.detector_service
            self._stats_widget.update_model(model_name)
            
            # Aktifkan kembali tombol Start jika ada kamera
            if self._camera_combo.currentData() is not None:
                self._start_btn.setEnabled(True)
            
            self._status_bar.showMessage(f"✓ Model loaded: {model_name}")
        else:
            # Aktifkan kembali Start jika sudah ada model sebelumnya
            if self._detector_service is not None:
                self._start_btn.setEnabled(True)
            
            self._status_bar.showMessage(f"⚠️ Failed to load model: {error}")
            QMessageBox.warning(
                self,
                "Peringatan Model AI",
                f"Gagal memuat model {model_name}:\n\n{error}\n\n"
                "Deteksi mungkin tidak berfungsi. "
                "Coba pilih model lain atau periksa file model."
            )
    
    # =========================================================================
    # Preview & Deteksi
    # =========================================================================
    
    def _start_preview(self):
        """Buka preview kamera (frame mentah, tanpa deteksi AI)."""
        if self._is_previewing or self._is_running:
            return
        
        camera_index = self._camera_combo.currentData()
        if camera_index is None:
            camera_index = 0
        
        camera_name = self._camera_combo.currentText()
        self._video_widget.show_loading(camera_name)
        
        self._video_service.start_capture(camera_index)
        self._is_previewing = True
        self._capture_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        self._status_bar.showMessage("Camera preview - click Start for detection")
    
    def _stop_preview(self):
        """Hentikan preview kamera."""
        if not self._is_previewing:
            return
        
        # Hentikan perekaman aktif apa pun sebelum menutup kamera
        if self._recording_service.is_recording():
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
        
        self._video_service.stop_capture()
        self._is_previewing = False
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._video_widget.clear_display()
    
    def _on_start(self):
        """Aktifkan deteksi AI pada tampilan kamera."""
        if self._is_running:
            return
        
        # Cegah start jika model masih loading
        if self._is_loading_model:
            self._status_bar.showMessage("⏳ Model masih dimuat, harap tunggu...")
            return
        
        self._status_bar.showMessage("Starting detection...")
        
        # Muat model AI jika belum dimuat sebelumnya
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._status_bar.showMessage("Loading AI model...")
            
            # Coba muat secara sinkron jika belum ada sama sekali
            self._detector_service = DetectorService(model_name, use_gpu=False)
            
            if self._detector_service.init_error:
                error = self._detector_service.init_error
                self._detector_service = None
                self._status_bar.showMessage(f"⚠️ Model failed: {error}")
                QMessageBox.warning(
                    self,
                    "Peringatan Model AI",
                    f"Gagal memuat model {model_name}:\n\n{error}\n\n"
                    "Deteksi tidak dapat dimulai."
                )
                return
            
            self._stats_widget.update_model(model_name)
        
        # Buka kamera jika preview belum berjalan
        if not self._is_previewing:
            camera_index = self._camera_combo.currentData()
            if camera_index is None:
                camera_index = 0
            self._video_service.start_capture(camera_index)
            self._is_previewing = True
        
        # Kunci kontrol selama deteksi
        self._is_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._camera_combo.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        
        # Reset low-specs state if needed
        if self._processing_mode == 'low-specs':
            self._lowspec_ready = True
            self._last_annotated_frame = None
        
        self._stats_widget.update_status("Running", True)
        mode_label = "Low-Specs" if self._processing_mode == 'low-specs' else "Streaming"
        self._status_bar.showMessage(f"Detection running ({mode_label} mode)...")
    
    def _on_stop(self):
        """Hentikan deteksi AI tetapi biarkan preview kamera tetap berjalan."""
        if not self._is_running:
            return
        
        # Hentikan perekaman aktif apa pun
        if self._recording_service.is_recording():
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"Recording saved: {saved}")
        
        # Buka kunci kontrol
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        
        # Reset low-specs state
        self._lowspec_ready = True
        self._last_annotated_frame = None
        self._lowspec_timer.stop()
        
        self._stats_widget.update_status("Stopped", False)
        self._stats_widget.reset_stats()
        self._status_bar.showMessage("Detection stopped - camera preview active")
    
    def _on_mode_changed(self, mode_text: str):
        """Handle processing mode change."""
        new_mode = 'low-specs' if mode_text == 'Low-Specs' else 'streaming'
        
        if new_mode == self._processing_mode:
            return
        
        self._processing_mode = new_mode
        
        # Apply FPS settings if running
        if self._is_running:
            if new_mode == 'low-specs':
                self._video_service.set_target_fps(10)
            else:
                self._video_service.set_target_fps(DEFAULT_CAPTURE_FPS)
        
        # Reset low-specs state
        self._lowspec_ready = True
        self._last_annotated_frame = None
        self._frame_times.clear()
        
        if new_mode == 'low-specs':
            self._status_bar.showMessage("Mode: Low-Specs (reduced CPU usage)")
        else:
            self._status_bar.showMessage("Mode: Streaming (real-time processing)")
    
    def _lowspec_mark_ready(self):
        """Mark low-specs mode as ready to process the next frame."""
        self._lowspec_ready = True
    
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Tangani frame baru dari layanan video.
        Video selalu streaming pada kecepatan kamera.
        Deteksi YOLO hanya berjalan sesuai target FPS.
        Pada frame tanpa deteksi, kotak pembatas terakhir digambar ulang.
        """
        if not self._is_previewing and not self._is_running:
            return
        
        display_frame = frame
        
        if self._is_running and self._detector_service is not None:
            current_time = time.time()
            detection_interval = 1.0 / self._video_service.get_target_fps()
            should_detect = (current_time - self._last_detection_time) >= detection_interval
            
            if should_detect:
                # Run YOLO and cache results
                self._last_detection_time = current_time
                annotated_frame, person_count, detections = self._detector_service.detect_humans(frame)
                self._cached_detections = detections
                self._cached_person_count = person_count
                display_frame = annotated_frame
                
                self._stats_widget.update_person_count(person_count)
                
                # Track detection FPS
                if self._last_frame_time > 0:
                    frame_interval = current_time - self._last_frame_time
                    if frame_interval > 0:
                        self._frame_times.append(frame_interval)
                self._last_frame_time = current_time
                
                if current_time - self._last_fps_update >= 0.25:
                    if len(self._frame_times) > 0:
                        avg_interval = sum(self._frame_times) / len(self._frame_times)
                        real_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                        self._stats_widget.update_fps(real_fps)
                    self._last_fps_update = current_time
            elif self._cached_detections:
                # Non-detection frame: redraw cached boxes on live video
                display_frame = self._detector_service._redraw_detections(frame, self._cached_detections)
            # else: no cached detections yet, show raw frame
        
        # Tambahkan ke perekaman jika aktif
        if self._recording_service.is_recording():
            self._recording_service.write_frame(display_frame)
        
        self._video_widget.update_frame(display_frame)
    
    def _on_video_error(self, error: str):
        """Tangani kesalahan kamera."""
        self._video_widget.show_error(error)
        self._status_bar.showMessage(f"⚠️ Camera error: {error}")
        
        # Hentikan perekaman aktif jika ada
        if self._recording_service.is_recording():
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
        
        # Atur ulang semua status
        self._is_previewing = False
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        
        self._stats_widget.update_status("Error", False)
        self._stats_widget.reset_stats()
        
        QMessageBox.warning(
            self,
            "Peringatan Kamera",
            f"Terjadi kesalahan kamera:\n\n{error}\n\n"
            "Silakan periksa koneksi kamera dan coba klik Refresh."
        )
    
    # =========================================================================
    # Penangan Perekaman / Screenshot
    # =========================================================================
    
    def _update_folder_tooltip(self):
        """Refresh tooltip tombol folder."""
        path = self._recording_service.get_output_folder()
        self._folder_btn.setToolTip(
            f"📂 {path}\n\n"
            "Klik kiri: Buka folder di Explorer\n"
            "Klik kanan: Pilih folder output baru"
        )
    
    def _on_folder_open(self):
        """Buka folder output di Windows Explorer."""
        try:
            self._recording_service.open_output_folder()
        except Exception as e:
            self._status_bar.showMessage(f"⚠️ Gagal membuka folder: {e}")
            QMessageBox.warning(self, "Peringatan", f"Gagal membuka folder output:\n\n{e}")
    
    def _on_folder_select(self, pos):
        """Menu konteks klik kanan: pilih folder output baru."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2e;
                border: 1px solid #2d2d44;
                color: #ffffff;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #00d9ff;
                color: #000000;
            }
        """)
        select_action = menu.addAction("📁 Pilih folder output...")
        action = menu.exec_(self._folder_btn.mapToGlobal(pos))
        
        if action == select_action:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Pilih folder output",
                self._recording_service.get_output_folder()
            )
            if folder:
                self._recording_service.set_output_folder(folder)
                self._update_folder_tooltip()
                self._status_bar.showMessage(f"Output folder: {folder}")
    
    def _on_capture(self):
        """Simpan frame yang ditampilkan saat ini sebagai screenshot PNG."""
        frame = self._video_widget._current_frame
        if frame is None:
            self._status_bar.showMessage("⚠️ Tidak ada frame untuk ditangkap")
            QMessageBox.warning(
                self,
                "Peringatan Screenshot",
                "Tidak ada frame aktif untuk ditangkap.\n"
                "Pastikan kamera sedang berjalan."
            )
            return
        
        try:
            path = self._recording_service.capture_screenshot(frame)
            self._status_bar.showMessage(f"✓ Screenshot disimpan: {path}")
        except Exception as e:
            self._status_bar.showMessage(f"⚠️ Screenshot gagal: {e}")
            QMessageBox.warning(
                self,
                "Peringatan Screenshot",
                f"Gagal menyimpan screenshot:\n\n{e}\n\n"
                "Periksa folder output dan ruang disk."
            )
    
    def _on_record_toggle(self):
        """Mulai atau hentikan perekaman video."""
        if self._recording_service.is_recording():
            # Hentikan perekaman
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"✓ Recording saved: {saved}")
        else:
            # Mulai perekaman
            frame = self._video_widget._current_frame
            if frame is None:
                self._status_bar.showMessage("⚠️ No frame available — start camera first")
                QMessageBox.warning(
                    self,
                    "Peringatan Perekaman",
                    "Tidak ada frame aktif untuk direkam.\n"
                    "Pastikan kamera sedang berjalan."
                )
                return
            
            try:
                h, w = frame.shape[:2]
                path = self._recording_service.start_recording(w, h)
                self._record_btn.setText("⏹ Recording..." if not self._compact_mode else "⏹")
                self._record_btn.setStyleSheet(styles.get_button_style("#ff4757", "#ff3344"))
                self._status_bar.showMessage(f"Recording to: {path}")
            except Exception as e:
                self._status_bar.showMessage(f"⚠️ Record failed: {e}")
                QMessageBox.warning(
                    self,
                    "Peringatan Perekaman",
                    f"Gagal memulai perekaman:\n\n{e}\n\n"
                    "Periksa folder output dan ruang disk."
                )
    
    # =========================================================================
    # Window Events & Tata Letak Responsif
    # =========================================================================
    
    def _update_button_labels(self):
        """Beralih antara label tombol hanya icon dan icon+teks."""
        if self._compact_mode:
            icon_font = QFont()
            icon_font.setPixelSize(18)
            for btn in (self._record_btn, self._start_btn, self._stop_btn):
                btn.setFont(icon_font)
            
            if self._recording_service.is_recording():
                self._record_btn.setText("⏹")
            else:
                self._record_btn.setText("⏺")
            self._start_btn.setText("▶")
            self._stop_btn.setText("⏹")
        else:
            default_font = QFont()
            default_font.setPixelSize(14)
            for btn in (self._record_btn, self._start_btn, self._stop_btn):
                btn.setFont(default_font)
            
            if self._recording_service.is_recording():
                self._record_btn.setText("⏹ Recording...")
            else:
                self._record_btn.setText("⏺ Record")
            self._start_btn.setText("▶ Start")
            self._stop_btn.setText("⏹ Stop")
    
    def resizeEvent(self, event):
        """Beralih ke tata letak kompak saat lebar jendela turun di bawah 900px."""
        super().resizeEvent(event)
        compact = self.width() < 900
        if compact != self._compact_mode:
            self._compact_mode = compact
            self._update_button_labels()
    
    def closeEvent(self, event):
        """Bersihkan sumber daya saat keluar."""
        self._lowspec_timer.stop()
        self._recording_service.cleanup()
        if self._is_running or self._is_previewing:
            self._video_service.stop_capture()
        event.accept()
    
    # =========================================================================
    # Settings Dialog
    # =========================================================================
    
    def _on_settings_open(self):
        """Open the performance settings dialog."""
        dialog = SettingsDialog(
            parent=self,
            video_service=self._video_service,
            detector_service=self._detector_service,
            video_widget=self._video_widget
        )
        dialog.exec_()
    
    def _on_settings_applied(self, fps: int):
        """Callback when settings are applied from the dialog."""
        self._stats_widget.update_target_fps(fps)
        self._status_bar.showMessage(f"Settings applied — Target FPS: {fps}")
