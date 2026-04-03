"""
Main Application Window
Real-time person detection and counting using YOLO models.
Handles UI layout, camera management, recording, and detection orchestration.
"""

import os
import time
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QFrame,
    QSplitter, QStatusBar, QMessageBox, QFileDialog, QMenu, QAction,
    QDialog, QSlider, QSpinBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from services import CameraService, VideoService, DetectorService, RecordingService
from widgets import VideoWidget, StatsWidget
from utils.constants import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, YOLO_MODELS,
    DEFAULT_CAPTURE_FPS, MIN_FPS, MAX_FPS
)
from utils import styles


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
        # Set scale to Half (0.5x) — index 2
        self._scale_combo.setCurrentIndex(2)
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


class MainWindow(QMainWindow):
    """Main application window with camera preview, AI detection, and recording."""
    
    def __init__(self):
        super().__init__()
        
        # Core services
        self._camera_service = CameraService()       # Scans for available cameras
        self._video_service = VideoService()         # Captures frames in background thread
        self._recording_service = RecordingService() # Saves video/screenshots to disk
        self._detector_service = None                # YOLO model (pre-loaded after UI init)
        
        # State flags
        self._is_running = False      # True while AI detection is active
        self._is_previewing = False   # True while camera preview is active (no detection)
        self._current_camera = 0      # Index of the currently selected camera
        self._compact_mode = False    # True when window < 900px (show icon-only buttons)
        
        # FPS tracking (measures actual frame-to-frame intervals)
        self._frame_times = deque(maxlen=30)  # Last 30 frame intervals
        self._last_frame_time = 0              # Timestamp of last frame received
        self._last_fps_update = 0              # Throttle FPS display updates
        
        # Build UI, wire signals, scan cameras, and pre-load the AI model
        self._init_ui()
        self._connect_signals()
        self._refresh_cameras()
        QTimer.singleShot(500, self._preload_model)
    
    def _init_ui(self):
        """Build window layout: control bar, video panel, stats panel, status bar."""
        # Window setup
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(640, 480)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # App icon (loaded from assets/)
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        

        
        self.setStyleSheet(styles.get_main_theme())
        
        # Root layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Top control bar (camera/model selectors + action buttons)
        control_bar = self._create_control_bar()
        main_layout.addWidget(control_bar)
        
        # Resizable split: video (left) + stats (right)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(styles.get_splitter_handle_style())
        
        # Left: live video feed
        self._video_widget = VideoWidget()
        splitter.addWidget(self._video_widget)
        
        # Right: detection statistics
        self._stats_widget = StatsWidget()
        splitter.addWidget(self._stats_widget)
        
        # Default split: 75% video, 25% stats
        splitter.setSizes([750, 250])
        main_layout.addWidget(splitter)
        
        # Bottom status bar
        self._status_bar = QStatusBar()

        self._status_bar.setStyleSheet(styles.get_status_bar_style())
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready - Select a camera and click Start")
    

    def _create_control_bar(self) -> QWidget:
        """Build the top bar: camera selector, model selector, and action buttons."""
        control_bar = QFrame()
        control_bar.setStyleSheet(styles.get_control_bar_style())
        control_bar.setMaximumHeight(55)
        
        layout = QHBoxLayout(control_bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(12)
        
        # 1. Refresh Button
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
                background-color: #5a5a7a;
            }
        """)
        layout.addWidget(self._refresh_btn)
        
        # 2. Camera & Model Selectors
        self._create_camera_controls(layout)
        
        layout.addStretch()
        
        # 3. Recording Controls
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
        """Add camera and model dropdowns to the layout."""
        # Camera dropdown
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
        
        # AI model dropdown
        model_layout = QHBoxLayout()
        model_layout.setSpacing(5)
        model_label = QLabel("🧠")
        model_label.setStyleSheet("color: #ffffff; font-size: 14px;")
        
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(150)
        self._model_combo.setStyleSheet(styles.get_combo_style())
        
        # Populate with available YOLO models
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

    def _create_recording_controls(self, parent_layout: QHBoxLayout):
        """Add folder, screenshot, and record buttons to the layout."""
        # Folder button
        self._folder_btn = QPushButton("📂")
        self._folder_btn.setFixedSize(32, 32)
        self._folder_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#5a5a7a"))
        self._folder_btn.setContextMenuPolicy(Qt.CustomContextMenu)
        self._update_folder_tooltip()
        parent_layout.addWidget(self._folder_btn)
        
        # Screenshot button
        self._capture_btn = QPushButton("📸")
        self._capture_btn.setToolTip("Capture screenshot")
        self._capture_btn.setFixedSize(32, 32)
        self._capture_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#ffa502"))
        self._capture_btn.setEnabled(False)
        parent_layout.addWidget(self._capture_btn)
        
        # Record toggle button
        self._record_btn = QPushButton("⏺ Record")
        self._record_btn.setToolTip("Start / stop video recording")
        self._record_btn.setMinimumSize(32, 32)
        self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
        self._record_btn.setEnabled(False)
        parent_layout.addWidget(self._record_btn)
    

    
    def _connect_signals(self):
        """Wire widget signals to handler methods."""
        # Control bar buttons
        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)
        self._refresh_btn.clicked.connect(self._refresh_cameras)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        self._settings_btn.clicked.connect(self._on_settings_open)
        
        # Recording & capture buttons
        self._folder_btn.clicked.connect(self._on_folder_open)
        self._folder_btn.customContextMenuRequested.connect(self._on_folder_select)
        self._capture_btn.clicked.connect(self._on_capture)
        self._record_btn.clicked.connect(self._on_record_toggle)
        
        # Video capture callbacks
        self._video_service.frame_ready.connect(self._on_frame_ready)
        self._video_service.error_occurred.connect(self._on_video_error)
    
    def _refresh_cameras(self):
        """Scan for available cameras and populate the dropdown."""
        self._status_bar.showMessage("Scanning for cameras...")
        
        cameras = self._camera_service.get_available_cameras()
        self._camera_combo.clear()
        
        if not cameras:
            self._camera_combo.addItem("No cameras found")
            self._start_btn.setEnabled(False)
            self._video_widget.show_no_camera()
            self._status_bar.showMessage("No cameras detected")
        else:
            for camera in cameras:
                display_text = f"{camera['name']} ({camera['resolution'][0]}x{camera['resolution'][1]})"
                # Store camera index as item data for later retrieval
                self._camera_combo.addItem(display_text, camera['index'])
            
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"Found {len(cameras)} camera(s)")
            
            # Auto-open camera preview (detection not started yet)
            QTimer.singleShot(100, self._start_preview)
    
    def _on_camera_changed(self, index: int):
        """Switch to newly selected camera, preserving the current mode (preview or detection)."""
        if index >= 0:
            camera_index = self._camera_combo.itemData(index)
            if camera_index is not None:
                self._current_camera = camera_index
                
                # Stop current feed, then restart in the same mode
                was_detecting = self._is_running
                if self._is_running:
                    self._on_stop()
                if self._is_previewing:
                    self._stop_preview()
                
                if was_detecting:
                    QTimer.singleShot(200, self._on_start)
                else:
                    QTimer.singleShot(200, self._start_preview)
    
    def _on_model_changed(self, model_name: str):
        """Switch AI model. Warns user if selecting a heavier (slower) variant."""
        # Show performance warning for non-nano models
        model_lower = model_name.lower()
        is_heavy = '- balanced' in model_lower or 's -' in model_lower
        
        if is_heavy:
            QMessageBox.warning(
                self,
                "Performance Warning",
                f"⚠️ {model_name} is a heavier model.\n\n"
                "This may result in slower performance on CPU.\n"
                "For best results, use a 'Fast' (nano) model."
            )
        
        # Hot-swap model if detector is already loaded
        if self._detector_service is not None:
            self._detector_service.load_model(model_name, use_gpu=False)
            self._stats_widget.update_model(model_name)
            self._status_bar.showMessage(f"Loaded model: {model_name}")
    
    def _start_preview(self):
        """Open camera feed (raw frames, no AI detection)."""
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
    
    def _preload_model(self):
        """Pre-load AI model during startup so clicking Start is instant."""
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._status_bar.showMessage("Pre-loading AI model...")
            self._detector_service = DetectorService(model_name, use_gpu=False)
            self._stats_widget.update_model(model_name)
            self._status_bar.showMessage("Camera preview - click Start for detection")
    
    def _stop_preview(self):
        """Stop camera feed and release resources."""
        if not self._is_previewing:
            return
        
        # Stop any active recording before closing camera
        if self._recording_service.is_recording():
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"Recording saved: {saved}")
        
        self._video_service.stop_capture()
        self._is_previewing = False
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._video_widget.clear_display()
    
    def _on_start(self):
        """Enable AI detection on the camera feed."""
        if self._is_running:
            return
        
        self._status_bar.showMessage("Starting detection...")
        
        # Load AI model if not already pre-loaded
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._status_bar.showMessage("Loading AI model...")
            self._detector_service = DetectorService(model_name, use_gpu=False)
            self._stats_widget.update_model(model_name)
        
        # Open camera if preview isn't already running
        if not self._is_previewing:
            camera_index = self._camera_combo.currentData()
            if camera_index is None:
                camera_index = 0
            self._video_service.start_capture(camera_index)
            self._is_previewing = True
        
        # Lock controls during detection
        self._is_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._camera_combo.setEnabled(False)
        self._capture_btn.setEnabled(True)
        self._record_btn.setEnabled(True)
        
        self._stats_widget.update_status("Running", True)
        self._status_bar.showMessage("Detection running...")
    
    def _on_stop(self):
        """Stop AI detection but keep camera preview running."""
        if not self._is_running:
            return
        
        # Stop any active recording
        if self._recording_service.is_recording():
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"Recording saved: {saved}")
        
        # Unlock controls (camera stays open for preview)
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        
        self._stats_widget.update_status("Stopped", False)
        self._stats_widget.reset_stats()
        self._status_bar.showMessage("Detection stopped - camera preview active")
    
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Handle new frame from video service.
        In preview mode: just display the raw frame.
        In detection mode: run YOLO and display annotated frame.
        """
        if not self._is_previewing and not self._is_running:
            return
        
        # When detection is active, run YOLO and overlay bounding boxes
        if self._is_running and self._detector_service is not None:
            annotated_frame, person_count, detections = self._detector_service.detect_humans(frame)
            
            # Track actual frame rate (time between consecutive frames)
            current_time = time.time()
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
            
            display_frame = annotated_frame
            self._stats_widget.update_person_count(person_count)
        else:
            # Preview only: display raw camera frame
            display_frame = frame
        
        # Append to recording if active
        if self._recording_service.is_recording():
            self._recording_service.write_frame(display_frame)
        
        self._video_widget.update_frame(display_frame)
    
    def _on_video_error(self, error: str):
        """Handle camera errors — show error message and reset UI."""
        self._video_widget.show_error(error)
        self._status_bar.showMessage(f"Camera error: {error}")
        
        # Reset all state flags and re-enable controls
        self._is_previewing = False
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._on_stop()
    
    # =========================================================================
    # Recording / Capture Handlers
    # =========================================================================
    
    def _update_folder_tooltip(self):
        """Refresh the folder button tooltip with the current output path."""
        path = self._recording_service.get_output_folder()
        self._folder_btn.setToolTip(
            f"📂 {path}\n\n"
            "Left-click: Open folder in Explorer\n"
            "Right-click: Select output folder"
        )
    
    def _on_folder_open(self):
        """Open the output folder in Windows Explorer."""
        self._recording_service.open_output_folder()
    
    def _on_folder_select(self, pos):
        """Right-click context menu: pick a new output folder."""
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
        select_action = menu.addAction("📁 Select output folder...")
        action = menu.exec_(self._folder_btn.mapToGlobal(pos))
        
        if action == select_action:
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Output Folder",
                self._recording_service.get_output_folder()
            )
            if folder:
                self._recording_service.set_output_folder(folder)
                self._update_folder_tooltip()
                self._status_bar.showMessage(f"Output folder: {folder}")
    
    def _on_capture(self):
        """Save the current displayed frame as a PNG screenshot."""
        frame = self._video_widget._current_frame
        if frame is None:
            self._status_bar.showMessage("No frame to capture")
            return
        
        try:
            path = self._recording_service.capture_screenshot(frame)
            self._status_bar.showMessage(f"Screenshot saved: {path}")
        except Exception as e:
            self._status_bar.showMessage(f"Capture failed: {e}")
    
    def _on_record_toggle(self):
        """Start or stop video recording."""
        if self._recording_service.is_recording():
            # Stop recording
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"Recording saved: {saved}")
        else:
            # Start recording with current frame dimensions
            frame = self._video_widget._current_frame
            if frame is None:
                self._status_bar.showMessage("No frame available — start camera first")
                return
            
            try:
                h, w = frame.shape[:2]
                path = self._recording_service.start_recording(w, h)
                self._record_btn.setText("⏹ Recording..." if not self._compact_mode else "⏹")
                self._record_btn.setStyleSheet(styles.get_button_style("#ff4757", "#ff3344"))
                self._status_bar.showMessage(f"Recording to: {path}")
            except Exception as e:
                self._status_bar.showMessage(f"Record failed: {e}")
    
    # =========================================================================
    # Window Events & Responsive Layout
    # =========================================================================
    
    def _update_button_labels(self):
        """Switch between icon-only (compact) and icon+text (normal) button labels."""
        if self._compact_mode:
            # Compact: icons only with larger font
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
            # Normal: full text labels with standard font
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
        """Switch to compact layout when window width drops below 900px."""
        super().resizeEvent(event)
        compact = self.width() < 900
        if compact != self._compact_mode:
            self._compact_mode = compact
            self._update_button_labels()
    
    def closeEvent(self, event):
        """Clean up recording and camera resources on exit."""
        self._recording_service.cleanup()
        if self._is_running or self._is_previewing:
            self._video_service.stop_capture()
        event.accept()
    
    # =========================================================================
    # Settings
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
