"""
Aplikasi Utama
Deteksi dan penghitungan orang secara real-time menggunakan model YOLO.
Menangani tata letak UI, manajemen kamera, perekaman, dan proses deteksi.
"""

import os
import time
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QFrame,
    QSplitter, QStatusBar, QMessageBox, QFileDialog, QMenu, QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon

from services import CameraService, VideoService, DetectorService, RecordingService
from widgets import VideoWidget, StatsWidget
from utils.constants import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT, YOLO_MODELS
)
from utils import styles


class MainWindow(QMainWindow):
    """Aplikasi utama dengan pratinjau kamera, deteksi AI, dan perekaman."""
    
    def __init__(self):
        super().__init__()
        
        # Layanan inti
        self._camera_service = CameraService()       # Memindai kamera yang tersedia
        self._video_service = VideoService()         # Menangkap frame di thread latar belakang
        self._recording_service = RecordingService() # Menyimpan video/tangkapan layar ke disk
        self._detector_service = None                # Model YOLO (dimuat sebelumnya setelah inisialisasi UI)
        
        # Bendera status
        self._is_running = False      # True saat deteksi AI aktif
        self._is_previewing = False   # True saat pratinjau kamera aktif (tanpa deteksi)
        self._current_camera = 0      # Indeks kamera yang dipilih saat ini
        self._compact_mode = False    # True saat jendela < 900px (tampilkan tombol ikon saja)
        
        # Pelacakan FPS (rata-rata waktu pemrosesan deteksi)
        self._frame_times = deque(maxlen=30)   # 30 frame terakhir
        self._last_fps_update = 0              # Batasi pembaruan tampilan FPS
        
        # Inisiasi UI, hubungkan sinyal, pindai kamera, dan load model AI
        self._init_ui()
        self._connect_signals()
        self._refresh_cameras()
        QTimer.singleShot(500, self._preload_model)
    
    def _init_ui(self):
        """Tata letak window utama: bilah kontrol, panel video, panel statistik, bilah status."""
        # Pengaturan window utama
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(640, 480)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Icon aplikasi (dimuat dari assets/)
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
        
        # Bilah kontrol atas (pemilih kamera/model + tombol tindakan)
        control_bar = self._create_control_bar()
        main_layout.addWidget(control_bar)
        
        # Pemisah yang dapat diubah ukurannya: video (kiri) + statistik (kanan)
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
                background-color: #5a5a7a;
            }
        """)
        layout.addWidget(self._refresh_btn)
        
        # 2. Pemilih Kamera & Model
        self._create_camera_controls(layout)
        
        layout.addStretch()
        
        # 3. Kontrol Perekaman
        self._create_recording_controls(layout)
        
        # Pemisah vertikal
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: #2d2d44;")
        separator.setFixedHeight(24)
        layout.addWidget(separator)
        
        # 4. Tombol Mulai/Berhenti
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
        self._capture_btn.setStyleSheet(styles.get_icon_button_style("#4a4a6a", "#ffa502"))
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
        
        # Tombol rekam & screenshot
        self._folder_btn.clicked.connect(self._on_folder_open)
        self._folder_btn.customContextMenuRequested.connect(self._on_folder_select)
        self._capture_btn.clicked.connect(self._on_capture)
        self._record_btn.clicked.connect(self._on_record_toggle)
        
        # Callback penangkapan video
        self._video_service.frame_ready.connect(self._on_frame_ready)
        self._video_service.error_occurred.connect(self._on_video_error)
    
    def _refresh_cameras(self):
        """Pindai kamera yang tersedia dan isi dropdown."""
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
                # Simpan indeks kamera sebagai data item untuk pengambilan nanti
                self._camera_combo.addItem(display_text, camera['index'])
            
            self._start_btn.setEnabled(True)
            self._status_bar.showMessage(f"Found {len(cameras)} camera(s)")
            
            # Buka preview kamera secara otomatis (deteksi belum dimulai)
            QTimer.singleShot(100, self._start_preview)
    
    def _on_camera_changed(self, index: int):
        """Beralih ke kamera yang baru dipilih, pertahankan mode saat ini (pratinjau atau deteksi)."""
        if index >= 0:
            camera_index = self._camera_combo.itemData(index)
            if camera_index is not None:
                self._current_camera = camera_index
                
                # Hentikan deteksi saat ini, lalu mulai ulang dalam mode yang sama
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
        """Ganti model AI. Peringatkan pengguna jika memilih varian yang lebih berat (lebih lambat)."""
        # Tampilkan peringatan kinerja untuk model non-nano
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
        
        # Tukar model secara langsung jika detektor sudah dimuat
        if self._detector_service is not None:
            self._detector_service.load_model(model_name, use_gpu=False)
            self._stats_widget.update_model(model_name)
            self._status_bar.showMessage(f"Loaded model: {model_name}")
    
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
    
    def _preload_model(self):
        """Load model AI saat startup agar klik Start instan."""
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._status_bar.showMessage("Pre-loading AI model...")
            self._detector_service = DetectorService(model_name, use_gpu=False)
            self._stats_widget.update_model(model_name)
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
            self._status_bar.showMessage(f"Recording saved: {saved}")
        
        self._video_service.stop_capture()
        self._is_previewing = False
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._video_widget.clear_display()
    
    def _on_start(self):
        """Aktifkan deteksi AI pada tampilan kamera."""
        if self._is_running:
            return
        
        self._status_bar.showMessage("Starting detection...")
        
        # Muat model AI jika belum dimuat sebelumnya
        if self._detector_service is None:
            model_name = self._model_combo.currentText()
            self._status_bar.showMessage("Loading AI model...")
            self._detector_service = DetectorService(model_name, use_gpu=False)
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
        
        self._stats_widget.update_status("Running", True)
        self._status_bar.showMessage("Detection running...")
    
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
        
        # Buka kunci kontrol (kamera tetap terbuka untuk preview)
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        
        self._stats_widget.update_status("Stopped", False)
        self._stats_widget.reset_stats()
        self._status_bar.showMessage("Detection stopped - camera preview active")
    
    def _on_frame_ready(self, frame: np.ndarray):
        """
        Tangani frame baru dari layanan video.
        Dalam mode preview: hanya tampilkan frame mentah.
        Dalam mode deteksi: jalankan YOLO dan tampilkan frame terdeteksi.
        """
        if not self._is_previewing and not self._is_running:
            return
        
        # Saat deteksi aktif, jalankan YOLO dan overlay kotak pembatas
        if self._is_running and self._detector_service is not None:
            start_time = time.time()
            annotated_frame, person_count, detections = self._detector_service.detect_humans(frame)
            
            current_time = time.time()
            processing_time = current_time - start_time
            if processing_time > 0:
                self._frame_times.append(processing_time)
            
            if current_time - self._last_fps_update >= 0.25:
                if len(self._frame_times) > 0:
                    avg_time = sum(self._frame_times) / len(self._frame_times)
                    real_fps = 1.0 / avg_time if avg_time > 0 else 0
                    self._stats_widget.update_fps(real_fps)
                self._last_fps_update = current_time
            
            display_frame = annotated_frame
            self._stats_widget.update_person_count(person_count)
        else:
            # Preview saja: tampilkan frame kamera mentah
            display_frame = frame
        
        # Tambahkan ke perekaman jika aktif
        if self._recording_service.is_recording():
            self._recording_service.write_frame(display_frame)
        
        self._video_widget.update_frame(display_frame)
    
    def _on_video_error(self, error: str):
        """Tangani kesalahan kamera — tampilkan pesan kesalahan dan atur ulang UI."""
        self._video_widget.show_error(error)
        self._status_bar.showMessage(f"Camera error: {error}")
        
        # Atur ulang semua status dan aktifkan kembali kontrol
        self._is_previewing = False
        self._is_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._camera_combo.setEnabled(True)
        self._capture_btn.setEnabled(False)
        self._record_btn.setEnabled(False)
        self._on_stop()
    
    # =========================================================================
    # Penangan Perekaman / Screenshot
    # =========================================================================
    
    def _update_folder_tooltip(self):
        """Refresh tooltip tombol folder dengan path output saat ini."""
        path = self._recording_service.get_output_folder()
        self._folder_btn.setToolTip(
            f"📂 {path}\n\n"
            "Klik kiri: Buka folder di Explorer\n"
            "Klik kanan: Pilih folder output baru"
        )
    
    def _on_folder_open(self):
        """Buka folder output di Windows Explorer."""
        self._recording_service.open_output_folder()
    
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
            self._status_bar.showMessage("Tidak ada frame untuk ditangkap")
            return
        
        try:
            path = self._recording_service.capture_screenshot(frame)
            self._status_bar.showMessage(f"Screenshot disimpan: {path}")
        except Exception as e:
            self._status_bar.showMessage(f"Screenshot gagal: {e}")
    
    def _on_record_toggle(self):
        """Mulai atau hentikan perekaman video."""
        if self._recording_service.is_recording():
            # Hentikan perekaman
            saved = self._recording_service.stop_recording()
            self._record_btn.setText("⏺ Record" if not self._compact_mode else "⏺")
            self._record_btn.setStyleSheet(styles.get_button_style("#4a4a6a", "#ff4757"))
            self._status_bar.showMessage(f"Recording saved: {saved}")
        else:
            # Mulai perekaman dengan dimensi frame saat ini
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
    # Window Events & Tata Letak Responsif
    # =========================================================================
    
    def _update_button_labels(self):
        """Beralih antara label tombol hanya icon dan icon+teks (normal)."""
        if self._compact_mode:
            # hanya icon dengan font lebih besar
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
            # Normal: label teks lengkap dengan font standar
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
        """Bersihkan sumber daya perekaman dan kamera saat keluar."""
        self._recording_service.cleanup()
        if self._is_running or self._is_previewing:
            self._video_service.stop_capture()
        event.accept()
