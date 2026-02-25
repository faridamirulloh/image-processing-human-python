"""
Widget Video - Menampilkan umpan kamera / deteksi langsung.
Menangani rendering frame, penskalaan rasio aspek, dan pesan status.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap


class VideoWidget(QLabel):
    """
    Menampilkan frame video langsung di dalam QLabel bergaya.
    Secara otomatis menskalakan frame agar pas sambil mempertahankan rasio aspek.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
            }
        """)
        
        self._current_frame = None   # Frame terbaru (disimpan untuk tangkapan layar & ubah ukuran)
        self._is_active = False      # Benar saat menampilkan frame video
        
        self.show_no_camera()
    
    # =========================================================================
    # Pesan Status (ditampilkan saat tidak ada video yang diputar)
    # =========================================================================
    
    def show_no_camera(self):
        """Tampilkan placeholder saat tidak ada kamera yang terhubung"""
        self._is_active = False
        self._current_frame = None
        self.setText("📷  No Camera Connected\n\nPlease connect a camera and click Refresh")
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                color: #8b8b8b;
                font-size: 16px;
                padding: 20px;
            }
        """)
    
    def show_loading(self, camera_name: str = ""):
        """Tampilkan status memuat saat kamera terhubung"""
        self._is_active = False
        self._current_frame = None
        label = f"⏳  Connecting to camera...\n\n{camera_name}" if camera_name else "⏳  Connecting to camera..."
        self.setText(label)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #ffa502;
                border-radius: 8px;
                color: #ffa502;
                font-size: 16px;
                padding: 20px;
            }
        """)
    
    def show_error(self, message: str):
        """Tampilkan pesan kesalahan saat kamera gagal"""
        self._is_active = False
        self._current_frame = None
        self.setText(f"❌  Camera Error\n\n{message}")
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #ff4757;
                border-radius: 8px;
                color: #ff4757;
                font-size: 16px;
                padding: 20px;
            }
        """)
    
    # =========================================================================
    # Tampilan Frame
    # =========================================================================
    
    def update_frame(self, frame: np.ndarray):
        """
        Tampilkan frame video baru.
        
        Args:
            frame: BGR frame from OpenCV
        """
        if frame is None:
            return
        
        # Hindari crash saat widget belum memiliki ukuran (saat layout awal)
        if self.width() <= 0 or self.height() <= 0:
            return
            
        self._current_frame = frame
        
        # Hanya perbarui gaya saat beralih dari status tidak aktif ke aktif
        if not self._is_active:
            self._is_active = True
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a1a2e;
                    border: 2px solid #00d9ff;
                    border-radius: 8px;
                }
            """)
        
        # Konversi BGR (OpenCV) → RGB (Qt) untuk warna yang benar
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Buat gambar Qt dan skalakan agar pas dengan widget (mempertahankan rasio aspek)
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def clear_display(self):
        """Hapus video dan tampilkan placeholder tanpa kamera"""
        self._current_frame = None
        self.show_no_camera()
    
    def is_active(self) -> bool:
        """Periksa apakah video sedang ditampilkan"""
        return self._is_active
    
    # =========================================================================
    # Override Qt
    # =========================================================================
    
    def resizeEvent(self, event):
        """Ubah skala frame saat ini saat widget diubah ukurannya"""
        super().resizeEvent(event)
        if self._current_frame is not None:
            # Kami memanggil update_frame untuk mengubah skala, tetapi tidak perlu mengatur ulang gaya
            # logika di update_frame menanganinya secara efisien
            self.update_frame(self._current_frame)
    
    def sizeHint(self) -> QSize:
        """Ukuran yang disarankan default"""
        return QSize(640, 480)
