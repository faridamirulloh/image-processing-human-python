"""
Layanan Video - Menangani penangkapan video dari kamera di thread terpisah.
Memancarkan frame melalui sinyal PyQt untuk pemrosesan dan tampilan.
"""

import cv2
import time
import numpy as np
from typing import Optional
from PyQt5.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from utils.constants import DEFAULT_CAPTURE_FPS


class VideoService(QThread):
    """
    Layanan penangkapan video berjalan di thread terpisah.
    Memancarkan frame untuk tampilan dan pemrosesan.
    """
    
    # Sinyal untuk komunikasi asinkron dengan thread utama
    frame_ready = pyqtSignal(np.ndarray)  # Memancarkan frame kamera mentah
    error_occurred = pyqtSignal(str)       # Memancarkan pesan kesalahan
    capture_started = pyqtSignal()         # Memancarkan saat penangkapan dimulai
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_index = 0
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._mutex = QMutex()
        self._target_fps = DEFAULT_CAPTURE_FPS
        self._requested_resolution = None  # (width, height) atau None
    
    def _open_camera(self, index: int) -> Optional[cv2.VideoCapture]:
        """
        Coba buka kamera dengan backend berbeda.
        DirectShow lebih aman di Windows untuk kompatibilitas yang lebih baik.
        
        Args:
            index: Camera index
            
        Returns:
            VideoCapture object or None if failed
        """
        # Coba backend dalam urutan preferensi untuk Windows
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "Default"),
        ]
        
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                
                if cap.isOpened():
                    # Waktu pemanasan kamera (penting untuk beberapa kamera)
                    time.sleep(0.5)
                    
                    # Buang beberapa frame pertama (sering rusak atau hitam)
                    for _ in range(5):
                        cap.read()
                    
                    # Verifikasi kita benar-benar bisa membaca frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ Camera {index} opened with {name} backend")
                        return cap
                    else:
                        cap.release()
                else:
                    cap.release()
                    
            except Exception as e:
                print(f"✗ Failed to open camera {index} with {name}: {e}")
                continue
        
        return None
    
    def set_target_fps(self, fps: int):
        """
        Set target FPS untuk penangkapan frame (berlaku pada frame berikutnya).
        
        Args:
            fps: Target FPS (di-clamp ke 1-60)
        """
        self._target_fps = max(1, min(fps, 60))
    
    def get_target_fps(self) -> int:
        """Dapatkan target FPS saat ini."""
        return self._target_fps
    
    def set_camera_resolution(self, width: int, height: int):
        """
        Minta resolusi kamera tertentu (diterapkan saat kamera dibuka).
        
        Args:
            width: Lebar yang diminta
            height: Tinggi yang diminta
        """
        self._requested_resolution = (width, height)
    
    def start_capture(self, camera_index: Optional[int] = None) -> bool:
        """
        Mulai penangkapan video.
        
        Args:
            camera_index: Camera index to capture from
            
        Returns:
            True if capture thread started
        """
        # Cegah crash jika thread sudah berjalan
        if self.isRunning():
            print("Warning: Capture thread already running, ignoring duplicate start")
            return True
        
        if camera_index is not None:
            self._camera_index = camera_index
            
        self._running = True
        self.start()  # Mulai QThread
        return True
    
    def stop_capture(self):
        """Hentikan penangkapan video dengan baik tanpa memblokir UI."""
        self._running = False
        
        # Tunggu thread selesai, dengan timeout yang aman
        if self.isRunning():
            if not self.wait(3000):  # Tunggu hingga 3 detik
                print("Warning: Capture thread not responding, forcing termination")
                self.terminate()  # Paksa hentikan hanya jika benar-benar macet
                self.wait(500)
        
        # Pastikan kamera dilepaskan
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
    
    def run(self):
        """Loop penangkapan utama - berjalan di thread terpisah"""
        # Buka kamera dengan backend terbaik yang tersedia
        self._capture = self._open_camera(self._camera_index)
        
        if self._capture is None or not self._capture.isOpened():
            self.error_occurred.emit(
                f"Tidak dapat membuka kamera {self._camera_index}.\n"
                "Silahkan cek:\n"
                "• Tidak ada aplikasi lain yang menggunakan kamera\n"
                "• Privasi kamera diaktifkan di Windows Settings\n"
                "• Driver kamera sudah diperbarui"
            )
            return
        
        # Kurangi ukuran buffer untuk latensi lebih rendah
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass  # Beberapa kamera tidak mendukung ini
        
        # Terapkan resolusi yang diminta jika diatur
        if self._requested_resolution is not None:
            w, h = self._requested_resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        self.capture_started.emit()
        
        consecutive_failures = 0
        max_failures = 30
        
        while self._running:
            # Akses kamera aman thread
            with QMutexLocker(self._mutex):
                if self._capture is None or not self._capture.isOpened():
                    break
                ret, frame = self._capture.read()
            
            if ret and frame is not None:
                consecutive_failures = 0
                self.frame_ready.emit(frame)
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.error_occurred.emit("Camera disconnected or stopped responding.")
                    break
                time.sleep(0.05)  # Jeda singkat sebelum coba lagi
                continue
        
        # Pembersihan saat keluar
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def is_running(self) -> bool:
        """Periksa apakah penangkapan sedang berjalan"""
        return self._running
