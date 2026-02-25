"""
Layanan Video - Menangani penangkapan video dari kamera di thread terpisah.
Memancarkan frame melalui sinyal PyQt untuk pemrosesan dan tampilan.
"""

import cv2
import time
import numpy as np
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


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
        self._camera_index: int = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._running: bool = False
        self._mutex = QMutex()  # Keamanan thread untuk akses kamera
        self._target_fps: int = 30
    
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
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, name in backends:
            try:
                print(f"Trying to open camera {index} with {name}...")
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
                        print(f"✗ {name}: Camera opened but couldn't read frame")
                        cap.release()
                else:
                    print(f"✗ {name}: Failed to open camera")
            except Exception as e:
                print(f"✗ {name}: Exception: {e}")
                continue
        
        return None
    
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
        except Exception:
            pass  # Beberapa kamera tidak mendukung ini
        
        self.capture_started.emit()
        
        frame_delay = 1.0 / self._target_fps
        consecutive_failures = 0
        max_failures = 30
        
        while self._running:
            loop_start = time.time()
            
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
            
            # Kontrol kecepatan frame
            processing_time = time.time() - loop_start
            if processing_time < frame_delay:
                time.sleep(frame_delay - processing_time)
        
        # Pembersihan saat keluar
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def is_running(self) -> bool:
        """Periksa apakah penangkapan sedang berjalan"""
        return self._running
