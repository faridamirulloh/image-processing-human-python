"""
Layanan Kamera - Menangani deteksi dan enumerasi kamera.
Memindai kamera yang tersedia menggunakan beberapa backend untuk kompatibilitas Windows.
"""

import cv2
import time
from typing import List, Dict
from utils.constants import MAX_CAMERA_INDEX


class CameraService:
    """
    Layanan untuk mendeteksi dan melihat kamera yang tersedia.
    Menggunakan backend DirectShow dan MSMF untuk kompatibilitas Windows yang lebih baik.
    """
    
    def __init__(self):
        self._cameras: List[Dict] = []
    
    def get_available_cameras(self) -> List[Dict]:
        """
        Pindai dan kembalikan daftar kamera yang tersedia.
        
        Returns:
            Daftar dict dengan info kamera:
            {'index': int, 'name': str, 'resolution': (width, height)}
        """
        cameras = []
        
        # Backend DirectShow dan MSMF bekerja paling baik di Windows
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
        
        for index in range(MAX_CAMERA_INDEX):
            for backend, backend_name in backends:
                try:
                    cap = cv2.VideoCapture(index, backend)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    # Pemanasan singkat sebelum membaca
                    time.sleep(0.1)
                    
                    # Verifikasi kamera benar-benar bisa menangkap frame
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        cameras.append({
                            'index': index,
                            'name': f"Camera {index}",
                            'resolution': (width, height)
                        })
                        
                        cap.release()
                        break  # Ditemukan kamera yang berfungsi, coba indeks berikutnya
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"Error scanning camera {index} with {backend_name}: {e}")
                    continue
        
        self._cameras = cameras
        return cameras
