"""
Layanan Kamera - Menangani deteksi dan enumerasi kamera.
Memindai kamera yang tersedia menggunakan beberapa backend untuk kompatibilitas Windows.
Pemindaian dilakukan di thread latar belakang agar UI tidak freeze.
"""

import cv2
import time
from typing import List, Dict
from PyQt5.QtCore import QThread, pyqtSignal
from utils.constants import MAX_CAMERA_INDEX


class CameraScanThread(QThread):
    """Thread latar belakang untuk memindai kamera tanpa memblokir UI."""
    
    # Sinyal: daftar kamera ditemukan, atau pesan error
    cameras_found = pyqtSignal(list)
    scan_error = pyqtSignal(str)
    
    def run(self):
        """Pindai kamera di thread terpisah."""
        try:
            cameras = []
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
            
            self.cameras_found.emit(cameras)
            
        except Exception as e:
            self.scan_error.emit(f"Gagal memindai kamera: {e}")


class CameraService:
    """
    Layanan untuk mendeteksi dan melihat kamera yang tersedia.
    Menggunakan backend DirectShow dan MSMF untuk kompatibilitas Windows yang lebih baik.
    Mendukung pemindaian sinkron dan asinkron (via thread).
    """
    
    def __init__(self):
        self._cameras: List[Dict] = []
        self._scan_thread: CameraScanThread = None
    
    def scan_async(self) -> CameraScanThread:
        """
        Pindai kamera di thread latar belakang (tidak memblokir UI).
        
        Returns:
            CameraScanThread — hubungkan sinyal cameras_found / scan_error sebelum mulai.
        """
        self._scan_thread = CameraScanThread()
        return self._scan_thread
    
    def get_available_cameras(self) -> List[Dict]:
        """
        Pindai dan kembalikan daftar kamera yang tersedia (sinkron, memblokir).
        Gunakan scan_async() untuk pemindaian non-blocking.
        
        Returns:
            Daftar dict dengan info kamera:
            {'index': int, 'name': str, 'resolution': (width, height)}
        """
        cameras = []
        
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
                    
                    time.sleep(0.1)
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
                        break
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"Error scanning camera {index} with {backend_name}: {e}")
                    continue
        
        self._cameras = cameras
        return cameras
