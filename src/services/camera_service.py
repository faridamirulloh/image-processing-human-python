"""
Camera Service - Handles camera detection and enumeration.
Scans for available cameras using multiple backends for Windows compatibility.
"""

import cv2
import time
from typing import List, Dict
from utils.constants import MAX_CAMERA_INDEX


class CameraService:
    """
    Service for detecting and listing available cameras.
    Uses DirectShow and MSMF backends for better Windows compatibility.
    """
    
    def __init__(self):
        self._cameras: List[Dict] = []
    
    def get_available_cameras(self) -> List[Dict]:
        """
        Scan and return list of available cameras.
        
        Returns:
            List of dicts with camera info:
            {'index': int, 'name': str, 'resolution': (width, height)}
        """
        cameras = []
        
        # DirectShow and MSMF backends work best on Windows
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
                    
                    # Brief warmup before reading
                    time.sleep(0.1)
                    
                    # Verify camera can actually capture frames
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
                        break  # Found working camera, try next index
                    else:
                        cap.release()
                        
                except Exception as e:
                    print(f"Error scanning camera {index} with {backend_name}: {e}")
                    continue
        
        self._cameras = cameras
        return cameras
