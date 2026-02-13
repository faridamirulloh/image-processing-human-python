"""
Video Service - Handles video capture from camera in a separate thread.
Emits frames via PyQt signals for processing and display.
"""

import cv2
import time
import numpy as np
from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class VideoService(QThread):
    """
    Video capture service running in a separate thread.
    Emits frames for display and processing.
    """
    
    # Signals for async communication with main thread
    frame_ready = pyqtSignal(np.ndarray)  # Emits raw camera frame
    error_occurred = pyqtSignal(str)       # Emits error message
    capture_started = pyqtSignal()         # Emits when capture starts
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera_index: int = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._running: bool = False
        self._mutex = QMutex()  # Thread safety for camera access
        self._target_fps: int = 30
    
    def _open_camera(self, index: int) -> Optional[cv2.VideoCapture]:
        """
        Try to open camera with different backends.
        DirectShow is preferred on Windows for better compatibility.
        
        Args:
            index: Camera index
            
        Returns:
            VideoCapture object or None if failed
        """
        # Try backends in order of preference for Windows
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
                    # Camera warm-up time (important for some cameras)
                    time.sleep(0.5)
                    
                    # Discard first few frames (often corrupted or black)
                    for _ in range(5):
                        cap.read()
                    
                    # Verify we can actually read frames
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
        Start video capture.
        
        Args:
            camera_index: Camera index to capture from
            
        Returns:
            True if capture thread started
        """
        if camera_index is not None:
            self._camera_index = camera_index
            
        self._running = True
        self.start()  # Start QThread
        return True
    
    def stop_capture(self):
        """Stop video capture gracefully"""
        self._running = False
        self.wait(2000)  # Wait up to 2 seconds for thread to finish
        
        # If thread finished, it already released the camera.
        # If it timed out, we force release here.
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def run(self):
        """Main capture loop - runs in separate thread"""
        # Open camera with best available backend
        self._capture = self._open_camera(self._camera_index)
        
        if self._capture is None or not self._capture.isOpened():
            self.error_occurred.emit(
                f"Could not open camera {self._camera_index}.\n"
                "Please check:\n"
                "• No other app is using the camera\n"
                "• Camera privacy is enabled in Windows Settings\n"
                "• Camera drivers are up to date"
            )
            return
        
        # Reduce buffer size for lower latency
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass  # Some cameras don't support this
        
        self.capture_started.emit()
        
        frame_delay = 1.0 / self._target_fps
        consecutive_failures = 0
        max_failures = 30
        
        while self._running:
            loop_start = time.time()
            
            # Thread-safe camera access
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
                time.sleep(0.05)  # Brief pause before retry
                continue
            
            # Frame rate control
            processing_time = time.time() - loop_start
            if processing_time < frame_delay:
                time.sleep(frame_delay - processing_time)
        
        # Cleanup on exit
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def is_running(self) -> bool:
        """Check if capture is currently running"""
        return self._running
