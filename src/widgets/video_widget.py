"""
Video Widget - Displays the live camera / detection feed.
Handles frame rendering, aspect ratio scaling, and status messages.
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap


class VideoWidget(QLabel):
    """
    Displays live video frames inside a styled QLabel.
    Automatically scales frames to fit while keeping the aspect ratio.
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
        
        self._current_frame = None   # Latest frame (kept for screenshots & resize)
        self._is_active = False      # True when displaying video frames
        
        self.show_no_camera()
    
    # =========================================================================
    # Status Messages (shown when no video is playing)
    # =========================================================================
    
    def show_no_camera(self):
        """Show placeholder when no camera is connected"""
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
        """Show loading state while camera is connecting"""
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
        """Show error message when camera fails"""
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
    # Frame Display
    # =========================================================================
    
    def update_frame(self, frame: np.ndarray):
        """
        Display a new video frame.
        
        Args:
            frame: BGR frame from OpenCV
        """
        if frame is None:
            return
            
        self._current_frame = frame
        
        # Only update style when switching from inactive to active state
        if not self._is_active:
            self._is_active = True
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a1a2e;
                    border: 2px solid #00d9ff;
                    border-radius: 8px;
                }
            """)
        
        # Convert BGR (OpenCV) → RGB (Qt) for correct colors
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create Qt image and scale to fit widget (keeping aspect ratio)
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.setPixmap(scaled_pixmap)
    
    def clear_display(self):
        """Clear video and show the no-camera placeholder"""
        self._current_frame = None
        self.show_no_camera()
    
    def is_active(self) -> bool:
        """Check if video is currently being displayed"""
        return self._is_active
    
    # =========================================================================
    # Qt Overrides
    # =========================================================================
    
    def resizeEvent(self, event):
        """Re-scale the current frame when the widget is resized"""
        super().resizeEvent(event)
        if self._current_frame is not None:
            # We call update_frame to re-scale, but we don't need to re-set the style
            # logic in update_frame handles it efficiently
            self.update_frame(self._current_frame)
    
    def sizeHint(self) -> QSize:
        """Default suggested size"""
        return QSize(640, 480)
