"""
Application Constants
Konfigurasi pusat untuk pengaturan jendela, parameter deteksi, dan model YOLO.
"""

import os

# =============================================================================
# Window Settings
# =============================================================================
WINDOW_TITLE = "Poltekad - Fire & Smoke Detection"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# =============================================================================
# Pengaturan Deteksi
# =============================================================================
CONFIDENCE_THRESHOLD = 0.25  # Kepercayaan minimum untuk deteksi (0.0 - 1.0)

# =============================================================================
# Model YOLO - Fire & Smoke Detection
# Custom-trained YOLOv10 model for fire and smoke detection
# =============================================================================
FIRE_SMOKE_MODEL = {
    "name": "YOLOv10 Fire & Smoke",
    "file": "best.pt",
    "description": "YOLOv10 model fine-tuned for fire and smoke detection",
    "size": "64 MB"
}

DEFAULT_MODEL = FIRE_SMOKE_MODEL["name"]

# =============================================================================
# Warna Anotasi Deteksi (format BGR untuk OpenCV)
# Per-class colors for fire and smoke
# =============================================================================
DETECTION_CLASS_COLORS = {
    "fire": (0, 0, 255),       # Red in BGR
    "smoke": (255, 150, 50),   # Blue in BGR
}
DETECTION_DEFAULT_COLOR = (0, 255, 0)  # Fallback green

# =============================================================================
# Pengaturan Kamera
# =============================================================================
MAX_CAMERA_INDEX = 10  # Indeks kamera maksimum untuk dipindai

# =============================================================================
# Perekaman & Tangkapan
# =============================================================================
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), "Documents", "FireSmokeDetectionApp")
RECORDING_FPS = 20.0        # Output video framerate
RECORDING_CODEC = "mp4v"    # FourCC codec for .mp4 output

# =============================================================================
# Performance Settings
# =============================================================================
DEFAULT_CAPTURE_FPS = 1     # Default detection FPS (does not affect video stream speed)
MIN_FPS = 1                 # Minimum allowed FPS
MAX_FPS = 60                # Maximum allowed FPS

# Named FPS presets for the settings UI
FPS_PRESETS = {
    "Very Low (5 FPS)": 5,
    "Low (10 FPS)": 10,
    "Medium (20 FPS)": 20,
    "High (30 FPS)": 30,
}

# Inference downscale factor (1.0 = full res, 0.5 = half, 0.25 = quarter)
INFERENCE_SCALE = 1.0

# Skip-frame detection: run YOLO every Nth frame (1 = every frame)
SKIP_FRAMES_DEFAULT = 1
