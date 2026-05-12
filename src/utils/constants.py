"""
Application Constants
Konfigurasi pusat untuk pengaturan jendela, parameter deteksi, dan model YOLO.
"""

import os

# =============================================================================
# Window Settings
# =============================================================================
WINDOW_TITLE = "Poltekad - Fire Detection"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# =============================================================================
# Pengaturan Deteksi
# =============================================================================
CONFIDENCE_THRESHOLD = 0.25  # Kepercayaan minimum untuk deteksi (0.0 - 1.0)

# =============================================================================
# Model YOLO - Fire Detection
# YOLO-MP model and compatibility modules live in YOLO-MP-master.
# =============================================================================
YOLO_MP_DIR = "YOLO-MP-master"
YOLO_MP_MODEL_FILE = "YOLO-MP.pt"

FIRE_MODEL = {
    "name": "YOLO-MP Fire",
    "file": os.path.join(YOLO_MP_DIR, YOLO_MP_MODEL_FILE),
    "description": "YOLO-MP lightweight forest fire detection model",
    "size": "5 MB"
}

DEFAULT_MODEL = FIRE_MODEL["name"]

# =============================================================================
# Warna Anotasi Deteksi (format BGR untuk OpenCV)
# Per-class colors for fire detection
# =============================================================================
DETECTION_CLASS_COLORS = {
    "fire": (0, 0, 255),       # Red in BGR
}
DETECTION_DEFAULT_COLOR = (0, 255, 0)  # Fallback green

# =============================================================================
# Pengaturan Kamera
# =============================================================================
MAX_CAMERA_INDEX = 10  # Indeks kamera maksimum untuk dipindai

# =============================================================================
# Perekaman & Tangkapan
# =============================================================================
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), "Documents", "FireDetectionApp")
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
