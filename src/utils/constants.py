"""
Application Constants
Central configuration for window settings, detection parameters, and YOLO models.
"""

import os

# =============================================================================
# Window Settings
# =============================================================================
WINDOW_TITLE = "Poltekad - Elkasista"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# =============================================================================
# Detection Settings
# =============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection (0.0 - 1.0)
PERSON_CLASS_ID = 0         # COCO class ID for person

# =============================================================================
# Available YOLO Models
# Only nano (Fast) and small (Balanced) models for CPU performance
# =============================================================================
YOLO_MODELS = {
    # YOLOv8 (2023 - stable, widely tested)
    "YOLOv8n - Fast": {
        "file": "yolov8n.pt",
        "description": "Nano model - Fastest, good for real-time on CPU",
        "size": "6.3 MB"
    },
    "YOLOv8s - Balanced": {
        "file": "yolov8s.pt", 
        "description": "Small model - Balanced speed and accuracy",
        "size": "22.5 MB"
    },
    # YOLOv11 (2024 - improved architecture)
    "YOLOv11n - Fast": {
        "file": "yolo11n.pt",
        "description": "Nano v11 - Faster with better accuracy than v8",
        "size": "5.4 MB"
    },
    "YOLOv11s - Balanced": {
        "file": "yolo11s.pt",
        "description": "Small v11 - Great balance for real-time detection",
        "size": "18.4 MB"
    },
    # YOLO12 (2025 - state of the art)
    "YOLO12n - Fast": {
        "file": "yolo12n.pt",
        "description": "Nano v12 - Latest, fastest with attention mechanisms",
        "size": "5.6 MB"
    },
    "YOLO12s - Balanced": {
        "file": "yolo12s.pt",
        "description": "Small v12 - Best balance with area attention",
        "size": "19.2 MB"
    },
}

DEFAULT_MODEL = "YOLOv8n - Fast"

# =============================================================================
# Detection Annotation Colors (BGR format for OpenCV)
# =============================================================================
DETECTION_BOX_COLOR = (0, 255, 0)  # Green bounding box

# =============================================================================
# Camera Settings
# =============================================================================
MAX_CAMERA_INDEX = 10  # Maximum camera index to scan when searching for cameras

# =============================================================================
# Recording & Capture
# =============================================================================
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), "Documents", "HumanDetectionApp")
RECORDING_FPS = 20.0        # Output video framerate
RECORDING_CODEC = "mp4v"    # FourCC codec for .mp4 output

# =============================================================================
# Performance Settings
# =============================================================================
DEFAULT_CAPTURE_FPS = 30    # Default camera capture FPS
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
