"""
Application Constants
Konfigurasi pusat untuk pengaturan jendela, parameter deteksi, dan model YOLO.
"""

import os

# =============================================================================
# Window Settings
# =============================================================================
WINDOW_TITLE = "Poltekad - Elkasista"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# =============================================================================
# Pengaturan Deteksi
# =============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Kepercayaan minimum untuk deteksi (0.0 - 1.0)
PERSON_CLASS_ID = 0         # ID class COCO untuk orang

# =============================================================================
# Model YOLO yang Tersedia
# Hanya model nano (Cepat) dan kecil (Seimbang) untuk kinerja CPU
# =============================================================================
YOLO_MODELS = {
    # YOLOv8 (2023 - stabil, diuji secara luas)
    "YOLOv8n - Fast": {
        "file": "yolov8n.pt",
        "description": "Model Nano - Tercepat, bagus untuk real-time di CPU",
        "size": "6.3 MB"
    },
    "YOLOv8s - Balanced": {
        "file": "yolov8s.pt", 
        "description": "Model Kecil - Kecepatan dan akurasi seimbang",
        "size": "22.5 MB"
    },
    # YOLOv11 (2024 - arsitektur yang ditingkatkan)
    "YOLOv11n - Fast": {
        "file": "yolo11n.pt",
        "description": "Nano v11 - Lebih cepat dengan akurasi lebih baik dari v8",
        "size": "5.4 MB"
    },
    "YOLOv11s - Balanced": {
        "file": "yolo11s.pt",
        "description": "Kecil v11 - Keseimbangan hebat untuk deteksi real-time",
        "size": "18.4 MB"
    },
    # YOLO12 (2025 - mutakhir)
    "YOLO12n - Fast": {
        "file": "yolo12n.pt",
        "description": "Nano v12 - Terbaru, tercepat dengan mekanisme atensi",
        "size": "5.6 MB"
    },
    "YOLO12s - Balanced": {
        "file": "yolo12s.pt",
        "description": "Kecil v12 - Keseimbangan terbaik dengan atensi area",
        "size": "19.2 MB"
    },
}

DEFAULT_MODEL = "YOLOv8n - Fast"

# =============================================================================
# Warna Anotasi Deteksi (format BGR untuk OpenCV)
# =============================================================================
DETECTION_BOX_COLOR = (0, 255, 0)  # Kotak pembatas hijau

# =============================================================================
# Pengaturan Kamera
# =============================================================================
MAX_CAMERA_INDEX = 10  # Indeks kamera maksimum untuk dipindai saat mencari kamera

# =============================================================================
# Perekaman & Tangkapan
# =============================================================================
DEFAULT_OUTPUT_FOLDER = os.path.join(os.path.expanduser("~"), "Documents", "HumanDetectionApp")
RECORDING_FPS = 20.0        # Framerate video output
RECORDING_CODEC = "mp4v"    # Codec FourCC untuk output .mp4
