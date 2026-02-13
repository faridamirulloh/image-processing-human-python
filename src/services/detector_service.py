"""
Detector Service - YOLO Human Detection
Provides AI-based person detection using YOLO models (v8, v11, v12).
CPU-only inference for maximum compatibility.
"""

import os
import sys
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

from utils.constants import (
    YOLO_MODELS, 
    DEFAULT_MODEL, 
    CONFIDENCE_THRESHOLD, 
    PERSON_CLASS_ID,
    DETECTION_BOX_COLOR
)

import time

# Confidence update interval in seconds (stabilizes UI flicker)
# Confidence update interval in seconds (stabilizes UI flicker)
CONFIDENCE_UPDATE_INTERVAL = 0.25

# Bounding box smoothing factor (0.0 - 1.0)
# Lower = smoother/slower, Higher = faster/jittery
BOX_SMOOTHING_FACTOR = 0.3


class DetectorService:
    """
    Human detection service using YOLO models.
    Handles model loading, inference, and result annotation.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = False):
        """
        Initialize the detector service.
        
        Args:
            model_name: Name of the model from YOLO_MODELS
            use_gpu: Ignored (CPU-only mode)
        """
        self._model = None
        self._model_name: str = model_name
        self._device: str = "cpu"
        self._confidence: float = CONFIDENCE_THRESHOLD
        self._last_detections: List[Dict] = []
        
        # Tracking for confidence stabilization
        # format: {track_id: {'conf': float, 'last_update': float, 'bbox': tuple}}
        self._trackers = {}
        self._next_track_id = 0
        
        self._torch_available = False
        self._init_error: Optional[str] = None
        
        # Check if PyTorch is available
        try:
            import torch
            self._torch_available = True
        except Exception as e:
            self._init_error = str(e)
            print(f"Warning: PyTorch not available: {e}")
        
        # Load model if PyTorch is available
        if self._torch_available:
            self.load_model(model_name)
    
    @property
    def torch_available(self) -> bool:
        """Check if PyTorch is available"""
        return self._torch_available
    
    @property
    def init_error(self) -> Optional[str]:
        """Get initialization error if any"""
        return self._init_error
    
    @property
    def current_model(self) -> str:
        """Get current model name"""
        return self._model_name
    
    def _get_model_path(self, model_file: str) -> str:
        """
        Resolve model file path.
        Checks in order: PyInstaller bundle, current dir, project dir.
        Falls back to original filename (ultralytics will download).
        
        Args:
            model_file: Model filename (e.g., 'yolov8n.pt')
            
        Returns:
            Full path to the model file
        """
        # Check PyInstaller bundle first (for packaged .exe)
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_path = os.path.join(sys._MEIPASS, model_file)
            if os.path.exists(bundle_path):
                print(f"Using bundled model: {bundle_path}")
                return bundle_path
        
        # Check current working directory
        if os.path.exists(model_file):
            return model_file
        
        # Check project directory (two levels up from this file)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        project_path = os.path.join(project_dir, model_file)
        if os.path.exists(project_path):
            return project_path
        
        # Model not found locally - ultralytics will download it
        print(f"Model not found locally, will attempt download: {model_file}")
        return model_file
    
    def load_model(self, model_name: str, use_gpu: bool = False) -> bool:
        """
        Load a YOLO model.
        
        Args:
            model_name: Name of the model from YOLO_MODELS
            use_gpu: Ignored (CPU-only mode)
            
        Returns:
            True if model loaded successfully
        """
        if not self._torch_available:
            return False
            
        try:
            from ultralytics import YOLO
            
            # Validate model name
            if model_name not in YOLO_MODELS:
                print(f"Unknown model: {model_name}, using default")
                model_name = DEFAULT_MODEL
            
            model_file = YOLO_MODELS[model_name]["file"]
            model_path = self._get_model_path(model_file)
            
            # Load model on CPU
            self._model = YOLO(model_path)
            self._model.to(self._device)
            self._model_name = model_name
            
            print(f"Loaded {model_name} on {self._device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self._init_error = str(e)
            return False
    
    def detect_humans(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Detect humans in a frame and annotate with bounding boxes.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Tuple of (annotated_frame, person_count, detections)
            - annotated_frame: Frame with drawn bounding boxes
            - person_count: Number of people detected
            - detections: List of detection dicts with bbox, confidence
        """
        if self._model is None:
            return frame, 0, []
        
        try:
            # Run YOLO inference
            results = self._model(frame, verbose=False, conf=self._confidence)
            
            detections = []
            annotated_frame = frame.copy()
            current_time = time.time()
            
            # Temporary list for current frame matches
            current_trackers = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    
                    # Filter for person class only
                    if cls_id != PERSON_CLASS_ID:
                        continue
                        
                    # Extract bbox and raw confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    raw_conf = float(box.conf[0])
                    current_bbox = (x1, y1, x2, y2)
                    
                    # Simple tracking: find best matching existing tracker via IoU
                    best_match_id = None
                    max_iou = 0.5  # IoU threshold for matching
                    
                    for tid, data in self._trackers.items():
                        iou = self._calculate_iou(current_bbox, data['bbox'])
                        if iou > max_iou:
                            max_iou = iou
                            best_match_id = tid
                    
                    # Determine stable confidence and smooth bbox
                    if best_match_id is not None:
                        # Found existing object
                        tracker = self._trackers[best_match_id]
                        
                        # Update confidence only if interval passed
                        if current_time - tracker['last_update'] > CONFIDENCE_UPDATE_INTERVAL:
                            display_conf = raw_conf
                            last_update = current_time
                        else:
                            display_conf = tracker['conf']
                            last_update = tracker['last_update']
                        
                        # Smooth bbox using Exponential Moving Average (EMA)
                        old_x1, old_y1, old_x2, old_y2 = tracker['bbox']
                        curr_x1, curr_y1, curr_x2, curr_y2 = current_bbox
                        
                        smooth_x1 = old_x1 * (1 - BOX_SMOOTHING_FACTOR) + curr_x1 * BOX_SMOOTHING_FACTOR
                        smooth_y1 = old_y1 * (1 - BOX_SMOOTHING_FACTOR) + curr_y1 * BOX_SMOOTHING_FACTOR
                        smooth_x2 = old_x2 * (1 - BOX_SMOOTHING_FACTOR) + curr_x2 * BOX_SMOOTHING_FACTOR
                        smooth_y2 = old_y2 * (1 - BOX_SMOOTHING_FACTOR) + curr_y2 * BOX_SMOOTHING_FACTOR
                        
                        final_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                            
                        # Update tracker
                        current_trackers[best_match_id] = {
                            'conf': display_conf,
                            'last_update': last_update,
                            'bbox': final_bbox
                        }
                    else:
                        # New object detected - use raw values
                        self._next_track_id += 1
                        display_conf = raw_conf
                        final_bbox = tuple(map(float, current_bbox)) # Store as float for smoothing
                        
                        current_trackers[self._next_track_id] = {
                            'conf': display_conf,
                            'last_update': current_time,
                            'bbox': final_bbox
                        }
                    
                    # Convert smoothed bbox to int for drawing
                    draw_x1, draw_y1, draw_x2, draw_y2 = map(int, final_bbox)
                    
                    detections.append({
                        'bbox': (draw_x1, draw_y1, draw_x2, draw_y2),
                        'confidence': display_conf,
                        'class_id': cls_id
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(
                        annotated_frame, 
                        (draw_x1, draw_y1), (draw_x2, draw_y2), 
                        DETECTION_BOX_COLOR, 
                        2
                    )
                    
                    # Draw label with stabilized confidence
                    label = f"Person {display_conf * 100:.0f}%"
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Label background
                    cv2.rectangle(
                        annotated_frame,
                        (draw_x1, draw_y1 - label_size[1] - 10),
                        (draw_x1 + label_size[0], draw_y1),
                        DETECTION_BOX_COLOR,
                        -1
                    )
                    
                    # Label text
                    cv2.putText(
                        annotated_frame, label,
                        (draw_x1, draw_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2
                    )
            
            # Update trackers list (remove lost objects)
            self._trackers = current_trackers
            self._last_detections = detections
            
            return annotated_frame, len(detections), detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, 0, []
    
    def get_last_detections(self) -> List[Dict]:
        """Get the last detection results"""
        return self._last_detections
    
    def set_confidence(self, confidence: float):
        """Set detection confidence threshold (0.1 to 1.0)"""
        self._confidence = max(0.1, min(confidence, 1.0))

    def _calculate_iou(self, box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection coords
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        # Calculate intersection area
        inter_width = max(0, xi_max - xi_min)
        inter_height = max(0, yi_max - yi_min)
        inter_area = inter_width * inter_height
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
            
        return inter_area / union_area
