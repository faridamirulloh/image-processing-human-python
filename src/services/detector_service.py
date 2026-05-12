"""
Layanan Detektor - Deteksi Api YOLO
Menyediakan deteksi api berbasis AI menggunakan model YOLO-MP.
Inferensi hanya CPU — dioptimalkan untuk menggunakan semua core CPU.
"""

import os
import sys
import inspect
import importlib
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional

from utils.constants import (
    FIRE_MODEL,
    DEFAULT_MODEL,
    CONFIDENCE_THRESHOLD,
    DETECTION_CLASS_COLORS,
    DETECTION_DEFAULT_COLOR,
    INFERENCE_SCALE,
    SKIP_FRAMES_DEFAULT
)

import time

CONFIDENCE_UPDATE_INTERVAL = 0.25
BOX_SMOOTHING_FACTOR = 0.3


class DetectorService:
    """Layanan deteksi api menggunakan model YOLO-MP."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = False):
        self._model = None
        self._model_name: str = model_name
        self._model_names: Dict[int, str] = {}
        self._device: str = "cpu"
        self._confidence: float = CONFIDENCE_THRESHOLD
        self._last_detections: List[Dict] = []
        self._trackers = {}
        self._next_track_id = 0
        self._inference_scale: float = INFERENCE_SCALE
        self._skip_frames: int = SKIP_FRAMES_DEFAULT
        self._frame_counter: int = 0
        self._last_annotated_detections: List[Dict] = []
        self._torch_available = False
        self._init_error: Optional[str] = None
        
        try:
            import torch
            self._torch_available = True
        except Exception as e:
            self._init_error = str(e)
            print(f"Warning: PyTorch not available: {e}")
        
        if self._torch_available:
            try:
                self._optimize_cpu()
            except Exception as e:
                print(f"Warning: CPU optimization failed (non-fatal): {e}")
        
        cv2.setUseOptimized(True)
        
        if self._torch_available:
            self.load_model(model_name)
    
    def _optimize_cpu(self):
        """Optimalkan penggunaan CPU secara dinamis berdasarkan perangkat."""
        import torch
        total_cores = os.cpu_count() or 2
        if total_cores >= 8:
            infer_threads = total_cores
        elif total_cores >= 4:
            infer_threads = total_cores - 1
        else:
            infer_threads = max(1, total_cores - 1)
        torch.set_num_threads(infer_threads)
        try:
            interop = max(1, total_cores // 4)
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            pass
        print(f"CPU optimization: {infer_threads}/{total_cores} cores, torch threads={torch.get_num_threads()}")
    
    @property
    def torch_available(self) -> bool:
        return self._torch_available
    
    @property
    def init_error(self) -> Optional[str]:
        return self._init_error
    
    @property
    def current_model(self) -> str:
        return self._model_name
    
    def _get_model_path(self, model_file: str) -> str:
        """Cari path file model: bundel PyInstaller, CWD, dir proyek, subdir referensi."""
        model_basename = os.path.basename(model_file)
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_path = os.path.join(sys._MEIPASS, model_file)
            if os.path.exists(bundle_path):
                return bundle_path
            bundle_root_path = os.path.join(sys._MEIPASS, model_basename)
            if os.path.exists(bundle_root_path):
                return bundle_root_path
        if os.path.exists(model_file):
            return model_file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        project_path = os.path.join(project_dir, model_file)
        if os.path.exists(project_path):
            return project_path
        root_model_path = os.path.join(project_dir, model_basename)
        if os.path.exists(root_model_path):
            return root_model_path
        ref_path = os.path.join(project_dir, "YOLO-MP-master", model_basename)
        if os.path.exists(ref_path):
            return ref_path
        print(f"Model not found locally, will attempt download: {model_file}")
        return model_file

    def _get_yolo_mp_nn_path(self) -> Optional[str]:
        """Return the local YOLO-MP ultralytics/nn folder when present."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
        nn_path = os.path.join(project_dir, "YOLO-MP-master", "ultralytics", "nn")
        return os.path.abspath(nn_path) if os.path.isdir(nn_path) else None

    def _install_yolo_mp_compat_modules(self):
        """Make local YOLO-MP extra_modules visible to the installed Ultralytics package."""
        import ultralytics.nn as yolo_nn

        nn_path = self._get_yolo_mp_nn_path()
        if nn_path and nn_path not in yolo_nn.__path__:
            yolo_nn.__path__.append(nn_path)

    def _allow_yolo_mp_safe_globals(self):
        """
        Allow PyTorch 2.6+ to load older Ultralytics checkpoints while
        keeping weights_only=True. This is narrower than disabling checkpoint
        safety and only permits classes from torch.nn and ultralytics.nn.
        """
        import torch
        import torch.nn as torch_nn
        from dill._dill import _load_type
        from torch.serialization import add_safe_globals
        import ultralytics.nn.modules as yolo_modules
        import ultralytics.nn.tasks as yolo_tasks
        extra_blocks = importlib.import_module("ultralytics.nn.extra_modules.block")

        safe_classes = [_load_type]
        for module in (torch_nn, yolo_modules, yolo_tasks, extra_blocks):
            safe_classes.extend(
                obj for _, obj in vars(module).items()
                if inspect.isclass(obj) and obj.__module__.startswith(("torch.nn", "ultralytics.nn"))
            )
        add_safe_globals(safe_classes)
    
    def load_model(self, model_name: str = DEFAULT_MODEL, use_gpu: bool = False) -> bool:
        """Muat model YOLO untuk deteksi api."""
        if not self._torch_available:
            return False
        try:
            from ultralytics import YOLO
            self._install_yolo_mp_compat_modules()
            self._allow_yolo_mp_safe_globals()
            model_file = FIRE_MODEL["file"]
            model_path = self._get_model_path(model_file)
            self._model = YOLO(model_path)
            self._model.to(self._device)
            self._model_name = FIRE_MODEL["name"]
            if hasattr(self._model, 'names'):
                self._model_names = self._model.names
                print(f"Model classes: {self._model_names}")
            print(f"Loaded {self._model_name} from {model_path} on {self._device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self._init_error = str(e)
            return False
    
    def _get_class_name(self, cls_id: int) -> str:
        if self._model_names and cls_id in self._model_names:
            return self._model_names[cls_id]
        return f"class_{cls_id}"
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        name_lower = class_name.lower()
        for key, color in DETECTION_CLASS_COLORS.items():
            if key in name_lower:
                return color
        return DETECTION_DEFAULT_COLOR
    
    def set_inference_scale(self, scale: float):
        self._inference_scale = max(0.25, min(scale, 1.0))
    
    def get_inference_scale(self) -> float:
        return self._inference_scale
    
    def set_skip_frames(self, n: int):
        self._skip_frames = max(1, min(n, 10))
        self._frame_counter = 0
    
    def get_skip_frames(self) -> int:
        return self._skip_frames
    
    def _redraw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Redraw cached detection bounding boxes on a new frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det.get('class_name', 'unknown')
            color = self._get_class_color(class_name)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name.capitalize()} {conf * 100:.0f}%"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated

    def detect_fire(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Detect fire in a frame and annotate with bounding boxes.
        
        Returns:
            Tuple of (annotated_frame, fire_count, detections)
        """
        if self._model is None:
            return frame, 0, []
        
        self._frame_counter += 1
        if self._skip_frames > 1 and self._frame_counter % self._skip_frames != 1:
            if self._last_annotated_detections:
                annotated = self._redraw_detections(frame, self._last_annotated_detections)
                fire_count = self._count_fire(self._last_annotated_detections)
                return annotated, fire_count, self._last_annotated_detections
        
        try:
            h, w = frame.shape[:2]
            if self._inference_scale < 1.0:
                new_w = int(w * self._inference_scale)
                new_h = int(h * self._inference_scale)
                inference_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                scale_x, scale_y = w / new_w, h / new_h
            else:
                inference_frame = frame
                scale_x = scale_y = 1.0
            
            results = self._model(inference_frame, verbose=False, conf=self._confidence)
            detections = []
            annotated_frame = frame.copy()
            current_time = time.time()
            current_trackers = {}
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self._get_class_name(cls_id)
                    if not self._is_fire_class(class_name):
                        continue
                    color = self._get_class_color(class_name)
                    bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                    x1, y1 = int(bx1 * scale_x), int(by1 * scale_y)
                    x2, y2 = int(bx2 * scale_x), int(by2 * scale_y)
                    raw_conf = float(box.conf[0])
                    current_bbox = (x1, y1, x2, y2)
                    
                    best_match_id = None
                    max_iou = 0.5
                    for tid, data in self._trackers.items():
                        iou = self._calculate_iou(current_bbox, data['bbox'])
                        if iou > max_iou:
                            max_iou = iou
                            best_match_id = tid
                    
                    if best_match_id is not None:
                        tracker = self._trackers[best_match_id]
                        if current_time - tracker['last_update'] > CONFIDENCE_UPDATE_INTERVAL:
                            display_conf, last_update = raw_conf, current_time
                        else:
                            display_conf, last_update = tracker['conf'], tracker['last_update']
                        old = tracker['bbox']
                        s = BOX_SMOOTHING_FACTOR
                        final_bbox = (
                            old[0]*(1-s)+x1*s, old[1]*(1-s)+y1*s,
                            old[2]*(1-s)+x2*s, old[3]*(1-s)+y2*s
                        )
                        current_trackers[best_match_id] = {
                            'conf': display_conf, 'last_update': last_update,
                            'bbox': final_bbox, 'class_name': class_name
                        }
                    else:
                        self._next_track_id += 1
                        display_conf = raw_conf
                        final_bbox = tuple(map(float, current_bbox))
                        current_trackers[self._next_track_id] = {
                            'conf': display_conf, 'last_update': current_time,
                            'bbox': final_bbox, 'class_name': class_name
                        }
                    
                    dx1, dy1, dx2, dy2 = map(int, final_bbox)
                    detections.append({
                        'bbox': (dx1, dy1, dx2, dy2), 'confidence': display_conf,
                        'class_id': cls_id, 'class_name': class_name
                    })
                    cv2.rectangle(annotated_frame, (dx1, dy1), (dx2, dy2), color, 2)
                    label = f"{class_name.capitalize()} {display_conf * 100:.0f}%"
                    ls, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (dx1, dy1 - ls[1] - 10), (dx1 + ls[0], dy1), color, -1)
                    cv2.putText(annotated_frame, label, (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self._trackers = current_trackers
            self._last_detections = detections
            self._last_annotated_detections = detections
            return annotated_frame, self._count_fire(detections), detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, 0, []
    
    def _is_fire_class(self, class_name: str) -> bool:
        """Return True when a detection belongs to the fire class."""
        return "fire" in class_name.lower()

    def _count_fire(self, detections: List[Dict]) -> int:
        """Count fire detections."""
        fire = 0
        for det in detections:
            name = det.get('class_name', '').lower()
            if 'fire' in name:
                fire += 1
        return fire
    
    def get_last_detections(self) -> List[Dict]:
        return self._last_detections
    
    def set_confidence(self, confidence: float):
        self._confidence = max(0.1, min(confidence, 1.0))

    def _calculate_iou(self, box1, box2) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        xi_min, yi_min = max(x1_min, x2_min), max(y1_min, y2_min)
        xi_max, yi_max = min(x1_max, x2_max), min(y1_max, y2_max)
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        union_area = (x1_max-x1_min)*(y1_max-y1_min) + (x2_max-x2_min)*(y2_max-y2_min) - inter_area
        return inter_area / union_area if union_area else 0.0
