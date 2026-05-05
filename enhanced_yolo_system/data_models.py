"""
Data Models for Detection Performance Enhancement System

Defines core data structures for detections, frames, snapshots, and performance metrics.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np
import time


@dataclass
class Detection:
    """Represents a single object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    color: Tuple[int, int, int] = (255, 0, 0)  # BGR color for visualization
    
    def to_dict(self) -> dict:
        """Convert detection to dictionary"""
        return {
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [int(x) for x in self.bbox],
            "color": [int(x) for x in self.color]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create detection from dictionary"""
        return cls(
            class_id=data["class_id"],
            class_name=data["class_name"],
            confidence=data["confidence"],
            bbox=tuple(data["bbox"]),
            color=tuple(data.get("color", (255, 0, 0)))
        )


@dataclass
class FrameData:
    """Represents a processed frame with metadata"""
    frame: np.ndarray
    detections: List[Detection]
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0
    
    def get_detection_count(self) -> int:
        """Get number of detections in frame"""
        return len(self.detections)
    
    def get_unique_classes(self) -> List[str]:
        """Get list of unique class names detected"""
        return list(set(d.class_name for d in self.detections))
    
    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Filter detections by minimum confidence threshold"""
        return [d for d in self.detections if d.confidence >= min_confidence]


@dataclass
class SnapshotMetadata:
    """Metadata for a captured snapshot"""
    file_path: str
    timestamp: float
    detections: List[Detection]
    description: Optional[str] = None
    description_status: str = "pending"  # pending, analyzing, done, error
    
    def to_dict(self) -> dict:
        """Convert snapshot metadata to dictionary"""
        return {
            "file_path": self.file_path,
            "timestamp": float(self.timestamp),
            "detections": [d.to_dict() for d in self.detections],
            "description": self.description,
            "description_status": self.description_status
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SnapshotMetadata':
        """Create snapshot metadata from dictionary"""
        return cls(
            file_path=data["file_path"],
            timestamp=data["timestamp"],
            detections=[Detection.from_dict(d) for d in data["detections"]],
            description=data.get("description"),
            description_status=data.get("description_status", "pending")
        )
    
    def set_analyzing(self) -> None:
        """Set status to analyzing"""
        self.description_status = "analyzing"
    
    def set_done(self, description: str) -> None:
        """Set status to done with description"""
        self.description = description
        self.description_status = "done"
    
    def set_error(self, error_msg: str) -> None:
        """Set status to error with message"""
        self.description = error_msg
        self.description_status = "error"


@dataclass
class PerformanceMetrics:
    """Performance monitoring data"""
    avg_inference_time_ms: float = 0.0
    avg_latency_ms: float = 0.0
    current_fps: float = 0.0
    frame_drop_rate: float = 0.0
    memory_usage_mb: float = 0.0
    total_frames_processed: int = 0
    
    # Internal tracking
    _inference_times: List[float] = field(default_factory=list, repr=False)
    _latencies: List[float] = field(default_factory=list, repr=False)
    _frame_drops: int = field(default=0, repr=False)
    _last_fps_update: float = field(default_factory=time.time, repr=False)
    _frames_since_last_update: int = field(default=0, repr=False)
    
    def record_inference_time(self, duration_ms: float) -> None:
        """Record single inference duration"""
        self._inference_times.append(duration_ms)
        # Keep only last 100 measurements
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.avg_inference_time_ms = sum(self._inference_times) / len(self._inference_times)
    
    def record_latency(self, duration_ms: float) -> None:
        """Record end-to-end latency"""
        self._latencies.append(duration_ms)
        # Keep only last 100 measurements
        if len(self._latencies) > 100:
            self._latencies.pop(0)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)
    
    def record_frame_drop(self) -> None:
        """Record frame drop event"""
        self._frame_drops += 1
        if self.total_frames_processed > 0:
            self.frame_drop_rate = self._frame_drops / self.total_frames_processed
    
    def record_frame_processed(self) -> None:
        """Record successful frame processing"""
        self.total_frames_processed += 1
        self._frames_since_last_update += 1
        
        # Update FPS every second
        current_time = time.time()
        time_elapsed = current_time - self._last_fps_update
        if time_elapsed >= 1.0:
            self.current_fps = self._frames_since_last_update / time_elapsed
            self._frames_since_last_update = 0
            self._last_fps_update = current_time
        
        # Update frame drop rate
        if self.total_frames_processed > 0:
            self.frame_drop_rate = self._frame_drops / (self.total_frames_processed + self._frame_drops)
    
    def update_memory_usage(self, memory_mb: float) -> None:
        """Update memory usage"""
        self.memory_usage_mb = memory_mb
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return {
            "avg_inference_time_ms": round(self.avg_inference_time_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "current_fps": round(self.current_fps, 2),
            "frame_drop_rate": round(self.frame_drop_rate * 100, 2),  # as percentage
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "total_frames_processed": self.total_frames_processed
        }
    
    def __str__(self) -> str:
        """String representation of metrics"""
        return (f"FPS: {self.current_fps:.1f} | "
                f"Inference: {self.avg_inference_time_ms:.1f}ms | "
                f"Latency: {self.avg_latency_ms:.1f}ms | "
                f"Drops: {self.frame_drop_rate*100:.1f}% | "
                f"Memory: {self.memory_usage_mb:.1f}MB")


@dataclass
class DetectionConfig:
    """Detection configuration parameters"""
    confidence_threshold: float = 0.5  # 0.3 - 0.9
    nms_threshold: float = 0.4  # 0.2 - 0.6
    input_resolution: int = 416  # 320, 416, or 608
    backend: str = "opencv"  # opencv, cuda, openvino
    target: str = "cpu"  # cpu, cuda, opencl
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        errors = []
        
        if not (0.3 <= self.confidence_threshold <= 0.9):
            errors.append("confidence_threshold must be between 0.3 and 0.9")
        
        if not (0.2 <= self.nms_threshold <= 0.6):
            errors.append("nms_threshold must be between 0.2 and 0.6")
        
        if self.input_resolution not in [320, 416, 608]:
            errors.append("input_resolution must be 320, 416, or 608")
        
        if self.backend not in ["opencv", "cuda", "openvino"]:
            errors.append("backend must be 'opencv', 'cuda', or 'openvino'")
        
        if self.target not in ["cpu", "cuda", "opencl"]:
            errors.append("target must be 'cpu', 'cuda', or 'opencl'")
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DetectionConfig':
        """Create config from dictionary"""
        return cls(
            confidence_threshold=data.get("confidence_threshold", 0.5),
            nms_threshold=data.get("nms_threshold", 0.4),
            input_resolution=data.get("input_resolution", 416),
            backend=data.get("backend", "opencv"),
            target=data.get("target", "cpu")
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "input_resolution": self.input_resolution,
            "backend": self.backend,
            "target": self.target
        }


if __name__ == "__main__":
    # Example usage
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=(100, 100, 50, 100),
        color=(255, 0, 0)
    )
    print("Detection:", detection)
    
    frame_data = FrameData(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[detection],
        frame_id=1
    )
    print(f"Frame has {frame_data.get_detection_count()} detections")
    
    metrics = PerformanceMetrics()
    metrics.record_inference_time(50.0)
    metrics.record_frame_processed()
    print("Metrics:", metrics)
