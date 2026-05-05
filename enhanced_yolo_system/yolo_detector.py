"""
YOLO Detection Engine with Performance Optimization

Implements optimized YOLO object detection with GPU acceleration support,
configurable parameters, and model reuse.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple
from data_models import Detection, DetectionConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class YOLODetector:
    """YOLO-based object detector with optimization features"""
    
    def __init__(self, config_path: str, weights_path: str, classes_path: str, config: DetectionConfig):
        """
        Initialize YOLO model with configuration
        
        Args:
            config_path: Path to YOLO config file (.cfg)
            weights_path: Path to YOLO weights file (.weights)
            classes_path: Path to class names file (.txt)
            config: Detection configuration
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.config = config
        
        # Load class names
        self.classes = self._load_classes()
        
        # Generate colors for each class (seeded for reproducibility)
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # Load YOLO network
        self.net = self._load_network()
        
        # Get output layer names
        self.output_layers = self._get_output_layers()
        
        # Try to enable GPU acceleration
        self.gpu_enabled = self.enable_gpu()
        
        logging.info(f"YOLODetector initialized with {len(self.classes)} classes")
        logging.info(f"Input resolution: {self.config.input_resolution}x{self.config.input_resolution}")
        logging.info(f"Confidence threshold: {self.config.confidence_threshold}")
        logging.info(f"NMS threshold: {self.config.nms_threshold}")
    
    def _load_classes(self) -> List[str]:
        """Load class names from file"""
        try:
            with open(self.classes_path, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
        except Exception as e:
            logging.error(f"Failed to load classes from {self.classes_path}: {e}")
            raise
    
    def _load_network(self) -> cv2.dnn.Net:
        """Load YOLO network from config and weights"""
        try:
            net = cv2.dnn.readNet(self.weights_path, self.config_path)
            return net
        except Exception as e:
            logging.error(f"Failed to load YOLO network: {e}")
            raise
    
    def _get_output_layers(self) -> List[str]:
        """Get output layer names from network"""
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.x
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # OpenCV 3.x
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def enable_gpu(self) -> bool:
        """
        Enable GPU acceleration if available
        
        Returns:
            True if GPU enabled, False otherwise
        """
        try:
            # Check if CUDA is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logging.info("GPU acceleration enabled (CUDA)")
                return True
        except:
            pass
        
        # Fallback to CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logging.info("Using CPU backend")
        return False
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update confidence threshold
        
        Args:
            threshold: New confidence threshold (0.3 - 0.9)
        """
        if 0.3 <= threshold <= 0.9:
            self.config.confidence_threshold = threshold
            logging.info(f"Confidence threshold updated to {threshold}")
        else:
            logging.warning(f"Invalid confidence threshold {threshold}, must be between 0.3 and 0.9")
    
    def set_nms_threshold(self, threshold: float) -> None:
        """
        Update NMS threshold
        
        Args:
            threshold: New NMS threshold (0.2 - 0.6)
        """
        if 0.2 <= threshold <= 0.6:
            self.config.nms_threshold = threshold
            logging.info(f"NMS threshold updated to {threshold}")
        else:
            logging.warning(f"Invalid NMS threshold {threshold}, must be between 0.2 and 0.6")
    
    def set_input_resolution(self, resolution: int) -> None:
        """
        Update input resolution
        
        Args:
            resolution: New input resolution (320, 416, or 608)
        """
        if resolution in [320, 416, 608]:
            self.config.input_resolution = resolution
            logging.info(f"Input resolution updated to {resolution}x{resolution}")
        else:
            logging.warning(f"Invalid resolution {resolution}, must be 320, 416, or 608")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on frame, return list of detections
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of Detection objects
        """
        height, width = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=(self.config.input_resolution, self.config.input_resolution),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
        
        # Set input and run forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                # Filter by confidence threshold
                if confidence > self.config.confidence_threshold:
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = center_x - w // 2
                    y = center_y - h // 2
                    
                    class_ids.append(class_id)
                    confidences.append(confidence)
                    boxes.append([x, y, w, h])
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.config.confidence_threshold,
            self.config.nms_threshold
        )
        
        # Create Detection objects
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                class_id = class_ids[i]
                confidence = confidences[i]
                box = boxes[i]
                
                detection = Detection(
                    class_id=class_id,
                    class_name=self.classes[class_id],
                    confidence=confidence,
                    bbox=tuple(box),
                    color=tuple(self.colors[class_id].tolist())
                )
                detections.append(detection)
        
        return detections
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get color for a specific class
        
        Args:
            class_name: Name of the class
            
        Returns:
            BGR color tuple
        """
        try:
            class_id = self.classes.index(class_name)
            return tuple(self.colors[class_id].tolist())
        except ValueError:
            return (255, 255, 255)  # White for unknown classes


if __name__ == "__main__":
    # Example usage
    config = DetectionConfig(
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_resolution=416,
        backend="opencv",
        target="cpu"
    )
    
    try:
        detector = YOLODetector(
            config_path="yolov3.cfg",
            weights_path="yolov3.weights",
            classes_path="yolov3.txt",
            config=config
        )
        
        # Test with a dummy frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(test_frame)
        print(f"Detected {len(detections)} objects")
        
    except Exception as e:
        print(f"Error: {e}")
