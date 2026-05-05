"""
Frame Processing Pipeline with Threading

Implements multi-threaded frame capture and detection processing with
queue-based communication and graceful shutdown.
"""

import cv2
import threading
import queue
import time
import logging
import numpy as np
from typing import Optional, Tuple, List
from data_models import FrameData, Detection
from yolo_detector import YOLODetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FramePipeline:
    """Manages frame capture and detection processing pipeline"""
    
    def __init__(self, detector: YOLODetector, camera_id: int = 0, queue_size: int = 3):
        """
        Initialize pipeline with detector and configuration
        
        Args:
            detector: YOLODetector instance
            camera_id: Camera device ID
            queue_size: Maximum size of frame queue
        """
        self.detector = detector
        self.camera_id = camera_id
        self.queue_size = queue_size
        
        # Frame queue for thread communication
        self.frame_queue = queue.Queue(maxsize=queue_size)
        
        # Latest processed frame data
        self.latest_frame_data: Optional[FrameData] = None
        self.frame_lock = threading.Lock()
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Threads
        self.capture_thread: Optional[threading.Thread] = None
        self.detection_thread: Optional[threading.Thread] = None
        
        # Camera
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Frame counter
        self.frame_id = 0
        
        # Statistics
        self.frames_dropped = 0
    
    def start(self) -> bool:
        """
        Start capture and detection threads
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logging.warning("Pipeline already running")
            return False
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start threads
        self.running = True
        self.paused = False
        
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        
        self.capture_thread.start()
        self.detection_thread.start()
        
        logging.info("Frame pipeline started")
        return True
    
    def stop(self) -> None:
        """Gracefully stop all threads"""
        if not self.running:
            return
        
        logging.info("Stopping frame pipeline...")
        self.running = False
        
        # Wait for threads to finish (with timeout)
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logging.info("Frame pipeline stopped")
    
    def pause(self) -> None:
        """Pause detection processing"""
        self.paused = True
        logging.info("Pipeline paused")
    
    def resume(self) -> None:
        """Resume detection processing"""
        self.paused = False
        logging.info("Pipeline resumed")
    
    def toggle_pause(self) -> bool:
        """
        Toggle pause state
        
        Returns:
            New pause state
        """
        self.paused = not self.paused
        logging.info(f"Pipeline {'paused' if self.paused else 'resumed'}")
        return self.paused
    
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], List[Detection]]:
        """
        Get most recent annotated frame and detections
        
        Returns:
            Tuple of (frame, detections)
        """
        with self.frame_lock:
            if self.latest_frame_data is None:
                return None, []
            return self.latest_frame_data.frame.copy(), self.latest_frame_data.detections.copy()
    
    def get_latest_frame_data(self) -> Optional[FrameData]:
        """
        Get most recent FrameData object
        
        Returns:
            FrameData or None
        """
        with self.frame_lock:
            return self.latest_frame_data
    
    def _capture_worker(self) -> None:
        """Worker thread for frame capture"""
        logging.info("Capture thread started")
        
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                logging.error("Camera not available")
                break
            
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Failed to read frame")
                time.sleep(0.01)
                continue
            
            # Try to put frame in queue (non-blocking)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Queue is full, drop oldest frame
                try:
                    self.frame_queue.get(block=False)
                    self.frames_dropped += 1
                    self.frame_queue.put(frame, block=False)
                except:
                    pass
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
        
        logging.info("Capture thread stopped")
    
    def _detection_worker(self) -> None:
        """Worker thread for detection processing"""
        logging.info("Detection thread started")
        
        while self.running:
            # Skip processing if paused
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Run detection
            try:
                start_time = time.time()
                detections = self.detector.detect(frame)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Create FrameData
                self.frame_id += 1
                frame_data = FrameData(
                    frame=frame,
                    detections=detections,
                    timestamp=time.time(),
                    frame_id=self.frame_id
                )
                
                # Update latest frame data
                with self.frame_lock:
                    self.latest_frame_data = frame_data
                
            except Exception as e:
                logging.error(f"Detection error: {e}")
        
        logging.info("Detection thread stopped")
    
    def get_stats(self) -> dict:
        """
        Get pipeline statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            "frames_processed": self.frame_id,
            "frames_dropped": self.frames_dropped,
            "queue_size": self.frame_queue.qsize(),
            "running": self.running,
            "paused": self.paused
        }


if __name__ == "__main__":
    # Example usage
    from data_models import DetectionConfig
    
    config = DetectionConfig(
        confidence_threshold=0.5,
        nms_threshold=0.4,
        input_resolution=416
    )
    
    try:
        detector = YOLODetector(
            config_path="yolov3.cfg",
            weights_path="yolov3.weights",
            classes_path="yolov3.txt",
            config=config
        )
        
        pipeline = FramePipeline(detector, camera_id=0, queue_size=3)
        
        if pipeline.start():
            print("Pipeline started. Press Ctrl+C to stop...")
            
            try:
                while True:
                    frame, detections = pipeline.get_latest_frame()
                    if frame is not None:
                        print(f"Frame with {len(detections)} detections")
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                pipeline.stop()
        else:
            print("Failed to start pipeline")
            
    except Exception as e:
        print(f"Error: {e}")
