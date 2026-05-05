"""
Enhanced Object Detection System

Main application integrating all components for optimized real-time object detection
with AI-powered scene descriptions.
"""

import cv2
import argparse
import sys
import time
import logging
import numpy as np
import os
from typing import Optional, Tuple

# Import our modules
from config_manager import ConfigManager
from data_models import DetectionConfig, Detection
from yolo_detector import YOLODetector
from frame_pipeline import FramePipeline
from snapshot_manager import SnapshotManager
from performance_monitor import PerformanceMonitor
from ai_description_generator import AIDescriptionGenerator
import threading
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UIRenderer:
    """Handles all UI rendering and visual feedback"""
    
    def __init__(self):
        """Initialize UI renderer"""
        self.show_description_panel = False
        self.show_help = False
        self.snapshot_flash_frames = 0
        self.current_description = ""
        self.description_status = ""
        self.current_snapshot_name = ""
    
    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x, y, w, h = detection.bbox
            color = detection.color
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x + 3, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def draw_status_bar(self, frame: np.ndarray, fps: float, object_count: int, 
                       inference_time: float) -> np.ndarray:
        """Draw status bar with FPS and object count"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_text = (f"FPS: {fps:.1f} | Objects: {object_count} | "
                      f"Inference: {inference_time:.1f}ms | "
                      f"[S] Snapshot [H] Help [D] Descriptions [P] Pause [Q] Quit")
        
        cv2.putText(frame, status_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_snapshot_flash(self, frame: np.ndarray) -> np.ndarray:
        """Draw visual feedback for snapshot capture"""
        if self.snapshot_flash_frames > 0:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), 10)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            self.snapshot_flash_frames -= 1
        return frame
    
    def draw_help_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw keyboard shortcuts help overlay"""
        if not self.show_help:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        panel_w, panel_h = 400, 280
        px = (w - panel_w) // 2
        py = (h - panel_h) // 2
        
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Title
        cv2.rectangle(frame, (px, py), (px + panel_w, py + 30), (0, 120, 200), -1)
        cv2.putText(frame, "Keyboard Shortcuts", (px + 10, py + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Shortcuts
        shortcuts = [
            "S - Capture snapshot + AI description",
            "D - Toggle description panel",
            "H - Toggle this help",
            "P - Pause/Resume detection",
            "Q - Quit application",
            "",
            "Press H to close"
        ]
        
        y_offset = py + 50
        for shortcut in shortcuts:
            cv2.putText(frame, shortcut, (px + 20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        return frame
    
    def draw_description_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw AI description panel"""
        if not self.show_description_panel or not self.current_description:
            return frame
        
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_w = min(500, w - 40)
        panel_h = min(400, h - 40)
        px = w - panel_w - 20
        py = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title bar color by status
        if self.description_status == "done":
            title_color = (30, 180, 80)
            title_text = "AI Scene Description"
        elif self.description_status == "analyzing":
            title_color = (200, 140, 0)
            title_text = "Analyzing..."
        else:
            title_color = (0, 50, 200)
            title_text = "AI Description"
        
        # Title bar
        cv2.rectangle(frame, (px, py), (px + panel_w, py + 28), title_color, -1)
        cv2.putText(frame, title_text, (px + 10, py + 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Snapshot name
        if self.current_snapshot_name:
            cv2.putText(frame, f"Snapshot: {self.current_snapshot_name}", (px + 10, py + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Border
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), title_color, 2)
        
        # Description text (word-wrapped)
        lines = []
        for line in self.current_description.split('\n'):
            if len(line) > 55:
                # Simple word wrap
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 55:
                        current_line += word + " "
                    else:
                        if current_line:
                            lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
            else:
                lines.append(line)
        
        # Draw description lines
        y_offset = py + 75
        line_height = 20
        max_lines = (panel_h - 85) // line_height
        
        for i, line in enumerate(lines[:max_lines]):
            cv2.putText(frame, line, (px + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            y_offset += line_height
        
        return frame
    
    def trigger_snapshot_flash(self) -> None:
        """Trigger snapshot flash effect"""
        self.snapshot_flash_frames = 3


class EnhancedDetectionSystem:
    """Main application class"""
    
    def __init__(self, args):
        """Initialize the detection system"""
        self.args = args
        
        # Load configuration
        self.config_manager = ConfigManager(args.config if args.config else "config.json")
        self.config = self.config_manager.load_config()
        
        # Create detection config
        detection_config = DetectionConfig.from_dict(self.config["detection"])
        
        # Initialize components
        logging.info("Initializing YOLO detector...")
        self.detector = YOLODetector(
            config_path=args.yolo_config,
            weights_path=args.yolo_weights,
            classes_path=args.classes,
            config=detection_config
        )
        
        logging.info("Initializing frame pipeline...")
        self.pipeline = FramePipeline(
            detector=self.detector,
            camera_id=args.camera,
            queue_size=self.config["performance"]["frame_queue_size"]
        )
        
        logging.info("Initializing snapshot manager...")
        self.snapshot_manager = SnapshotManager(
            snapshot_dir=self.config["snapshot"]["directory"],
            jpeg_quality=self.config["snapshot"]["jpeg_quality"]
        )
        
        logging.info("Initializing performance monitor...")
        self.performance_monitor = PerformanceMonitor(
            enable_profiling=self.config["performance"]["enable_profiling"],
            log_interval=self.config["performance"]["log_interval_seconds"]
        )
        
        logging.info("Initializing AI description generator...")
        self.ai_generator = AIDescriptionGenerator(
            provider=self.config["ai_description"]["provider"],
            model=self.config["ai_description"]["model"],
            timeout_seconds=self.config["ai_description"]["timeout_seconds"],
            max_retries=self.config["ai_description"]["max_retries"],
            cache_enabled=self.config["ai_description"]["cache_enabled"]
        )
        
        # Description processing queue and thread
        self.description_queue = queue.Queue()
        self.description_thread = threading.Thread(target=self._description_worker, daemon=True)
        self.description_thread.start()
        
        # UI renderer
        self.ui_renderer = UIRenderer()
        
        # Control flags
        self.running = False
        self.paused = False
        
        logging.info("Enhanced Detection System initialized successfully")
    
    def run(self) -> None:
        """Main application loop"""
        # Start pipeline
        if not self.pipeline.start():
            logging.error("Failed to start frame pipeline")
            return
        
        self.running = True
        logging.info("Detection system started. Press 'H' for help.")
        
        try:
            while self.running:
                # Get latest frame
                frame, detections = self.pipeline.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Record metrics
                self.performance_monitor.record_frame_processed()
                
                # Draw detections
                frame = self.ui_renderer.draw_detections(frame, detections)
                
                # Draw status bar
                metrics = self.performance_monitor.get_metrics()
                frame = self.ui_renderer.draw_status_bar(
                    frame,
                    metrics.current_fps,
                    len(detections),
                    metrics.avg_inference_time_ms
                )
                
                # Draw snapshot flash
                frame = self.ui_renderer.draw_snapshot_flash(frame)
                
                # Draw description panel
                frame = self.ui_renderer.draw_description_panel(frame)
                
                # Draw help overlay
                frame = self.ui_renderer.draw_help_overlay(frame)
                
                # Display frame
                cv2.imshow("Enhanced Object Detection System", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard(key, frame, detections):
                    break
        
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def handle_keyboard(self, key: int, frame: np.ndarray, detections: list) -> bool:
        """
        Handle keyboard input
        
        Returns:
            True to continue, False to exit
        """
        if key == ord('q'):
            logging.info("Quit requested")
            return False
        
        elif key == ord('s'):
            # Capture snapshot
            snapshot_path = self.snapshot_manager.capture_snapshot(frame, detections)
            self.ui_renderer.trigger_snapshot_flash()
            logging.info(f"Snapshot captured: {snapshot_path}")
            
            # Queue for AI description
            self.description_queue.put((snapshot_path, detections))
            self.ui_renderer.description_status = "analyzing"
            self.ui_renderer.current_description = "Generating AI description... Please wait."
            self.ui_renderer.current_snapshot_name = os.path.basename(snapshot_path)
            self.ui_renderer.show_description_panel = True
        
        elif key == ord('d'):
            # Toggle description panel
            self.ui_renderer.show_description_panel = not self.ui_renderer.show_description_panel
        
        elif key == ord('h'):
            # Toggle help
            self.ui_renderer.show_help = not self.ui_renderer.show_help
        
        elif key == ord('p'):
            # Toggle pause
            self.paused = self.pipeline.toggle_pause()
        
        return True
    
    def _description_worker(self) -> None:
        """Worker thread for AI description generation"""
        logging.info("AI description thread started")
        
        while True:
            try:
                # Get snapshot from queue
                snapshot_path, detections = self.description_queue.get(timeout=1.0)
                
                logging.info(f"Generating AI description for {os.path.basename(snapshot_path)}...")
                
                # Generate description
                description = self.ai_generator.generate_description(snapshot_path, detections)
                
                # Update UI
                self.ui_renderer.current_description = description
                self.ui_renderer.description_status = "done"
                
                logging.info(f"AI description ready for {os.path.basename(snapshot_path)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error generating description: {e}")
                self.ui_renderer.current_description = f"Error generating description: {str(e)}"
                self.ui_renderer.description_status = "error"
    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        logging.info("Shutting down...")
        
        # Stop pipeline
        self.pipeline.stop()
        
        # Save snapshot metadata
        self.snapshot_manager.save_metadata_cache()
        
        # Log final metrics
        logging.info(f"Final metrics: {self.performance_monitor.get_metrics()}")
        
        # Close windows
        cv2.destroyAllWindows()
        
        logging.info("Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Object Detection System")
    parser.add_argument('-c', '--yolo-config', required=True, help='Path to YOLO config file')
    parser.add_argument('-w', '--yolo-weights', required=True, help='Path to YOLO weights file')
    parser.add_argument('-cl', '--classes', required=True, help='Path to class names file')
    parser.add_argument('--config', help='Path to system config file (default: config.json)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    try:
        system = EnhancedDetectionSystem(args)
        system.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
