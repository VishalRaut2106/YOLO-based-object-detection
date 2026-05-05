"""
Snapshot Manager with Asynchronous Saving

Handles snapshot capture, saving, and queuing for AI description generation.
"""

import cv2
import os
import time
import threading
import queue
import logging
import json
from typing import List, Optional
from data_models import SnapshotMetadata, Detection
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SnapshotManager:
    """Manages snapshot capture and asynchronous saving"""
    
    def __init__(self, snapshot_dir: str = "snapshots", jpeg_quality: int = 85):
        """
        Initialize snapshot manager
        
        Args:
            snapshot_dir: Directory to save snapshots
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.snapshot_dir = snapshot_dir
        self.jpeg_quality = jpeg_quality
        
        # Create snapshot directory if it doesn't exist
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Queue for asynchronous saving
        self.save_queue = queue.Queue()
        
        # Queue for description requests
        self.description_queue = queue.Queue()
        
        # Snapshot metadata cache
        self.snapshots: List[SnapshotMetadata] = []
        
        # Detection history buffer (max 10 frames)
        self.detection_history: List[List[Detection]] = []
        self.max_history_size = 10
        
        # Save thread
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        logging.info(f"Snapshot manager initialized (dir: {snapshot_dir}, quality: {jpeg_quality})")
    
    def capture_snapshot(self, frame: np.ndarray, detections: List[Detection]) -> str:
        """
        Save snapshot with timestamp, return file path
        
        Args:
            frame: Frame to save
            detections: List of detections in frame
            
        Returns:
            Path to saved snapshot
        """
        # Generate timestamp-based filename
        timestamp = time.time()
        filename = time.strftime("snap_%Y%m%d_%H%M%S.jpg", time.localtime(timestamp))
        file_path = os.path.join(self.snapshot_dir, filename)
        
        # Create snapshot metadata
        metadata = SnapshotMetadata(
            file_path=file_path,
            timestamp=timestamp,
            detections=detections.copy()
        )
        
        # Add to snapshots list
        self.snapshots.append(metadata)
        
        # Add to detection history
        self._add_to_history(detections)
        
        # Queue for asynchronous saving
        self.save_queue.put((frame.copy(), file_path))
        
        logging.info(f"Snapshot queued: {filename}")
        return file_path
    
    def queue_for_description(self, snapshot_path: str, detections: List[Detection]) -> None:
        """
        Queue snapshot for AI description generation
        
        Args:
            snapshot_path: Path to snapshot file
            detections: List of detections in snapshot
        """
        self.description_queue.put((snapshot_path, detections))
    
    def get_description_queue(self) -> queue.Queue:
        """
        Get description queue for processing
        
        Returns:
            Description queue
        """
        return self.description_queue
    
    def update_snapshot_description(self, snapshot_path: str, description: str, status: str = "done") -> None:
        """
        Update snapshot with description
        
        Args:
            snapshot_path: Path to snapshot
            description: Generated description
            status: Description status
        """
        for snapshot in self.snapshots:
            if snapshot.file_path == snapshot_path:
                if status == "done":
                    snapshot.set_done(description)
                elif status == "error":
                    snapshot.set_error(description)
                elif status == "analyzing":
                    snapshot.set_analyzing()
                break
    
    def get_snapshot_metadata(self, snapshot_path: str) -> Optional[SnapshotMetadata]:
        """
        Get metadata for a snapshot
        
        Args:
            snapshot_path: Path to snapshot
            
        Returns:
            SnapshotMetadata or None
        """
        for snapshot in self.snapshots:
            if snapshot.file_path == snapshot_path:
                return snapshot
        return None
    
    def get_snapshot_list(self) -> List[str]:
        """
        Get list of all snapshots in directory
        
        Returns:
            List of snapshot file paths
        """
        snapshots = []
        if os.path.exists(self.snapshot_dir):
            for filename in os.listdir(self.snapshot_dir):
                if filename.endswith('.jpg') and filename.startswith('snap_'):
                    snapshots.append(os.path.join(self.snapshot_dir, filename))
        return sorted(snapshots)
    
    def process_existing_snapshots(self) -> List[str]:
        """
        Batch process existing snapshots in directory
        
        Returns:
            List of snapshot paths to process
        """
        existing_snapshots = self.get_snapshot_list()
        to_process = []
        
        for snapshot_path in existing_snapshots:
            # Check if already has cached description
            metadata = self.get_snapshot_metadata(snapshot_path)
            if metadata is None or metadata.description is None:
                to_process.append(snapshot_path)
        
        logging.info(f"Found {len(to_process)} snapshots to process")
        return to_process
    
    def _add_to_history(self, detections: List[Detection]) -> None:
        """
        Add detections to history buffer (max 10 frames)
        
        Args:
            detections: List of detections to add
        """
        self.detection_history.append(detections.copy())
        
        # Keep only last 10 frames
        if len(self.detection_history) > self.max_history_size:
            self.detection_history.pop(0)
    
    def get_detection_history(self) -> List[List[Detection]]:
        """
        Get detection history buffer
        
        Returns:
            List of detection lists
        """
        return self.detection_history.copy()
    
    def _save_worker(self) -> None:
        """Worker thread for asynchronous snapshot saving"""
        logging.info("Snapshot save thread started")
        
        while True:
            try:
                # Get frame and path from queue
                frame, file_path = self.save_queue.get(timeout=1.0)
                
                # Save with JPEG compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                success = cv2.imwrite(file_path, frame, encode_param)
                
                if success:
                    logging.info(f"Snapshot saved: {os.path.basename(file_path)}")
                else:
                    logging.error(f"Failed to save snapshot: {file_path}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error saving snapshot: {e}")
    
    def save_metadata_cache(self, cache_file: str = "snapshot_cache.json") -> None:
        """
        Save snapshot metadata to cache file
        
        Args:
            cache_file: Path to cache file
        """
        cache_path = os.path.join(self.snapshot_dir, cache_file)
        try:
            data = [s.to_dict() for s in self.snapshots]
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Metadata cache saved: {cache_path}")
        except Exception as e:
            logging.error(f"Failed to save metadata cache: {e}")
    
    def load_metadata_cache(self, cache_file: str = "snapshot_cache.json") -> None:
        """
        Load snapshot metadata from cache file
        
        Args:
            cache_file: Path to cache file
        """
        cache_path = os.path.join(self.snapshot_dir, cache_file)
        if not os.path.exists(cache_path):
            return
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            self.snapshots = [SnapshotMetadata.from_dict(d) for d in data]
            logging.info(f"Loaded {len(self.snapshots)} snapshots from cache")
        except Exception as e:
            logging.error(f"Failed to load metadata cache: {e}")


if __name__ == "__main__":
    # Example usage
    manager = SnapshotManager()
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Snapshot", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Capture snapshot
    from data_models import Detection
    test_detection = Detection(0, "test", 0.9, (100, 100, 50, 50))
    snapshot_path = manager.capture_snapshot(test_frame, [test_detection])
    
    print(f"Snapshot captured: {snapshot_path}")
    
    # Wait for save to complete
    time.sleep(1)
    
    # List snapshots
    snapshots = manager.get_snapshot_list()
    print(f"Total snapshots: {len(snapshots)}")
