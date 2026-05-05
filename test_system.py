"""
Simple test script to verify the enhanced detection system components
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)

def test_config_manager():
    """Test configuration manager"""
    print("\n=== Testing ConfigManager ===")
    try:
        from config_manager import ConfigManager
        manager = ConfigManager("test_config.json")
        config = manager.load_config()
        print(f"✓ Config loaded: {len(config)} sections")
        
        # Test validation
        is_valid, errors = manager.validate_config(config)
        print(f"✓ Config validation: {'passed' if is_valid else 'failed'}")
        
        return True
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

def test_data_models():
    """Test data models"""
    print("\n=== Testing Data Models ===")
    try:
        from data_models import Detection, FrameData, SnapshotMetadata, PerformanceMetrics, DetectionConfig
        import numpy as np
        
        # Test Detection
        detection = Detection(0, "person", 0.95, (100, 100, 50, 100))
        print(f"✓ Detection created: {detection.class_name}")
        
        # Test FrameData
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_data = FrameData(frame, [detection])
        print(f"✓ FrameData created with {frame_data.get_detection_count()} detections")
        
        # Test PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.record_inference_time(50.0)
        metrics.record_frame_processed()
        print(f"✓ PerformanceMetrics: {metrics.current_fps:.1f} FPS")
        
        # Test DetectionConfig
        config = DetectionConfig()
        is_valid, errors = config.validate()
        print(f"✓ DetectionConfig validation: {'passed' if is_valid else 'failed'}")
        
        return True
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        return False

def test_snapshot_manager():
    """Test snapshot manager"""
    print("\n=== Testing SnapshotManager ===")
    try:
        from snapshot_manager import SnapshotManager
        from data_models import Detection
        import numpy as np
        import time
        
        manager = SnapshotManager("test_snapshots")
        print(f"✓ SnapshotManager initialized")
        
        # Test snapshot capture
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_detection = Detection(0, "test", 0.9, (100, 100, 50, 50))
        snapshot_path = manager.capture_snapshot(test_frame, [test_detection])
        print(f"✓ Snapshot captured: {snapshot_path}")
        
        # Wait for async save
        time.sleep(0.5)
        
        # Test history buffer
        history = manager.get_detection_history()
        print(f"✓ Detection history: {len(history)} frames")
        
        return True
    except Exception as e:
        print(f"✗ SnapshotManager test failed: {e}")
        return False

def test_performance_monitor():
    """Test performance monitor"""
    print("\n=== Testing PerformanceMonitor ===")
    try:
        from performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(enable_profiling=False)
        print(f"✓ PerformanceMonitor initialized")
        
        # Record some metrics
        for i in range(10):
            monitor.record_inference_time(50.0 + i)
            monitor.record_frame_processed()
        
        metrics = monitor.get_metrics()
        print(f"✓ Metrics: FPS={metrics.current_fps:.1f}, Inference={metrics.avg_inference_time_ms:.1f}ms")
        
        return True
    except Exception as e:
        print(f"✗ PerformanceMonitor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Enhanced Detection System - Component Tests")
    print("=" * 60)
    
    tests = [
        test_config_manager,
        test_data_models,
        test_snapshot_manager,
        test_performance_monitor
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! System is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
