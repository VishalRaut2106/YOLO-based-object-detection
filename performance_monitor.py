"""
Performance Monitoring System

Tracks and reports performance metrics including inference time, latency,
FPS, frame drops, and memory usage.
"""

import time
import logging
import psutil
from data_models import PerformanceMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PerformanceMonitor:
    """Tracks and reports system performance metrics"""
    
    def __init__(self, enable_profiling: bool = False, log_interval: float = 10.0):
        """
        Initialize performance monitoring
        
        Args:
            enable_profiling: Enable detailed performance profiling
            log_interval: Interval in seconds for periodic logging
        """
        self.enable_profiling = enable_profiling
        self.log_interval = log_interval
        
        # Metrics
        self.metrics = PerformanceMetrics()
        
        # Timing
        self.last_log_time = time.time()
        
        # Process for memory tracking
        self.process = psutil.Process()
        
        logging.info(f"Performance monitor initialized (profiling: {enable_profiling})")
    
    def record_inference_time(self, duration_ms: float) -> None:
        """
        Record single inference duration
        
        Args:
            duration_ms: Inference duration in milliseconds
        """
        self.metrics.record_inference_time(duration_ms)
    
    def record_frame_drop(self) -> None:
        """Record frame drop event"""
        self.metrics.record_frame_drop()
    
    def record_end_to_end_latency(self, duration_ms: float) -> None:
        """
        Record capture-to-display latency
        
        Args:
            duration_ms: End-to-end latency in milliseconds
        """
        self.metrics.record_latency(duration_ms)
    
    def record_frame_processed(self) -> None:
        """Record successful frame processing"""
        self.metrics.record_frame_processed()
        
        # Update memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.metrics.update_memory_usage(memory_mb)
        
        # Periodic logging
        if self.enable_profiling:
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                self.log_metrics()
                self.last_log_time = current_time
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics
        
        Returns:
            PerformanceMetrics object
        """
        return self.metrics
    
    def log_metrics(self) -> None:
        """Log metrics to console"""
        logging.info(f"Performance: {self.metrics}")
    
    def get_metrics_dict(self) -> dict:
        """
        Get metrics as dictionary
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = PerformanceMetrics()
        logging.info("Performance metrics reset")


if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor(enable_profiling=True, log_interval=5.0)
    
    # Simulate some processing
    for i in range(100):
        monitor.record_inference_time(50.0 + i % 10)
        monitor.record_frame_processed()
        time.sleep(0.05)
    
    print("\nFinal metrics:")
    print(monitor.get_metrics())
