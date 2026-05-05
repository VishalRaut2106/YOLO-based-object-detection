"""
AI Description Generator with Retry Logic

Generates natural language descriptions of detected objects using Gemini API.
"""

import time
import logging
import os
from typing import List, Optional, Callable
from PIL import Image
import google.generativeai as genai
from data_models import Detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AIDescriptionGenerator:
    """Generates AI descriptions for snapshots with detected objects"""
    
    def __init__(self, provider: str = "gemini", model: str = "gemini-2.0-flash", 
                 timeout_seconds: int = 10, max_retries: int = 2, cache_enabled: bool = True):
        """
        Initialize AI description generator
        
        Args:
            provider: AI provider (gemini or llama)
            model: Model name
            timeout_seconds: Timeout for API calls
            max_retries: Maximum retry attempts
            cache_enabled: Enable description caching
        """
        self.provider = provider
        self.model_name = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        
        # Description cache
        self.cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 2 requests per second max
        
        # Initialize Gemini
        if provider == "gemini":
            api_key = "AIzaSyCAWy42SCD5KFYE2NkpRMXSNLiCiAHI4nA"
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logging.info(f"AI Description Generator initialized with {model}")
        else:
            logging.warning(f"Provider {provider} not yet implemented")
            self.model = None
    
    def generate_description(self, snapshot_path: str, detections: List[Detection]) -> str:
        """
        Generate description for snapshot with detections
        
        Args:
            snapshot_path: Path to snapshot image
            detections: List of detections in snapshot
            
        Returns:
            Generated description string
        """
        # Check cache first
        if self.cache_enabled and snapshot_path in self.cache:
            logging.info(f"Using cached description for {os.path.basename(snapshot_path)}")
            return self.cache[snapshot_path]
        
        # Check if model is available
        if self.model is None:
            return self._generate_fallback_description(detections)
        
        # Rate limiting
        self._apply_rate_limit()
        
        # Try to generate description with retries
        for attempt in range(self.max_retries + 1):
            try:
                description = self._call_api(snapshot_path, detections)
                
                # Cache the description
                if self.cache_enabled:
                    self.cache[snapshot_path] = description
                
                return description
                
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    logging.warning(f"API error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logging.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"API error after {self.max_retries + 1} attempts: {e}")
                    return self._generate_fallback_description(detections)
        
        return self._generate_fallback_description(detections)
    
    def generate_description_async(self, snapshot_path: str, detections: List[Detection],
                                  callback: Callable[[str], None]) -> None:
        """
        Generate description asynchronously with callback
        
        Args:
            snapshot_path: Path to snapshot
            detections: List of detections
            callback: Function to call with description
        """
        import threading
        
        def worker():
            description = self.generate_description(snapshot_path, detections)
            callback(description)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _call_api(self, snapshot_path: str, detections: List[Detection]) -> str:
        """
        Call Gemini API to generate description
        
        Args:
            snapshot_path: Path to snapshot
            detections: List of detections
            
        Returns:
            Generated description
        """
        # Load image
        image = Image.open(snapshot_path)
        
        # Build prompt with detected objects
        object_counts = {}
        for detection in detections:
            object_counts[detection.class_name] = object_counts.get(detection.class_name, 0) + 1
        
        if object_counts:
            objects_text = ", ".join(
                f"{count} {name}{'s' if count > 1 else ''}" 
                for name, count in object_counts.items()
            )
        else:
            objects_text = "no clearly identifiable objects"
        
        prompt = f"""You are analyzing a camera snapshot. An object detection system identified these objects: {objects_text}.

Based on these detected objects and the image, please provide:

1. Scene Summary: A natural, friendly 1-2 sentence description of what this scene shows.

2. Detected Objects:
   - List each type of object with its count

3. Context: One sentence about the likely setting or activity.

Keep it concise and conversational."""
        
        # Generate description
        response = self.model.generate_content([image, prompt])
        description = response.text.strip()
        
        return description
    
    def _generate_fallback_description(self, detections: List[Detection]) -> str:
        """
        Generate fallback description without API
        
        Args:
            detections: List of detections
            
        Returns:
            Simple description based on detections
        """
        if not detections:
            return "Scene Summary: No objects detected in this snapshot.\n\nDetected Objects: None\n\nContext: The scene appears to be empty or the objects are not clearly visible."
        
        # Count objects
        object_counts = {}
        for detection in detections:
            object_counts[detection.class_name] = object_counts.get(detection.class_name, 0) + 1
        
        # Build description
        total_objects = len(detections)
        unique_types = len(object_counts)
        
        description = f"Scene Summary: This snapshot contains {total_objects} detected object{'s' if total_objects > 1 else ''} of {unique_types} different type{'s' if unique_types > 1 else ''}.\n\n"
        
        description += "Detected Objects:\n"
        for name, count in sorted(object_counts.items()):
            description += f"  - {count} {name}{'s' if count > 1 else ''}\n"
        
        # Add context based on objects
        if 'person' in object_counts:
            description += "\nContext: The scene appears to involve human activity."
        elif any(obj in object_counts for obj in ['car', 'truck', 'bus', 'motorcycle']):
            description += "\nContext: The scene appears to be a traffic or transportation setting."
        elif any(obj in object_counts for obj in ['chair', 'couch', 'bed', 'dining table']):
            description += "\nContext: The scene appears to be an indoor living space."
        else:
            description += "\nContext: The scene contains various objects in their environment."
        
        return description
    
    def _apply_rate_limit(self) -> None:
        """Apply rate limiting (max 2 requests per second)"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_cached_description(self, snapshot_path: str) -> Optional[str]:
        """
        Retrieve cached description if available
        
        Args:
            snapshot_path: Path to snapshot
            
        Returns:
            Cached description or None
        """
        return self.cache.get(snapshot_path)
    
    def cache_description(self, snapshot_path: str, description: str) -> None:
        """
        Cache description for future retrieval
        
        Args:
            snapshot_path: Path to snapshot
            description: Description to cache
        """
        self.cache[snapshot_path] = description


if __name__ == "__main__":
    # Example usage
    from data_models import Detection
    
    generator = AIDescriptionGenerator()
    
    # Test with fallback description
    test_detections = [
        Detection(0, "person", 0.95, (100, 100, 50, 100)),
        Detection(1, "laptop", 0.87, (200, 150, 80, 60))
    ]
    
    description = generator._generate_fallback_description(test_detections)
    print("Fallback Description:")
    print(description)
