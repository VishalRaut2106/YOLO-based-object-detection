"""
Configuration Manager for Detection Performance Enhancement System

Handles loading, saving, validating, and managing system configuration.
"""

import json
import os
import logging
from typing import Dict, Any, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ConfigManager:
    """Manages system configuration with validation and persistence"""
    
    DEFAULT_CONFIG_PATH = "config.json"
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize ConfigManager with config file path
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration values
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            "detection": {
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "input_resolution": 416,
                "backend": "opencv",
                "target": "cpu"
            },
            "performance": {
                "frame_queue_size": 3,
                "enable_profiling": False,
                "log_interval_seconds": 10
            },
            "ai_description": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "timeout_seconds": 10,
                "max_retries": 2,
                "cache_enabled": True
            },
            "snapshot": {
                "directory": "snapshots",
                "jpeg_quality": 85,
                "include_bounding_boxes": True
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration values
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate detection section
        if "detection" in config:
            detection = config["detection"]
            
            # Validate confidence_threshold (0.3 - 0.9)
            if "confidence_threshold" in detection:
                threshold = detection["confidence_threshold"]
                if not isinstance(threshold, (int, float)) or threshold < 0.3 or threshold > 0.9:
                    errors.append("confidence_threshold must be between 0.3 and 0.9")
            
            # Validate nms_threshold (0.2 - 0.6)
            if "nms_threshold" in detection:
                threshold = detection["nms_threshold"]
                if not isinstance(threshold, (int, float)) or threshold < 0.2 or threshold > 0.6:
                    errors.append("nms_threshold must be between 0.2 and 0.6")
            
            # Validate input_resolution (320, 416, or 608)
            if "input_resolution" in detection:
                resolution = detection["input_resolution"]
                if resolution not in [320, 416, 608]:
                    errors.append("input_resolution must be 320, 416, or 608")
            
            # Validate backend
            if "backend" in detection:
                backend = detection["backend"]
                if backend not in ["opencv", "cuda", "openvino"]:
                    errors.append("backend must be 'opencv', 'cuda', or 'openvino'")
            
            # Validate target
            if "target" in detection:
                target = detection["target"]
                if target not in ["cpu", "cuda", "opencl"]:
                    errors.append("target must be 'cpu', 'cuda', or 'opencl'")
        
        # Validate performance section
        if "performance" in config:
            performance = config["performance"]
            
            # Validate frame_queue_size (must be positive)
            if "frame_queue_size" in performance:
                queue_size = performance["frame_queue_size"]
                if not isinstance(queue_size, int) or queue_size < 1:
                    errors.append("frame_queue_size must be a positive integer")
            
            # Validate log_interval_seconds (must be positive)
            if "log_interval_seconds" in performance:
                interval = performance["log_interval_seconds"]
                if not isinstance(interval, (int, float)) or interval <= 0:
                    errors.append("log_interval_seconds must be positive")
        
        # Validate ai_description section
        if "ai_description" in config:
            ai_desc = config["ai_description"]
            
            # Validate provider
            if "provider" in ai_desc:
                provider = ai_desc["provider"]
                if provider not in ["gemini", "llama"]:
                    errors.append("provider must be 'gemini' or 'llama'")
            
            # Validate timeout_seconds (must be positive)
            if "timeout_seconds" in ai_desc:
                timeout = ai_desc["timeout_seconds"]
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    errors.append("timeout_seconds must be positive")
            
            # Validate max_retries (must be non-negative)
            if "max_retries" in ai_desc:
                retries = ai_desc["max_retries"]
                if not isinstance(retries, int) or retries < 0:
                    errors.append("max_retries must be a non-negative integer")
        
        # Validate snapshot section
        if "snapshot" in config:
            snapshot = config["snapshot"]
            
            # Validate jpeg_quality (1-100)
            if "jpeg_quality" in snapshot:
                quality = snapshot["jpeg_quality"]
                if not isinstance(quality, int) or quality < 1 or quality > 100:
                    errors.append("jpeg_quality must be between 1 and 100")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file, create default if missing
        
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(self.config_path):
            logging.info(f"Configuration file not found at {self.config_path}, creating default")
            default_config = self.get_default_config()
            self.save_config(default_config)
            self.config = default_config
            return default_config
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate loaded configuration
            is_valid, errors = self.validate_config(config)
            
            if not is_valid:
                logging.warning(f"Configuration validation errors: {errors}")
                logging.warning("Using default values for invalid fields")
                
                # Merge with defaults for invalid/missing fields
                default_config = self.get_default_config()
                config = self._merge_with_defaults(config, default_config)
            
            self.config = config
            logging.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse configuration file: {e}")
            logging.info("Using default configuration")
            default_config = self.get_default_config()
            self.config = default_config
            return default_config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            logging.info("Using default configuration")
            default_config = self.get_default_config()
            self.config = default_config
            return default_config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Persist configuration to file
        
        Args:
            config: Configuration dictionary to save
        """
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    def _merge_with_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with defaults for missing/invalid fields
        
        Args:
            config: User configuration
            defaults: Default configuration
            
        Returns:
            Merged configuration
        """
        merged = defaults.copy()
        
        for section, values in config.items():
            if section in merged and isinstance(values, dict):
                for key, value in values.items():
                    if key in merged[section]:
                        # Validate individual field
                        temp_config = {section: {key: value}}
                        is_valid, _ = self.validate_config(temp_config)
                        if is_valid:
                            merged[section][key] = value
        
        return merged
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration (load if not already loaded)
        
        Returns:
            Current configuration dictionary
        """
        if self.config is None:
            return self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values and save
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful, False otherwise
        """
        if self.config is None:
            self.load_config()
        
        # Apply updates
        for section, values in updates.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
        
        # Validate updated configuration
        is_valid, errors = self.validate_config(self.config)
        
        if not is_valid:
            logging.error(f"Configuration update validation failed: {errors}")
            return False
        
        # Save updated configuration
        self.save_config(self.config)
        return True


if __name__ == "__main__":
    # Example usage
    manager = ConfigManager()
    config = manager.load_config()
    print("Loaded configuration:")
    print(json.dumps(config, indent=2))
