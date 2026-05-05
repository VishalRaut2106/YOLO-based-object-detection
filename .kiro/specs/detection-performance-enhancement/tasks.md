# Implementation Plan: Detection Performance Enhancement

## Overview

This implementation plan breaks down the detection performance enhancement feature into discrete coding tasks. The implementation will enhance the existing YOLO-based object detection system with performance optimizations, asynchronous snapshot capture with AI descriptions, and robust threading architecture. The system will maintain compatibility with existing implementations while adding configurable detection parameters, GPU acceleration support, and comprehensive error handling.

## Tasks

- [x] 1. Set up project structure and configuration management
  - Create `config.json` with default settings for detection, performance, AI description, and snapshot parameters
  - Implement `ConfigManager` class with load, save, validate, and get_default_config methods
  - Add configuration schema validation for threshold ranges and required fields
  - _Requirements: 6.1, 6.3, 6.5, 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ]* 1.1 Write property test for configuration validation
  - **Property 8: Confidence Threshold Validation**
  - **Validates: Requirements 6.1**
  - Test that confidence thresholds in range [0.3, 0.9] are accepted and values outside are rejected

- [ ]* 1.2 Write property test for configuration persistence
  - **Property 12: Configuration Persistence Round-Trip**
  - **Validates: Requirements 6.5**
  - Test that saving and loading configuration preserves all values

- [ ]* 1.3 Write property test for NMS threshold validation
  - **Property 10: NMS Threshold Validation**
  - **Validates: Requirements 6.3**
  - Test that NMS thresholds in range [0.2, 0.6] are accepted and values outside are rejected

- [ ]* 1.4 Write property test for invalid configuration handling
  - **Property 18: Invalid Configuration Handling**
  - **Validates: Requirements 15.4**
  - Test that invalid configurations fail validation and system uses default values

- [ ]* 1.5 Write unit tests for ConfigManager
  - Test valid configuration passes validation
  - Test invalid threshold values are rejected
  - Test missing configuration file creates default
  - Test configuration file persistence
  - _Requirements: 6.1, 6.3, 6.5, 15.1, 15.2, 15.4_

- [x] 2. Implement core detection data models and interfaces
  - Create `Detection` dataclass with class_id, class_name, confidence, bbox, and color fields
  - Create `FrameData` dataclass with frame, detections, timestamp, and frame_id fields
  - Create `SnapshotMetadata` dataclass with file_path, timestamp, detections, description, and description_status fields
  - Create `PerformanceMetrics` dataclass with inference time, latency, FPS, frame drop rate, and memory usage fields
  - Create `DetectionConfig` dataclass with confidence_threshold, nms_threshold, input_resolution, backend, and target fields
  - _Requirements: 2.1, 3.3, 4.1, 10.1, 10.2, 10.3_

- [ ]* 2.1 Write property test for bounding box preservation
  - **Property 2: Bounding Box Preservation Through Save/Load**
  - **Validates: Requirements 3.3**
  - Test that saving snapshot with detections and loading preserves all bbox coordinates and labels

- [ ]* 2.2 Write unit tests for data models
  - Test Detection dataclass initialization and field access
  - Test FrameData with valid and empty detection lists
  - Test SnapshotMetadata status transitions
  - Test PerformanceMetrics calculations
  - _Requirements: 2.1, 3.3, 4.1_

- [x] 3. Implement YOLO detection engine with optimization
  - Create `YOLODetector` class with initialization, detect, and configuration methods
  - Implement GPU acceleration detection and automatic enablement when CUDA is available
  - Add support for configurable input resolutions (320, 416, 608)
  - Implement confidence threshold filtering for detections
  - Implement NMS (Non-Maximum Suppression) with configurable threshold
  - Reuse neural network model instance across frames
  - _Requirements: 1.1, 1.2, 1.4, 6.1, 6.2, 6.3, 6.4, 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ]* 3.1 Write property test for detection filtering by confidence
  - **Property 9: Detection Filtering by Confidence Threshold**
  - **Validates: Requirements 6.2**
  - Test that all returned detections have confidence >= threshold

- [ ]* 3.2 Write property test for NMS overlap elimination
  - **Property 11: NMS Eliminates Overlapping Detections**
  - **Validates: Requirements 6.4**
  - Test that NMS eliminates detections with IoU above threshold

- [ ]* 3.3 Write unit tests for YOLODetector
  - Test model initialization with valid config
  - Test GPU acceleration detection
  - Test input resolution changes
  - Test confidence threshold filtering
  - Test NMS threshold application
  - Test model reuse across multiple frames
  - _Requirements: 1.1, 1.4, 6.2, 6.4, 9.1, 9.5_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement frame processing pipeline with threading
  - Create `FramePipeline` class with start, stop, get_latest_frame, pause, and resume methods
  - Implement separate thread for frame capture from webcam
  - Implement separate thread for YOLO detection processing
  - Use `queue.Queue(maxsize=3)` for thread-safe frame buffering
  - Implement frame drop logic when queue is full (drop oldest frame)
  - Use shared memory for frame data between threads (avoid copying)
  - Implement graceful thread shutdown with 2-second timeout
  - _Requirements: 1.3, 1.5, 2.2, 2.4, 3.5, 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 5.1 Write property test for frame queue overflow behavior
  - **Property 14: Frame Queue Overflow Behavior**
  - **Validates: Requirements 8.5**
  - Test that adding frame to full queue drops oldest and maintains size of 3

- [ ]* 5.2 Write unit tests for FramePipeline
  - Test pipeline start and stop
  - Test frame capture and queuing
  - Test detection thread processing
  - Test pause and resume functionality
  - Test graceful shutdown within timeout
  - Test frame drop when queue is full
  - _Requirements: 1.3, 3.5, 8.1, 8.3, 8.4, 8.5_

- [x] 6. Implement performance monitoring
  - Create `PerformanceMonitor` class with metric recording and reporting methods
  - Track average inference time per frame
  - Track average end-to-end latency from capture to display
  - Track current FPS and frame drop rate
  - Track memory usage using psutil
  - Implement periodic logging every 10 seconds when profiling is enabled
  - Add command-line flag for enabling detailed performance profiling
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ]* 6.1 Write unit tests for PerformanceMonitor
  - Test inference time recording and averaging
  - Test frame drop counting
  - Test end-to-end latency tracking
  - Test FPS calculation
  - Test memory usage tracking
  - Test periodic logging
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 7. Implement snapshot manager with asynchronous saving
  - Create `SnapshotManager` class with capture, queue, and batch processing methods
  - Implement snapshot capture with timestamp-based filename (snap_YYYYMMDD_HHMMSS.jpg)
  - Save snapshots with JPEG compression (85% quality) to reduce file size
  - Implement asynchronous snapshot saving in separate thread to avoid blocking
  - Create snapshots directory if it doesn't exist
  - Preserve bounding boxes and labels in saved snapshots
  - Implement queue for snapshots awaiting AI description
  - Implement batch processing for existing snapshots in directory
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 12.3, 13.1, 13.4_

- [ ]* 7.1 Write property test for detection history buffer size
  - **Property 1: Detection History Buffer Size Constraint**
  - **Validates: Requirements 2.3**
  - Test that buffer never exceeds 10 frames regardless of how many are added

- [ ]* 7.2 Write unit tests for SnapshotManager
  - Test snapshot capture with timestamp filename
  - Test snapshot directory creation
  - Test JPEG compression quality
  - Test asynchronous saving doesn't block
  - Test snapshot queuing for description
  - Test batch processing of existing snapshots
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 13.1, 13.4_

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement AI description generator with retry logic
  - Create `AIDescriptionGenerator` class supporting Gemini and Llama providers
  - Implement synchronous description generation with timeout
  - Implement asynchronous description generation with callback
  - Format descriptions with scene summary, object list, and contextual interpretation
  - Include accurate object counts for each detected class
  - Implement retry logic with exponential backoff (1s, 2s for general failures)
  - Implement rate limiting for Gemini API (max 2 requests per second)
  - Implement description caching to avoid redundant API calls
  - Handle API timeouts (5s for Gemini, 10s for Llama)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.4, 5.5, 5.6, 12.1, 12.2, 12.4, 12.5_

- [ ]* 9.1 Write property test for description includes object counts
  - **Property 3: Description Includes Object Counts**
  - **Validates: Requirements 4.2**
  - Test that generated description includes count for each detected class

- [ ]* 9.2 Write property test for accurate duplicate counting
  - **Property 4: Accurate Duplicate Object Counting**
  - **Validates: Requirements 4.3**
  - Test that reported count equals actual number of occurrences for each class

- [ ]* 9.3 Write property test for description format structure
  - **Property 5: Description Format Structure**
  - **Validates: Requirements 4.6**
  - Test that description contains scene summary, object list, and context sections

- [ ]* 9.4 Write property test for API retry and exponential backoff
  - **Property 6: API Retry and Exponential Backoff**
  - **Validates: Requirements 5.4, 12.4**
  - Test retry count and backoff timing for API failures

- [ ]* 9.5 Write property test for description caching
  - **Property 7: Description Caching Prevents Redundant API Calls**
  - **Validates: Requirements 5.6**
  - Test that multiple requests for same snapshot result in only one API call

- [ ]* 9.6 Write property test for API rate limiting
  - **Property 15: API Rate Limiting**
  - **Validates: Requirements 12.1**
  - Test that API request rate never exceeds 2 per second

- [ ]* 9.7 Write unit tests for AIDescriptionGenerator
  - Test Gemini provider initialization
  - Test Llama provider initialization
  - Test synchronous description generation
  - Test asynchronous description with callback
  - Test description format validation
  - Test retry logic on API failure
  - Test rate limiting enforcement
  - Test description caching
  - Test timeout handling
  - _Requirements: 4.1, 4.4, 4.5, 4.6, 5.1, 5.4, 5.5, 5.6, 12.1, 12.2, 12.4_

- [ ] 10. Implement description thread for asynchronous processing
  - Create separate thread for AI description generation
  - Implement queue for snapshot description requests
  - Process description requests asynchronously without blocking main loop
  - Update snapshot metadata with description status (pending, analyzing, done, error)
  - Display status indicator while generating description
  - Handle API errors gracefully and display user-friendly error messages
  - _Requirements: 5.1, 5.2, 5.3, 5.5, 8.1_

- [ ]* 10.1 Write property test for batch processing completeness
  - **Property 16: Batch Processing Completeness**
  - **Validates: Requirements 13.2**
  - Test that all snapshots in batch receive descriptions upon completion

- [ ]* 10.2 Write property test for skip cached snapshots
  - **Property 17: Skip Cached Snapshots in Batch Processing**
  - **Validates: Requirements 13.5**
  - Test that batch processor skips snapshots with cached descriptions

- [ ]* 10.3 Write unit tests for description thread
  - Test thread starts and stops gracefully
  - Test queue processing
  - Test status updates during processing
  - Test error handling and user messages
  - Test asynchronous processing doesn't block main loop
  - _Requirements: 5.1, 5.2, 5.3, 5.5, 8.1_

- [ ] 11. Implement UI renderer with visual feedback
  - Create `UIRenderer` class with drawing methods for detections, status, descriptions, and overlays
  - Draw bounding boxes with labels and confidence scores
  - Use distinct colors for different object classes (seeded random for reproducibility)
  - Draw status bar with current FPS and object count
  - Draw AI description overlay panel with semi-transparent background
  - Implement snapshot flash effect for visual confirmation
  - Implement help overlay showing keyboard shortcuts
  - _Requirements: 3.4, 4.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ]* 11.1 Write property test for distinct class colors
  - **Property 13: Distinct Colors for Different Classes**
  - **Validates: Requirements 7.6**
  - Test that different object classes receive different colors

- [ ]* 11.2 Write unit tests for UIRenderer
  - Test bounding box drawing
  - Test label and confidence display
  - Test color assignment for classes
  - Test status bar rendering
  - Test description panel overlay
  - Test snapshot flash effect
  - Test help overlay display
  - _Requirements: 3.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement keyboard input handling
  - Implement non-blocking keyboard input using OpenCV's waitKey
  - Handle 's' key for snapshot capture and AI description trigger
  - Handle 'h' key to toggle AI description panel visibility
  - Handle 'q' key for graceful exit and resource cleanup
  - Handle 'p' key to pause/resume live detection
  - Handle 'F1' key to toggle help overlay
  - _Requirements: 3.1, 14.1, 14.2, 14.3, 14.4, 14.5_

- [ ]* 13.1 Write unit tests for keyboard input handling
  - Test 's' key triggers snapshot capture
  - Test 'h' key toggles description panel
  - Test 'q' key initiates graceful exit
  - Test 'p' key pauses and resumes detection
  - Test 'F1' key toggles help overlay
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

- [ ] 14. Implement error handling and robustness
  - Create `ErrorHandler` class with static methods for different error types
  - Handle webcam initialization failure with descriptive error and troubleshooting steps
  - Handle YOLO model loading failure with descriptive error and troubleshooting steps
  - Handle frame read failures gracefully (log and continue)
  - Handle AI API errors with retry logic and user-friendly messages
  - Log all exceptions with full stack traces for debugging
  - Continue detection when AI API is unavailable
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [ ]* 14.1 Write unit tests for ErrorHandler
  - Test webcam error handling and exit
  - Test model load error handling and exit
  - Test frame read error logging and recovery
  - Test API error retry logic
  - Test exception logging
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 15. Integrate all components into main application
  - Create main application entry point with command-line argument parsing
  - Initialize ConfigManager and load configuration
  - Initialize YOLODetector with YOLO model files and configuration
  - Initialize FramePipeline with detector and configuration
  - Initialize SnapshotManager with snapshot directory and configuration
  - Initialize AIDescriptionGenerator with provider and configuration
  - Initialize PerformanceMonitor with profiling flag
  - Initialize UIRenderer with configuration
  - Start frame pipeline and description thread
  - Implement main loop for frame display and keyboard input
  - Implement graceful shutdown on exit
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 10.1, 11.1, 14.1, 15.1, 15.5_

- [ ]* 15.1 Write integration tests for main application
  - Test complete frame processing pipeline
  - Test snapshot capture and description workflow
  - Test configuration loading and validation
  - Test graceful shutdown
  - Test error recovery scenarios
  - _Requirements: 1.1, 3.1, 5.1, 8.3, 11.1, 15.1_

- [ ] 16. Add command-line interface and batch processing
  - Add command-line arguments for config file path, YOLO model files, and profiling flag
  - Implement batch processing mode for existing snapshots
  - Add command-line option to process snapshots directory
  - Generate summary report after batch processing
  - Skip snapshots with cached descriptions during batch processing
  - _Requirements: 10.5, 13.1, 13.2, 13.3, 13.4, 13.5, 15.5_

- [ ]* 16.1 Write unit tests for batch processing
  - Test batch processing of multiple snapshots
  - Test summary report generation
  - Test skipping cached snapshots
  - Test command-line argument parsing
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ] 17. Optimize memory usage and implement memory management
  - Implement frame buffer reuse (pre-allocate and reuse buffers)
  - Use shared memory for frame data between threads
  - Limit detection history buffer to 10 frames
  - Release frame buffers immediately after processing
  - Pre-allocate fixed-size arrays for bounding box data
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 17.1 Write performance tests for memory usage
  - Test memory usage stays below 2GB during operation
  - Test frame buffer reuse reduces allocations
  - Test detection history buffer limit
  - Test memory growth over extended runtime
  - _Requirements: 2.1, 2.2, 2.3_

- [ ] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Create documentation and usage examples
  - Create README with installation instructions
  - Document system requirements (minimum and recommended)
  - Document command-line usage and options
  - Document configuration file format and options
  - Document keyboard shortcuts
  - Create example configuration files for different environments (CPU-only, GPU, low-memory)
  - Document troubleshooting steps for common issues
  - _Requirements: All requirements_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout implementation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and error conditions
- Integration tests validate end-to-end workflows and component interactions
- The implementation uses Python with OpenCV, NumPy, and Google Generative AI libraries
- GPU acceleration is automatically enabled when CUDA is available
- Threading architecture separates I/O-bound operations (capture, API calls) from CPU-bound operations (YOLO inference)
- Configuration is persisted across sessions for user convenience
- Error handling ensures graceful degradation when components fail
