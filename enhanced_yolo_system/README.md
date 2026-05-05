# Enhanced YOLO Detection System

High-performance real-time object detection with optimized YOLO inference, AI-powered descriptions, and comprehensive performance monitoring.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download YOLO Weights
```bash
# Go back to parent directory where YOLO files are
cd ..
```

Make sure you have:
- `yolov3.cfg`
- `yolov3.weights`
- `yolov3.txt`

### 3. Run the System
```bash
python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt
```

## ✨ Features

- ⚡ **Fast Detection**: 15+ FPS on CPU (4+ cores)
- 📸 **Snapshot Capture**: Press 'S' to save frames
- 🤖 **AI Descriptions**: Automatic scene descriptions (with API key)
- 📊 **Performance Monitoring**: Real-time FPS, latency, memory tracking
- ⚙️ **Configurable**: Adjust thresholds, resolution, backend
- 🧵 **Multi-threaded**: Smooth, non-blocking architecture
- 💾 **Memory Efficient**: <2GB usage

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| **S** | Capture snapshot + generate AI description |
| **D** | Toggle description panel |
| **H** | Show help overlay |
| **P** | Pause/Resume detection |
| **Q** | Quit application |

## 📋 What YOLO Can Detect (80 Classes)

YOLO v3 detects these objects:
- People, vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, etc.)
- Furniture (chair, couch, bed, table)
- Electronics (laptop, mouse, keyboard, cell phone, TV)
- Kitchen items (bottle, cup, fork, knife, bowl)
- And 60+ more common objects

**Note**: YOLO cannot detect glasses, headphones, pens, watches, etc. For unlimited object detection, see the `experimental_detectors` folder.

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "detection": {
    "confidence_threshold": 0.5,
    "nms_threshold": 0.4,
    "input_resolution": 416
  },
  "performance": {
    "frame_queue_size": 3,
    "enable_profiling": false
  },
  "snapshot": {
    "directory": "snapshots",
    "jpeg_quality": 85
  }
}
```

## 🔧 Performance Tuning

### For Faster Processing (CPU)
```json
{
  "detection": {
    "input_resolution": 320
  }
}
```

### For Better Accuracy
```json
{
  "detection": {
    "input_resolution": 608,
    "confidence_threshold": 0.6
  }
}
```

## 📁 Project Structure

```
enhanced_yolo_system/
├── enhanced_detection_system.py  # Main application
├── config_manager.py             # Configuration management
├── data_models.py                # Data structures
├── yolo_detector.py              # YOLO detection engine
├── frame_pipeline.py             # Multi-threaded pipeline
├── snapshot_manager.py           # Snapshot handling
├── performance_monitor.py        # Performance tracking
├── ai_description_generator.py   # AI descriptions
├── config.json                   # Configuration file
├── requirements.txt              # Dependencies
└── snapshots/                    # Captured snapshots
```

## 🧪 Testing

Run the test suite:
```bash
python test_system.py
```

Expected output: `✓ All tests passed!`

## 📚 Documentation

- **QUICKSTART.md** - 5-minute setup guide
- **IMPLEMENTATION_SUMMARY.md** - What was built
- **SETUP_AI_DESCRIPTIONS.md** - Enable AI descriptions
- **config_examples.json** - Example configurations

## 🆚 Comparison with Other Detectors

| Feature | Enhanced YOLO | Gemini API | Ollama Vision |
|---------|---------------|------------|---------------|
| Speed | ⚡ Fast (15+ FPS) | 🐢 Slow (~1 FPS) | 🐢 Slow (~0.2 FPS) |
| Objects | 80 classes | Everything | Everything |
| API Key | Optional | Required | Not needed |
| Internet | No | Yes | No |
| Glasses/Headphones | ❌ No | ✅ Yes | ✅ Yes |

## 🔮 For Universal Detection

If you need to detect objects not in YOLO's 80 classes (glasses, headphones, pens, etc.), check out the **experimental_detectors** folder for:
- Gemini-based universal detector
- Ollama vision-based local detector

## 📝 License

See LICENSE file in parent directory.

## 🙏 Acknowledgments

- YOLO (You Only Look Once) by Joseph Redmon
- OpenCV DNN module
- Original implementation by Arun Ponnusamy
