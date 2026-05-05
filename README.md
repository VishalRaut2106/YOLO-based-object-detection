# Object Detection Project

Complete object detection system with multiple implementations for different use cases.

## 📁 Project Structure

```
object-detection-opencv/
├── enhanced_yolo_system/          # Main production system
│   ├── enhanced_detection_system.py
│   ├── config_manager.py
│   ├── yolo_detector.py
│   └── ... (optimized YOLO system)
│
├── experimental_detectors/        # Alternative detectors
│   ├── gemini_camera.py          # Gemini API (unlimited objects)
│   ├── ollama_universal_detector.py  # Local AI (unlimited objects)
│   └── yolo_gemini_camera.py     # YOLO + descriptions
│
├── yolov3.cfg                     # YOLO configuration
├── yolov3.weights                 # YOLO weights (download required)
├── yolov3.txt                     # YOLO class names
│
└── Original examples:
    ├── yolo_opencv.py             # Basic YOLO on images
    └── yolo_opencv_camera.py      # Basic YOLO on camera
```

## 🚀 Quick Start

### Option 1: Enhanced YOLO System (Recommended)
**Fast, optimized, production-ready**

```bash
cd enhanced_yolo_system
pip install -r requirements.txt
python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt
```

**Features:**
- ⚡ 15+ FPS real-time detection
- 📸 Snapshot capture (press 'S')
- 🤖 AI descriptions (optional)
- 📊 Performance monitoring
- ⚙️ Configurable parameters

**Detects:** 80 common objects (person, car, laptop, phone, etc.)

---

### Option 2: Ollama Universal Detector
**Detects EVERYTHING - 100% local, no API key!**

```bash
# Setup Ollama first (one-time)
# Download from: https://ollama.com/download
ollama serve
ollama pull llama3.2-vision:11b

# Run detector
cd experimental_detectors
python ollama_universal_detector.py
```

**Features:**
- ✅ Detects glasses, headphones, pens, watches, etc.
- ✅ 100% local and private
- ✅ No API keys or internet needed
- ✅ Unlimited object types

**Trade-off:** Slower (~2-5 seconds per detection)

---

### Option 3: Gemini Universal Detector
**Detects EVERYTHING - Cloud-based**

```bash
cd experimental_detectors
python gemini_camera.py
```

**Features:**
- ✅ Detects unlimited object types
- ✅ Very accurate
- ❌ Requires API key and internet

---

## 🎯 Which One Should I Use?

### For Fast General Detection
→ **Enhanced YOLO System**
- Best for: Real-time applications, common objects
- Speed: ⚡⚡⚡ Very fast (15+ FPS)
- Objects: 80 classes

### For Detecting Glasses, Headphones, Pens, etc.
→ **Ollama Universal Detector**
- Best for: Unlimited object types, privacy
- Speed: 🐢 Slow (~0.2 FPS)
- Objects: Everything
- Bonus: 100% local, no API key

### For Maximum Accuracy
→ **Gemini Universal Detector**
- Best for: When you have API access
- Speed: 🐢 Slow (~1 FPS)
- Objects: Everything

---

## 📊 Comparison

| System | Speed | Objects | Setup | API Key | Internet |
|--------|-------|---------|-------|---------|----------|
| **Enhanced YOLO** | ⚡⚡⚡ Fast | 80 classes | Easy | Optional | No |
| **Ollama** | 🐢 Slow | Everything | Medium | No | No |
| **Gemini** | 🐢 Slow | Everything | Easy | Yes | Yes |

---

## 📚 Documentation

### Enhanced YOLO System
- `enhanced_yolo_system/README.md` - Main system docs
- `enhanced_yolo_system/QUICKSTART.md` - 5-minute setup
- `enhanced_yolo_system/IMPLEMENTATION_SUMMARY.md` - Technical details

### Experimental Detectors
- `experimental_detectors/README.md` - Alternative detectors
- `experimental_detectors/SETUP_OLLAMA.md` - Ollama setup guide

---

## 🎮 Keyboard Controls

All systems support:
- **S** - Capture snapshot
- **Q** - Quit
- **H** - Help (Enhanced YOLO only)
- **P** - Pause/Resume (Enhanced YOLO only)

---

## 📦 Installation

### Prerequisites
```bash
pip install opencv-python numpy psutil
```

### For AI Descriptions (Optional)
```bash
pip install google-generativeai pillow
```

### For Ollama (Optional)
Download from: https://ollama.com/download

---

## 🔧 YOLO Weights

Download YOLO v3 weights:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

Or download manually from: https://pjreddie.com/media/files/yolov3.weights

---

## 🎓 Original Examples

Basic YOLO examples are still available:

**Static Image Detection:**
```bash
python yolo_opencv.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```

**Basic Camera Detection:**
```bash
python yolo_opencv_camera.py -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
```

---

## 🆕 What's New

### Enhanced YOLO System
- Multi-threaded architecture for smooth performance
- Real-time FPS and performance monitoring
- Snapshot capture with AI descriptions
- Configurable detection parameters
- Memory optimization (<2GB usage)
- GPU acceleration support

### Experimental Detectors
- Ollama vision model for unlimited local detection
- Gemini API integration for cloud-based detection
- Hybrid YOLO + AI description system

---

## 🐛 Troubleshooting

### Camera not opening
```bash
# Try different camera ID
python enhanced_detection_system.py ... --camera 1
```

### Low FPS
- Reduce resolution in config.json: `"input_resolution": 320`
- Close other applications
- Check if GPU acceleration is available

### YOLO weights not found
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

---

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

- YOLO (You Only Look Once) by Joseph Redmon
- OpenCV DNN module
- Google Gemini API
- Ollama local AI
- Original implementation by Arun Ponnusamy

---

## 🚀 Getting Started

1. **For fast detection of common objects:**
   ```bash
   cd enhanced_yolo_system
   python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt
   ```

2. **For detecting everything (glasses, headphones, etc.):**
   ```bash
   cd experimental_detectors
   python ollama_universal_detector.py
   ```

Choose the system that fits your needs and start detecting! 🎯

# YOLO-based-object-detection
