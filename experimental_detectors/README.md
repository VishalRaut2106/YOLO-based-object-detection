# Experimental Object Detectors

Alternative detection systems that can detect **unlimited object types** including glasses, headphones, pens, watches, and more!

## 🔬 Available Detectors

### 1. Gemini Universal Detector (`gemini_camera.py`)
**Detects EVERYTHING using Google's Gemini AI**

**Pros:**
- ✅ Detects unlimited object types
- ✅ Very accurate
- ✅ Includes glasses, headphones, pens, watches, etc.

**Cons:**
- ❌ Requires API key
- ❌ Requires internet
- ❌ Slower (~1-2 FPS)
- ❌ Has quota limits

**Usage:**
```bash
python gemini_camera.py
```

---

### 2. Ollama Universal Detector (`ollama_universal_detector.py`)
**100% Local AI - No API Key, No Internet!**

**Pros:**
- ✅ Detects unlimited object types
- ✅ 100% local and private
- ✅ No API key needed
- ✅ No internet needed (after setup)
- ✅ No quota limits
- ✅ Completely free

**Cons:**
- ❌ Slower (~2-5 seconds per detection)
- ❌ Requires Ollama installation
- ❌ Large model download (~7GB)

**Setup:**
1. Install Ollama: https://ollama.com/download
2. Start Ollama: `ollama serve`
3. Download vision model: `ollama pull llama3.2-vision:11b`
4. Run: `python ollama_universal_detector.py`

See `SETUP_OLLAMA.md` for detailed instructions.

---

### 3. YOLO + Ollama Hybrid (`yolo_gemini_camera.py`)
**Fast YOLO detection + AI descriptions**

**Pros:**
- ✅ Fast detection (15+ FPS)
- ✅ AI-powered scene descriptions
- ✅ Local Ollama (no API key)
- ✅ Press 'S' for detailed descriptions

**Cons:**
- ❌ Still limited to YOLO's 80 classes for detection
- ❌ Requires Ollama for descriptions

**Usage:**
```bash
python yolo_gemini_camera.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt
```

---

## 🎯 Which One Should I Use?

### For Detecting Glasses, Headphones, Pens, etc.

**Best Choice: Ollama Universal Detector**
- 100% local and free
- No API keys or internet needed
- Detects everything

**Alternative: Gemini Universal Detector**
- If you have API key and internet
- Slightly faster than Ollama

### For Fast General Detection

**Use: Enhanced YOLO System** (in `enhanced_yolo_system` folder)
- Very fast (15+ FPS)
- Good for common objects
- Limited to 80 classes

---

## 📊 Comparison Table

| Detector | Speed | Objects | API Key | Internet | Glasses/Pens |
|----------|-------|---------|---------|----------|--------------|
| Enhanced YOLO | ⚡⚡⚡ Fast | 80 classes | Optional | No | ❌ No |
| Gemini | 🐢 Slow | Everything | Required | Yes | ✅ Yes |
| Ollama | 🐢 Slow | Everything | No | No | ✅ Yes |
| YOLO+Ollama | ⚡⚡⚡ Fast | 80 classes | No | No | ❌ No* |

*Can describe them in text but not detect with bounding boxes

---

## 🧪 Testing Tools

- `test_gemini_api.py` - Test if Gemini API key works
- `list_gemini_models.py` - List available Gemini models

---

## 📚 Documentation

- `SETUP_OLLAMA.md` - Complete Ollama setup guide

---

## 💡 Recommendations

**For Production Use:**
→ Use **Enhanced YOLO System** (fast, reliable, no dependencies)

**For Research/Experimentation:**
→ Use **Ollama Universal Detector** (detects everything, completely local)

**For Maximum Accuracy:**
→ Use **Gemini Universal Detector** (if you have API access)

---

## 🔄 Switching Between Detectors

All detectors are standalone - just run the one you want:

```bash
# Fast YOLO (80 classes)
cd ../enhanced_yolo_system
python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt

# Universal Gemini (everything, needs API)
cd ../experimental_detectors
python gemini_camera.py

# Universal Ollama (everything, 100% local)
cd ../experimental_detectors
python ollama_universal_detector.py
```

---

## 🚀 Future Improvements

- [ ] Hybrid detector combining YOLO speed + Ollama coverage
- [ ] Real-time Ollama optimization
- [ ] Custom object training
- [ ] Multi-camera support
