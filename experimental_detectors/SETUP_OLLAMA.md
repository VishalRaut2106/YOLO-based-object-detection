# Setup Ollama for Local Object Detection

Use Ollama to run AI models **100% locally** - no API keys, no internet, no quota limits!

## Step 1: Install Ollama

### Windows:
Download and install from: https://ollama.com/download/windows

### Linux/Mac:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 2: Start Ollama Server

Open a **new terminal** and run:
```bash
ollama serve
```

Keep this terminal open while using the detector.

## Step 3: Download the Vision Model

In another terminal, run:
```bash
ollama pull llama3.2-vision:11b
```

This downloads the vision model (~7GB). Wait for it to complete.

## Step 4: Run the Universal Detector

```bash
python ollama_universal_detector.py
```

## What You Get

✅ **100% Local** - No internet needed after setup
✅ **No API Keys** - Completely free
✅ **No Quota Limits** - Use as much as you want
✅ **Detects Everything** - Glasses, headphones, pens, watches, etc.
✅ **Privacy** - Your images never leave your computer

## Detected Objects Include:

- 👓 Eyeglasses
- 🎧 Headphones
- 🖊️ Pens/Pencils
- ⌚ Watches
- 💳 Wallets
- 📱 Phones
- 💻 Laptops
- 🖱️ Mouse/Keyboard
- 👤 People
- 🐕 Animals
- 🪑 Furniture
- **And literally EVERYTHING else!**

## Performance

- **Speed**: ~2-5 seconds per detection (slower than YOLO but detects everything)
- **Accuracy**: Very high
- **Coverage**: Unlimited object types

## Troubleshooting

### "Connection refused" error
Make sure Ollama is running:
```bash
ollama serve
```

### Model not found
Download the vision model:
```bash
ollama pull llama3.2-vision:11b
```

### Slow detection
This is normal - vision models are slower but much more capable than YOLO.

### Out of memory
Try the smaller 3B model:
```bash
ollama pull llama3.2-vision:3b
```

Then edit `ollama_universal_detector.py` and change:
```python
OLLAMA_MODEL = "llama3.2-vision:3b"
```

## Alternative: Use for Descriptions Only

If you want fast YOLO detection + Ollama descriptions, run:
```bash
python yolo_gemini_camera.py -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
```

This uses:
- YOLO for fast detection (15+ FPS)
- Ollama for AI descriptions when you press 'S'

## Comparison

| Feature | YOLO | Gemini API | Ollama Vision |
|---------|------|------------|---------------|
| Speed | ⚡ Fast (15+ FPS) | 🐢 Slow (~1 FPS) | 🐢 Slow (~0.2 FPS) |
| Objects | 80 classes only | Everything | Everything |
| API Key | ❌ No | ✅ Yes | ❌ No |
| Internet | ❌ No | ✅ Yes | ❌ No |
| Cost | Free | Free tier | Free |
| Glasses/Headphones | ❌ No | ✅ Yes | ✅ Yes |

## Recommendation

**For detecting glasses, headphones, pens, etc.:**
Use **Ollama Vision** - it's completely local and free!

**For fast general detection:**
Use **YOLO** - it's very fast but limited to 80 classes.

**For best of both:**
Use **YOLO + Ollama** hybrid (yolo_gemini_camera.py)
