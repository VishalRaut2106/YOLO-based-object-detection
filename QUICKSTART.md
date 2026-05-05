# Quick Start Guide

Get the Enhanced Object Detection System running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download YOLO Weights

```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

Or download manually from: https://pjreddie.com/media/files/yolov3.weights

## Step 3: Verify Setup

```bash
python test_system.py
```

You should see: `✓ All tests passed! System is ready to use.`

## Step 4: Run the System

```bash
python enhanced_detection_system.py -c yolov3.cfg -w yolov3.weights -cl yolov3.txt
```

## Step 5: Use the System

Once running, you'll see a window with real-time object detection:

- **Press 'S'** to capture a snapshot
- **Press 'H'** to see help
- **Press 'P'** to pause/resume
- **Press 'Q'** to quit

## What You'll See

- Real-time bounding boxes around detected objects
- FPS counter and performance metrics
- Object counts and confidence scores
- Status bar with keyboard shortcuts

## Troubleshooting

### Camera not found?
```bash
# Try a different camera ID
python enhanced_detection_system.py ... --camera 1
```

### Low FPS?
Edit `config.json` and change:
```json
{
  "detection": {
    "input_resolution": 320
  }
}
```

### Missing YOLO files?
Make sure you have:
- `yolov3.cfg` (included in repo)
- `yolov3.weights` (download from link above)
- `yolov3.txt` (included in repo)

## Next Steps

- Check `README.md` for detailed documentation
- Customize `config.json` for your needs
- Explore captured snapshots in the `snapshots/` directory

## Performance Tips

**For CPU-only systems:**
- Use resolution 320 for faster processing
- Increase confidence threshold to 0.6

**For GPU systems:**
- System will auto-detect and use CUDA
- Can use resolution 416 or 608 for better accuracy

**For low-memory systems:**
- Set `frame_queue_size` to 1 in config
- Use resolution 320

## Need Help?

Check the full README.md for:
- Detailed configuration options
- Architecture overview
- Advanced troubleshooting
- Performance optimization tips
