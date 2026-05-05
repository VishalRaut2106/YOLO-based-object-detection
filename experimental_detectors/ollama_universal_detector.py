"""
Universal Object Detection using Ollama Vision Model (Local, No API Key!)

Uses llama3.2-vision model running locally via Ollama to detect ALL objects.
"""

import cv2
import threading
import time
import json
import base64
import urllib.request
import re
import numpy as np

# ─────────────────────────────────────────────
# OLLAMA SETUP (100% Local - No API Key!)
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"  # Vision model for object detection

PROMPT = """You are an expert object detector. Look at this image carefully and detect EVERY visible object — people, animals, furniture, electronics, food, glasses, headphones, pens, watches, hands, text, backgrounds, walls, floors, etc. Be thorough and detect even small objects.

Return ONLY a valid JSON array, no extra text or explanation. Each element must have:
- "label": short descriptive name (e.g. "person", "laptop", "eyeglasses", "headphones", "pen")
- "box": [ymin, xmin, ymax, xmax] — normalized integers 0–1000

Example format:
[
  {"label": "person", "box": [50, 100, 800, 450]},
  {"label": "eyeglasses", "box": [200, 250, 300, 400]},
  {"label": "headphones", "box": [100, 150, 250, 500]}
]

Detect ALL objects now:"""

# Distinct colors for labels
PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

def get_color(label):
    idx = hash(label) % len(PALETTE)
    return PALETTE[idx]

def draw_box(frame, label, box, height, width):
    ymin, xmin, ymax, xmax = box
    x1 = int(xmin * width / 1000)
    y1 = int(ymin * height / 1000)
    x2 = int(xmax * width / 1000)
    y2 = int(ymax * height / 1000)
    color = get_color(label)

    # Draw filled background rect for label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def encode_frame_to_base64(frame):
    """Encode frame to base64 for Ollama vision API"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def ask_ollama_vision(frame):
    """Send frame to Ollama vision model and get detected objects"""
    # Encode frame to base64
    image_base64 = encode_frame_to_base64(frame)
    
    # Prepare request
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": PROMPT,
        "images": [image_base64],
        "stream": False
    }).encode("utf-8")
    
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "").strip()

# Shared state
latest_frame = None
detected_objects = []
status_msg = "Starting Ollama vision model..."
lock = threading.Lock()
running = True

def analyze_worker():
    global latest_frame, detected_objects, status_msg, running
    
    print("Ollama vision detection thread started")
    print(f"Using model: {OLLAMA_MODEL}")
    print("Make sure Ollama is running: ollama serve")
    
    while running:
        frame_to_analyze = None
        with lock:
            if latest_frame is not None:
                frame_to_analyze = latest_frame.copy()

        if frame_to_analyze is None:
            time.sleep(0.1)
            continue

        try:
            print("Sending frame to Ollama vision model...")
            start_time = time.time()
            
            response_text = ask_ollama_vision(frame_to_analyze)
            
            inference_time = time.time() - start_time
            print(f"Ollama response received in {inference_time:.1f}s")

            # Strip markdown code fences if present
            response_text = re.sub(r"^```(?:json)?", "", response_text).strip()
            response_text = re.sub(r"```$", "", response_text).strip()

            # Try to parse JSON
            try:
                objects = json.loads(response_text)
                new_objects = []
                for obj in objects:
                    label = obj.get("label", "object")
                    box = obj.get("box", [])
                    if len(box) == 4:
                        new_objects.append((label, box))

                with lock:
                    detected_objects = new_objects
                    status_msg = f"Detected {len(new_objects)} objects ({inference_time:.1f}s)"
                
                print(f"Detected: {[obj[0] for obj in new_objects]}")

            except json.JSONDecodeError:
                # Fallback: try regex line-by-line parsing
                print("JSON parse failed, trying regex fallback...")
                new_objects = []
                for match in re.finditer(r'"label"\s*:\s*"([^"]+)".*?"box"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text, re.DOTALL):
                    label = match.group(1)
                    box = [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))]
                    new_objects.append((label, box))
                
                with lock:
                    detected_objects = new_objects
                    status_msg = f"Detected {len(new_objects)} objects (fallback)"

        except urllib.error.URLError as e:
            with lock:
                status_msg = "Error: Ollama not running? Start with: ollama serve"
            print(f"Connection error: {e}")
            time.sleep(5)
        except Exception as e:
            with lock:
                status_msg = f"Error: {str(e)[:50]}"
            print(f"Error: {e}")
            time.sleep(2)

        time.sleep(1)  # Wait before next detection

# Start analysis thread
thread = threading.Thread(target=analyze_worker, daemon=True)
thread.start()

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "="*70)
print("  Ollama Universal Object Detector - 100% Local, No API Key!")
print("="*70)
print(f"  Model: {OLLAMA_MODEL}")
print("  Detects: Glasses, Headphones, Pens, Watches, and EVERYTHING!")
print("  [Q] Quit")
print("="*70 + "\n")

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

print("Camera started. Waiting for detections...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()
        current_objects = detected_objects.copy()
        current_status = status_msg

    h, w = frame.shape[:2]

    # Draw all detected objects
    for label, box in current_objects:
        draw_box(frame, label, box, h, w)

    # Draw status bar at the bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"Ollama {OLLAMA_MODEL}  |  {current_status}  |  [Q] Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Ollama Universal Object Detector - 100% Local!", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
thread.join(timeout=2)

print("\nExited.")
