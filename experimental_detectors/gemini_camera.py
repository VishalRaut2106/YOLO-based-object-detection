import cv2
import threading
import os
import time
import re
import json
from PIL import Image

import google.generativeai as genai

# --- API Key (Hardcoded) ---
# api_key = "AIzaSyCAWy42SCD5KFYE2NkpRMXSNLiCiAHI4nA"
api_key = "AIzaSyBqhaDLVDeBV2Et52WtP7IYu8lZOKF8vfg"
genai.configure(api_key=api_key)

# Use gemini-2.0-flash for best vision & speed
model = genai.GenerativeModel("gemini-2.0-flash")

PROMPT = """You are an expert object detector. Look at this image carefully and detect EVERY visible object — people, animals, furniture, electronics, food, hands, text, backgrounds, sky, walls, floors, etc. Be thorough and detect even small/partial objects.

Return ONLY a valid JSON array, no extra text. Each element has:
- "label": short descriptive name (e.g. "person", "laptop", "coffee cup", "hand", "wall")
- "box": [ymin, xmin, ymax, xmax] — normalized integers 0–1000

Example:
[
  {"label": "person", "box": [50, 100, 800, 450]},
  {"label": "laptop", "box": [300, 200, 700, 900]}
]

Now detect ALL objects in the image:"""

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

# Shared state
latest_frame = None
detected_objects = []
status_msg = "Warming up..."
lock = threading.Lock()
running = True

def analyze_worker():
    global latest_frame, detected_objects, status_msg, running
    while running:
        frame_to_analyze = None
        with lock:
            if latest_frame is not None:
                frame_to_analyze = latest_frame.copy()

        if frame_to_analyze is None:
            time.sleep(0.1)
            continue

        try:
            rgb = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            response = model.generate_content([pil_img, PROMPT])
            text = response.text.strip()

            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

            objects = json.loads(text)
            new_objects = []
            for obj in objects:
                label = obj.get("label", "object")
                box = obj.get("box", [])
                if len(box) == 4:
                    new_objects.append((label, box))

            with lock:
                detected_objects = new_objects
                status_msg = f"Detected {len(new_objects)} objects"

        except json.JSONDecodeError:
            # Fallback: try regex line-by-line parsing
            new_objects = []
            for match in re.finditer(r'"label"\s*:\s*"([^"]+)".*?"box"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text, re.DOTALL):
                label = match.group(1)
                box = [int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))]
                new_objects.append((label, box))
            with lock:
                detected_objects = new_objects
                status_msg = f"Detected {len(new_objects)} objects (fallback parser)"
        except Exception as e:
            with lock:
                status_msg = f"API Error: {str(e)[:60]}"

        time.sleep(0.8)  # ~1.2 fps to Gemini API

# Start analysis thread
thread = threading.Thread(target=analyze_worker, daemon=True)
thread.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started. Press 'q' to quit.")

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
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Gemini 2.0 Flash  |  {current_status}  |  [Q] Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow("Gemini 2.0 Flash - Universal Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
thread.join(timeout=2)
