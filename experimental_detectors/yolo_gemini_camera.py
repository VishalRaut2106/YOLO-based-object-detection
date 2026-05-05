import cv2
import argparse
import numpy as np
import threading
import os
import time
import textwrap
import urllib.request
import json

# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',   required=True, help='Path to YOLOv3 config file')
ap.add_argument('-w', '--weights',  required=True, help='Path to YOLOv3 weights file')
ap.add_argument('-cl', '--classes', required=True, help='Path to class names file')
args = ap.parse_args()

# ─────────────────────────────────────────────
# OLLAMA SETUP (local model - no API key needed)
# ─────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

def ask_ollama(prompt_text):
    """Send a text prompt to local Ollama and return the response string."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "").strip()

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# YOLO SETUP
# ─────────────────────────────────────────────
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def run_yolo(frame):
    """Run YOLO on a frame, return list of (class_id, confidence, box)."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence > 0.45:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = cx - bw // 2
                y = cy - bh // 2
                class_ids.append(class_id)
                confidences.append(confidence)
                boxes.append([x, y, bw, bh])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((class_ids[i], confidences[i], boxes[i]))
    return results

def draw_yolo(frame, detections):
    """Draw YOLO bounding boxes on frame."""
    for class_id, conf, box in detections:
        x, y, w, h = box
        color = COLORS[class_id].tolist()
        label = f"{classes[class_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame

# ─────────────────────────────────────────────
# AI DESCRIPTION THREAD STATE
# ─────────────────────────────────────────────
ai_lock            = threading.Lock()
ai_status          = ""    # "analyzing" / "done" / "error"
ai_description     = []    # list of wrapped lines to display
ai_snapshot_path   = ""

def describe_with_llama(detections_snapshot, image_path):
    """
    Build a text prompt from YOLO detections and ask llama3.2:1b
    to generate a natural scene description.
    """
    global ai_status, ai_description

    # Build object list string
    if detections_snapshot:
        counts = {}
        for label, conf in detections_snapshot:
            counts[label] = counts.get(label, 0) + 1
        obj_list = ", ".join(
            f"{cnt} {lbl}{'s' if cnt > 1 else ''}" for lbl, cnt in counts.items()
        )
    else:
        obj_list = "nothing clearly identifiable"

    prompt = (
        f"A camera just captured a scene. "
        f"An object detection system identified these objects: {obj_list}.\n\n"
        f"Based on these detected objects, please provide:\n"
        f"1. A natural, friendly 1-2 sentence description of what this scene likely shows.\n"
        f"2. A short bullet list of the key objects.\n"
        f"3. One sentence about the likely setting or activity.\n\n"
        f"Keep it concise and conversational."
    )

    try:
        print(f"[Ollama] Asking llama3.2:1b about: {obj_list}")
        response_text = ask_ollama(prompt)

        lines = []
        for para in response_text.split('\n'):
            para = para.strip()
            if not para:
                lines.append("")
                continue
            for line in textwrap.wrap(para, width=58):
                lines.append(line)

        with ai_lock:
            ai_description = lines
            ai_status = "done"
        print("[Ollama] Description ready.")

    except Exception as e:
        err = str(e)
        print(f"[Ollama] Error: {err}")
        with ai_lock:
            ai_description = [f"Ollama error: {err[:70]}",
                               "Make sure Ollama is running: ollama serve"]
            ai_status = "error"

def trigger_analysis(frame, detections):
    """Save snapshot and launch Ollama description in background thread."""
    global ai_status, ai_snapshot_path

    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snap_{ts}.jpg")
    cv2.imwrite(path, frame)

    # Extract just labels + confidences for the prompt
    det_snapshot = [(classes[d[0]], d[1]) for d in detections]

    with ai_lock:
        ai_status        = "analyzing"
        ai_description   = ["Analyzing with llama3.2:1b... please wait"]
        ai_snapshot_path = path

    t = threading.Thread(
        target=describe_with_llama,
        args=(det_snapshot, path),
        daemon=True
    )
    t.start()
    print(f"[Snapshot] Saved -> {path}")

# ─────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────
def draw_status_bar(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

def draw_ai_panel(frame, lines, status):
    """Draw semi-transparent AI description panel on right side."""
    if not lines:
        return
    h, w = frame.shape[:2]

    panel_w = min(430, w - 20)
    line_h  = 20
    padding = 10
    panel_h = min(len(lines) * line_h + padding * 2 + 28, h - 20)

    px = w - panel_w - 10
    py = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Title bar color by status
    if status == "done":
        title_color = (30, 180, 80)
        title_text  = "Llama3.2 Scene Analysis"
    elif status == "analyzing":
        title_color = (200, 140, 0)
        title_text  = "Llama3.2 Analyzing..."
    else:
        title_color = (0, 50, 200)
        title_text  = "Llama3.2 Error"

    cv2.rectangle(frame, (px, py), (px + panel_w, py + 24), title_color, -1)
    cv2.putText(frame, title_text, (px + 6, py + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Border
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), title_color, 1)

    # Description lines
    max_lines = (panel_h - 28 - padding) // line_h
    for i, line in enumerate(lines[:max_lines]):
        ty = py + 28 + padding + i * line_h
        cv2.putText(frame, line, (px + 6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open camera!")
    exit(1)

show_panel = False
cooldown   = 0
last_detections = []

print("\n" + "="*58)
print("  YOLO + Llama3.2 Vision  |  100% Local, No API Key")
print("="*58)
print("  [S]  Snap frame -> Llama describes the scene")
print("  [H]  Toggle description panel")
print("  [Q]  Quit")
print("="*58 + "\n")
print(f"[Ollama] Using model: {OLLAMA_MODEL} (local)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    # YOLO runs every frame
    last_detections = run_yolo(frame)
    draw_yolo(frame, last_detections)

    # Unique label summary
    label_set = list({classes[d[0]] for d in last_detections})
    obj_text  = (
        f"YOLO: {len(last_detections)} obj  "
        f"[{', '.join(label_set[:5])}{'...' if len(label_set) > 5 else ''}]"
    )

    # Read AI state
    with ai_lock:
        cur_status = ai_status
        cur_lines  = ai_description.copy()
        snap_path  = ai_snapshot_path

    if show_panel and cur_lines:
        draw_ai_panel(frame, cur_lines, cur_status)

    hint = "  [S] Snap+Describe" if not (show_panel and cur_lines) \
           else f"  Snap: {os.path.basename(snap_path)}"
    draw_status_bar(frame, f"{obj_text}{hint}  [H] Panel  [Q] Quit")

    cv2.imshow("YOLO + Llama3.2 | Local AI Vision", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s') and cooldown == 0:
        trigger_analysis(frame.copy(), last_detections)
        show_panel = True
        cooldown   = 20
    elif key == ord('h'):
        show_panel = not show_panel

    if cooldown > 0:
        cooldown -= 1

cap.release()
cv2.destroyAllWindows()
print("Exited.")
