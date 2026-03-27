from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import cv2
import csv
import os
import time
import numpy as np

# =====================================
# SETTINGS
# =====================================
MODEL_PATH = "my_model.pt"
HISTORY_CSV = "smartcount_history.csv"
CAMERA_INDEX = 0

CONF_THRES = 0.70
IMGRES = 640
FRAME_SKIP = 1

SHOW_LABELS = True
SHOW_CONF = True

DEVICE = "cpu"

# =====================================
# LOAD MODEL
# =====================================
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# =====================================
# OPEN CAMERA
# =====================================
print("Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise FileNotFoundError("Could not open camera.")

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened.")


# =====================================
# COLORS 
# =====================================
def get_class_colors():
    return {
        "apples": (255, 80, 80),
        "bread": (255, 220, 0),
        "chips": (255, 255, 255),
        "noodles": (160, 255, 0),
        "oranges": (120, 70, 20),
        "packet drinks": (220, 80, 220),
        "soft drinks": (70, 100, 255),
        "sweets": (0, 255, 255),
    }


# =====================================
# OVERLAY UI
# =====================================
def draw_live_overlay(frame, class_counts, frame_idx, fps_value):
    padding = 10
    line_height = 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    total = sum(class_counts.values())
    lines = [
        "SMARTCOUNT (Press Q to finish)",
        f"Frame: {frame_idx}",
        f"FPS: {fps_value:.1f}",
    ]

    for cls_name in sorted(class_counts.keys()):
        lines.append(f"{cls_name}: {class_counts[cls_name]}")

    lines.append(f"TOTAL: {total}")

    box_w = 340
    box_h = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_height
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 255)
        elif "TOTAL" in line:
            color = (0, 255, 0)

        cv2.putText(frame, line, (padding, y), font, font_scale, color, thickness)

    return frame


# =====================================
# DRAW DETECTIONS (STRICT COLORS)
# =====================================
def draw_obb_fast(frame, res):
    if getattr(res, "obb", None) is None:
        raise RuntimeError("Model does not provide OBB results.")

    if len(res.obb) == 0:
        return frame

    names = res.names
    colors = get_class_colors()

    corners = res.obb.xyxyxyxy.cpu().numpy().astype(int)
    cls_ids = res.obb.cls.cpu().numpy().astype(int)
    confs = res.obb.conf.cpu().numpy()

    for pts, cid, conf in zip(corners, cls_ids, confs):
        if conf < CONF_THRES:
            continue

        class_name = str(names[cid]).strip().lower()

        if class_name not in colors:
            raise ValueError(f"Missing color for class: {class_name}")

        color = colors[class_name]

        pts = pts.reshape((-1, 1, 2))

        # draw box
        cv2.polylines(frame, [pts], True, color, 2)

        # label
        label_parts = []
        if SHOW_LABELS:
            label_parts.append(class_name)
        if SHOW_CONF:
            label_parts.append(f"{conf:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            x = int(np.min(pts[:, 0, 0]))
            y = int(np.min(pts[:, 0, 1])) - 8

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            padding = 4

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # label background
            cv2.rectangle(
                frame,
                (x, max(0, y - th - padding)),
                (x + tw + padding * 2, y + baseline + padding),
                color,
                -1
            )

            # label text
            cv2.putText(
                frame,
                label,
                (x + padding, y),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

    return frame


# =====================================
# SAVE HISTORY
# =====================================
def save_history(final_counts, source_type="Webcam"):
    filepath = os.path.join(os.getcwd(), HISTORY_CSV)
    file_exists = os.path.exists(filepath)

    total_items = sum(final_counts.values())
    low_stock_items = [name for name, count in final_counts.items() if count < 3]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "total_items": total_items,
        "classes_detected": len(final_counts),
        "low_stock_items": ", ".join(low_stock_items) if low_stock_items else "None",
        "counts_json": "; ".join([f"{k}: {v}" for k, v in sorted(final_counts.items())]),
    }

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Saved to:", filepath)


# =====================================
# MAIN LOOP
# =====================================
print("Running detection... press 'q' when ready to finalize")

frame_idx = 0
last_annotated = None
last_counts = {}

fps = 0.0
prev_time = time.time()
obb_checked = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        res = model.predict(
            frame,
            imgsz=IMGRES,
            conf=CONF_THRES,
            verbose=False,
            device=DEVICE,
        )[0]

        if not obb_checked:
            obb_checked = True
            if getattr(res, "obb", None) is None:
                raise RuntimeError("Model does not return OBB.")

        current_counts = defaultdict(int)

        if len(res.obb) > 0:
            class_ids = res.obb.cls.cpu().numpy().astype(int)
            confs = res.obb.conf.cpu().numpy()

            for cid, conf in zip(class_ids, confs):
                if conf < CONF_THRES:
                    continue
                class_name = str(res.names[cid]).strip().lower()
                current_counts[class_name] += 1

        annotated = frame.copy()
        annotated = draw_obb_fast(annotated, res)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_live_overlay(annotated, current_counts, frame_idx, fps)

        last_annotated = annotated
        last_counts = dict(current_counts)

        cv2.imshow("SMARTCOUNT", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    final_counts = {k: v for k, v in last_counts.items() if v > 0}
    save_history(final_counts, "Webcam")
