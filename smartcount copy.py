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

CONF_THRES = 0.75
IMGRES = 320
FRAME_SKIP = 2
MIN_SEEN_FRAMES = 2

SHOW_LABELS = True
SHOW_CONF = True

# Uncomment if you have CUDA
# DEVICE = 0
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
# HELPERS
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


def draw_obb_fast(frame, res):
    """
    Draw oriented bounding boxes manually.
    This avoids falling back to normal boxes.
    """
    if getattr(res, "obb", None) is None:
        raise RuntimeError("This model/output does not provide OBB results.")

    if len(res.obb) == 0:
        return frame

    names = res.names

    corners = res.obb.xyxyxyxy.cpu().numpy().astype(int)   # shape: (N, 4, 2)
    cls_ids = res.obb.cls.cpu().numpy().astype(int)
    confs = res.obb.conf.cpu().numpy()

    for pts, cid, conf in zip(corners, cls_ids, confs):
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        label_parts = []
        if SHOW_LABELS:
            label_parts.append(str(names[cid]))
        if SHOW_CONF:
            label_parts.append(f"{conf:.2f}")

        if label_parts:
            label = " ".join(label_parts)
            x = int(np.min(pts[:, 0, 0]))
            y = int(np.min(pts[:, 0, 1])) - 8
            cv2.putText(
                frame,
                label,
                (x, max(20, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return frame


def stable_count(values):
    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return 0

    nonzero.sort()
    idx = int(0.75 * (len(nonzero) - 1))
    return int(round(nonzero[idx]))


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
count_history = defaultdict(list)
seen_frames = defaultdict(int)
last_annotated = None

fps = 0.0
prev_time = time.time()
obb_checked = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame_idx += 1

        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            if last_annotated is not None:
                cv2.imshow("SMARTCOUNT", last_annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Finalizing...")
                break
            continue

        res = model.predict(
            frame,
            imgsz=IMGRES,
            conf=CONF_THRES,
            verbose=False,
            device=DEVICE,
        )[0]

        # Force OBB: never fall back to normal boxes
        if not obb_checked:
            obb_checked = True
            if getattr(res, "obb", None) is None:
                raise RuntimeError(
                    "Forced OBB mode is enabled, but this model does not output OBB results."
                )

        current_counts = defaultdict(int)

        if len(res.obb) > 0:
            class_ids = res.obb.cls.cpu().numpy().astype(int)
            for cid in class_ids:
                class_name = res.names[cid]
                current_counts[class_name] += 1

        for cls_name, count in current_counts.items():
            if count > 0:
                seen_frames[cls_name] += 1

        all_classes = set(count_history.keys()) | set(current_counts.keys())
        for cls_name in all_classes:
            count_history[cls_name].append(current_counts.get(cls_name, 0))

        annotated = frame.copy()
        annotated = draw_obb_fast(annotated, res)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        annotated = draw_live_overlay(annotated, current_counts, frame_idx, fps)

        last_annotated = annotated
        cv2.imshow("SMARTCOUNT", last_annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Finalizing...")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    final_counts = {}
    for cls_name, values in count_history.items():
        if seen_frames[cls_name] >= MIN_SEEN_FRAMES:
            estimate = stable_count(values)
            if estimate > 0:
                final_counts[cls_name] = estimate

    save_history(final_counts, "Webcam")

    print("\nFinal inventory:")
    if final_counts:
        for cls_name in sorted(final_counts.keys()):
            print(f"{cls_name}: {final_counts[cls_name]}")
        print("TOTAL:", sum(final_counts.values()))
    else:
        print("No items found.")
