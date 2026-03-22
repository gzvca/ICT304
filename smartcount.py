from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import cv2
import csv
import os

# =====================================
# SETTINGS
# =====================================
MODEL_PATH = "my_model.pt"
HISTORY_CSV = "smartcount_history.csv"
CAMERA_INDEX = 0

CONF_THRES = 0.7
IMGSZ = 640
FRAME_SKIP = 2
MIN_SEEN_FRAMES = 2

# =====================================
# LOAD MODEL
# =====================================
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded")

# =====================================
# OPEN CAMERA
# =====================================
print("Opening camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise FileNotFoundError("Could not open camera.")
print("Camera opened")

# =====================================
# HELPERS
# =====================================
def draw_overlay(frame, class_counts, frame_idx):
    padding = 12
    line_height = 28
    font = cv2.FONT_HERSHEY_SIMPLEX

    total = sum(class_counts.values())

    lines = [
        "SMARTCOUNT (Press Q to finish)",
        f"Frame: {frame_idx}",
    ]

    for cls_name in sorted(class_counts.keys()):
        lines.append(f"{cls_name}: {class_counts[cls_name]}")

    lines.append(f"TOTAL: {total}")

    box_w = 360
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

        cv2.putText(frame, line, (padding, y), font, 0.7, color, 2)

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
print("Running... press 'q' when ready to finalize")

frame_idx = 0
count_history = defaultdict(list)
seen_frames = defaultdict(int)
last_annotated = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_idx += 1

        if FRAME_SKIP > 1 and (frame_idx % FRAME_SKIP != 0):
            if last_annotated is not None:
                cv2.imshow("SMARTCOUNT", last_annotated)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("Finalizing...")
                break
            continue

        res = model.predict(
            frame,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            verbose=False
        )[0]

        current_counts = defaultdict(int)

        if getattr(res, "obb", None) is not None and len(res.obb) > 0:
            class_ids = res.obb.cls.cpu().numpy().astype(int)
            for cid in class_ids:
                class_name = res.names[cid]
                current_counts[class_name] += 1

        elif getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            class_ids = res.boxes.cls.cpu().numpy().astype(int)
            for cid in class_ids:
                class_name = res.names[cid]
                current_counts[class_name] += 1

        for cls_name, count in current_counts.items():
            if count > 0:
                seen_frames[cls_name] += 1

        all_classes = set(count_history.keys()) | set(current_counts.keys())
        for cls_name in all_classes:
            count_history[cls_name].append(current_counts.get(cls_name, 0))

        annotated = res.plot()
        annotated = draw_overlay(annotated, current_counts, frame_idx)

        last_annotated = annotated
        cv2.imshow("SMARTCOUNT", annotated)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
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
