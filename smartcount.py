from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import sqlite3
import statistics
import cv2
import csv
import os

# =====================================
# SETTINGS
# =====================================
MODEL_PATH = "my_model.pt"
DB_PATH = "smartcount.db"
CSV_PATH = "inventory_reports.csv"
CAMERA_INDEX = 0
CAMERA_NAME = "cam_1"

CONF_THRES = 0.7
IMGSZ = 640
FRAME_SKIP = 2
MIN_SEEN_FRAMES = 2

# =====================================
# DATABASE SETUP
# =====================================
session_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS inventory_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    session_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    final_count INTEGER NOT NULL,
    method TEXT NOT NULL,
    camera_name TEXT NOT NULL
)
""")

conn.commit()

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
    conn.close()
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


def save_final_report(final_counts):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cls_name, count in final_counts.items():
        cursor.execute("""
            INSERT INTO inventory_reports (
                event_time, session_id, class_name,
                final_count, method, camera_name
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_time,
            session_id,
            cls_name,
            int(count),
            "manual_finish_percentile",
            CAMERA_NAME
        ))

    conn.commit()


def append_final_csv(final_counts):
    filepath = os.path.join(os.getcwd(), CSV_PATH)
    file_exists = os.path.exists(filepath)
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "event_time",
                "session_id",
                "class_name",
                "final_count",
                "method",
                "camera_name"
            ])

        for cls_name, count in sorted(final_counts.items()):
            writer.writerow([
                event_time,
                session_id,
                cls_name,
                int(count),
                "manual_finish_percentile",
                CAMERA_NAME
            ])

    print("Updated CSV:", filepath)

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

        if res.obb is not None and len(res.obb) > 0:
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

    save_final_report(final_counts)
    append_final_csv(final_counts)

    print("\nFinal inventory:")
    if final_counts:
        for cls_name in sorted(final_counts.keys()):
            print(f"{cls_name}: {final_counts[cls_name]}")
        print("TOTAL:", sum(final_counts.values()))
    else:
        print("No items found.")

    conn.close()
    print("Saved to database:", DB_PATH)
