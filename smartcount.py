from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


# =====================================
# PATHS / SETTINGS
# =====================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "my_model.pt"
HISTORY_CSV = BASE_DIR / "smartcount_history.csv"

CAMERA_INDEX = 0
CONF_THRES = 0.65
IMGRES = 512
SHOW_LABELS = True
SHOW_CONF = True

# Camera quality
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Performance tuning
FRAME_SKIP = 2          # run detection every N frames in local webcam mode
MAX_FPS_SMOOTH = 30     # only for display smoothing

# Device selection
# Uses CUDA if available and allowed, otherwise CPU
DEFAULT_DEVICE = "cuda:0" if os.environ.get("SMARTCOUNT_FORCE_CPU", "0") != "1" else "cpu"


# =====================================
# MODEL LOADING
# =====================================
_MODEL: Optional[YOLO] = None
_MODEL_DEVICE: Optional[str] = None


def resolve_device(device: Optional[str] = None) -> str:
    if device and str(device).strip():
        return str(device).strip()
    return DEFAULT_DEVICE


def get_model(device: Optional[str] = None) -> YOLO:
    global _MODEL, _MODEL_DEVICE

    chosen_device = resolve_device(device)

    if _MODEL is None:
        print(f"Loading model from: {MODEL_PATH}")
        _MODEL = YOLO(str(MODEL_PATH))
        _MODEL_DEVICE = chosen_device
        print(f"Model loaded. Preferred device: {_MODEL_DEVICE}")
    elif _MODEL_DEVICE != chosen_device:
        _MODEL_DEVICE = chosen_device

    return _MODEL


# =====================================
# COLORS
# =====================================
def get_class_colors() -> Dict[str, Tuple[int, int, int]]:
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
# HELPERS
# =====================================
def ensure_obb_result(res) -> None:
    if getattr(res, "obb", None) is None:
        raise RuntimeError("Model does not provide OBB results.")


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower()


def predict_frame(
    frame: np.ndarray,
    conf_thres: float = CONF_THRES,
    imgres: int = IMGRES,
    device: Optional[str] = None,
):
    model = get_model(device)
    chosen_device = resolve_device(device)

    try:
        res = model.predict(
            frame,
            imgsz=imgres,
            conf=conf_thres,
            verbose=False,
            device=chosen_device,
        )[0]
    except Exception:
        # fallback for environments without GPU support
        res = model.predict(
            frame,
            imgsz=imgres,
            conf=conf_thres,
            verbose=False,
            device="cpu",
        )[0]

    ensure_obb_result(res)
    return res


def extract_counts(res, conf_thres: float = CONF_THRES) -> Dict[str, int]:
    ensure_obb_result(res)

    counts = defaultdict(int)
    if len(res.obb) == 0:
        return {}

    class_ids = res.obb.cls.cpu().numpy().astype(int)
    confs = res.obb.conf.cpu().numpy()

    for cid, conf in zip(class_ids, confs):
        if conf < conf_thres:
            continue
        class_name = normalize_class_name(res.names[cid])
        counts[class_name] += 1

    return dict(counts)


def extract_detections(
    res,
    conf_thres: float = CONF_THRES,
) -> List[Tuple[np.ndarray, str, float]]:
    ensure_obb_result(res)

    detections: List[Tuple[np.ndarray, str, float]] = []
    if len(res.obb) == 0:
        return detections

    corners = res.obb.xyxyxyxy.cpu().numpy().astype(np.int32)
    class_ids = res.obb.cls.cpu().numpy().astype(int)
    confs = res.obb.conf.cpu().numpy()

    for pts, cid, conf in zip(corners, class_ids, confs):
        if conf < conf_thres:
            continue
        class_name = normalize_class_name(res.names[cid])
        detections.append((pts, class_name, float(conf)))

    return detections


# =====================================
# DRAW DETECTIONS
# =====================================
def draw_obb_fast(
    frame: np.ndarray,
    res,
    conf_thres: float = CONF_THRES,
    show_labels: bool = SHOW_LABELS,
    show_conf: bool = SHOW_CONF,
) -> np.ndarray:
    ensure_obb_result(res)

    if len(res.obb) == 0:
        return frame

    colors = get_class_colors()
    detections = extract_detections(res, conf_thres)

    h, w = frame.shape[:2]
    line_thickness = max(2, int(min(h, w) / 320))
    font_scale = max(0.6, min(h, w) / 900)
    font_thickness = max(2, int(min(h, w) / 420))
    padding = max(4, int(min(h, w) / 260))

    for pts, class_name, conf in detections:
        color = colors.get(class_name, (0, 255, 0))
        pts_reshaped = pts.reshape((-1, 1, 2))

        cv2.polylines(frame, [pts_reshaped], True, color, line_thickness)

        label_parts = []
        if show_labels:
            label_parts.append(class_name)
        if show_conf:
            label_parts.append(f"{conf:.2f}")

        if label_parts:
            label = " ".join(label_parts)

            x = int(np.min(pts[:, 0]))
            y = int(np.min(pts[:, 1])) - 8

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            top = max(0, y - th - padding)
            bottom = max(th + padding, y + baseline + padding)

            cv2.rectangle(
                frame,
                (x, top),
                (x + tw + padding * 2, bottom),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x + padding, max(th, y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

    return frame


# =====================================
# OVERLAY UI
# =====================================
def draw_live_overlay(
    frame: np.ndarray,
    class_counts: Dict[str, int],
    frame_idx: int,
    fps_value: float,
    title: str = "SMARTCOUNT (Press Q to finish)",
) -> np.ndarray:
    padding = 10
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    total = sum(class_counts.values())
    lines = [
        title,
        f"Frame: {frame_idx}",
        f"FPS: {fps_value:.1f}",
    ]

    for cls_name in sorted(class_counts.keys()):
        lines.append(f"{cls_name}: {class_counts[cls_name]}")

    lines.append(f"TOTAL: {total}")

    box_w = 360
    box_h = padding * 2 + line_height * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)

    for i, line in enumerate(lines):
        y = padding + (i + 1) * line_height
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 255)
        elif line.startswith("TOTAL"):
            color = (0, 255, 0)

        cv2.putText(frame, line, (padding, y), font, font_scale, color, thickness)

    return frame


# =====================================
# ONE-STEP FRAME PROCESSOR
# Useful for Streamlit / WebRTC
# =====================================
def process_frame(
    frame: np.ndarray,
    conf_thres: float = CONF_THRES,
    imgres: int = IMGRES,
    device: Optional[str] = None,
    show_overlay: bool = False,
    frame_idx: int = 0,
    fps_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, int], object]:
    res = predict_frame(frame, conf_thres=conf_thres, imgres=imgres, device=device)
    counts = extract_counts(res, conf_thres=conf_thres)

    annotated = frame.copy()
    annotated = draw_obb_fast(
        annotated,
        res,
        conf_thres=conf_thres,
        show_labels=SHOW_LABELS,
        show_conf=SHOW_CONF,
    )

    if show_overlay:
        annotated = draw_live_overlay(
            annotated,
            counts,
            frame_idx=frame_idx,
            fps_value=fps_value,
        )

    return annotated, counts, res


# =====================================
# SAVE HISTORY
# =====================================
def save_history(final_counts: Dict[str, int], source_type: str = "Webcam") -> None:
    filepath = HISTORY_CSV
    file_exists = filepath.exists()

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
# LOCAL DESKTOP WEBCAM MODE
# Only runs when this file is executed directly
# =====================================
def run_local_webcam(
    camera_index: int = CAMERA_INDEX,
    conf_thres: float = CONF_THRES,
    imgres: int = IMGRES,
    device: Optional[str] = None,
    cam_width: int = CAM_WIDTH,
    cam_height: int = CAM_HEIGHT,
    frame_skip: int = FRAME_SKIP,
) -> Dict[str, int]:
    print("Opening camera...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    print("Camera opened.")
    print(f"Requested capture size: {cam_width}x{cam_height}")
    print("Running detection... press 'q' when ready to finalize")

    frame_idx = 0
    last_counts: Dict[str, int] = {}
    last_result = None
    prev_time = time.time()
    fps_smoothed = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            now = time.time()
            fps_instant = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_smoothed = (
                fps_instant if fps_smoothed == 0.0
                else min(MAX_FPS_SMOOTH, 0.85 * fps_smoothed + 0.15 * fps_instant)
            )

            should_process = (
                last_result is None or
                frame_skip <= 1 or
                frame_idx % frame_skip == 0
            )

            if should_process:
                try:
                    result = predict_frame(
                        frame,
                        conf_thres=conf_thres,
                        imgres=imgres,
                        device=device,
                    )
                    last_result = result
                    last_counts = extract_counts(result, conf_thres=conf_thres)
                except Exception as e:
                    print(f"Inference warning: {e}")

            annotated = frame.copy()

            if last_result is not None:
                annotated = draw_obb_fast(
                    annotated,
                    last_result,
                    conf_thres=conf_thres,
                    show_labels=SHOW_LABELS,
                    show_conf=SHOW_CONF,
                )

            annotated = draw_live_overlay(
                annotated,
                last_counts,
                frame_idx=frame_idx,
                fps_value=fps_smoothed,
            )

            cv2.imshow("SMARTCOUNT", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    final_counts = {k: v for k, v in last_counts.items() if v > 0}
    save_history(final_counts, "Webcam")
    return final_counts


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    run_local_webcam()
