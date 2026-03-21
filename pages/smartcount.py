import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
import numpy as np
import tempfile
import cv2
import math
import subprocess
import sys
import os

MODEL_PATH = "my_model.pt"
WEBCAM_SCRIPT = "smartcount.py"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def box_size(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def mean_box_size(detections):
    if not detections:
        return 0.0, 0.0

    widths = []
    heights = []
    for _, _, box in detections:
        w, h = box_size(box)
        widths.append(w)
        heights.append(h)

    return float(np.mean(widths)), float(np.mean(heights))


def get_class_rules():
    return {
        "Sweets": {"iou": 0.10, "dist_factor": 1.0},
        "Oranges": {"iou": 0.12, "dist_factor": 1.1},
        "Apples": {"iou": 0.10, "dist_factor": 0.90},
        "Soft Drinks": {"iou": 0.15, "dist_factor": 1.1},
        "Packet Drinks": {"iou": 0.15, "dist_factor": 0.80},
        "Bread": {"iou": 0.15, "dist_factor": 0.75},
        "Noodles": {"iou": 0.15, "dist_factor": 0.85},
        "Chips": {"iou": 0.18, "dist_factor": 0.85},
    }


def suppress_duplicates_per_class(detections):
    by_class = defaultdict(list)
    for det in detections:
        by_class[det[0]].append(det)

    kept = []
    rules = get_class_rules()

    for class_name, dets in by_class.items():
        dets = sorted(dets, key=lambda x: x[1], reverse=True)

        avg_w, avg_h = mean_box_size(dets)
        base_dist = max(avg_w, avg_h)

        rule = rules.get(class_name, {"iou": 0.15, "dist_factor": 0.80})
        iou_threshold = rule["iou"]
        min_center_dist = base_dist * rule["dist_factor"]

        selected = []

        for det in dets:
            _, _, box = det
            keep = True

            for kept_det in selected:
                _, _, kept_box = kept_det

                iou = compute_iou(box, kept_box)
                dist = center_distance(box, kept_box)

                if iou > iou_threshold or dist < min_center_dist:
                    keep = False
                    break

            if keep:
                selected.append(det)

        kept.extend(selected)

    return kept


def get_detections_and_counts(res, conf_threshold=0.75):
    detections = []

    if getattr(res, "obb", None) is not None and len(res.obb) > 0:
        class_ids = res.obb.cls.cpu().numpy().astype(int)
        confs = res.obb.conf.cpu().numpy()
        corners = res.obb.xyxyxyxy.cpu().numpy()

        for cid, conf, pts in zip(class_ids, confs, corners):
            if conf < conf_threshold:
                continue

            xs = pts[:, 0]
            ys = pts[:, 1]
            x1, y1 = float(xs.min()), float(ys.min())
            x2, y2 = float(xs.max()), float(ys.max())

            class_name = res.names[cid]
            detections.append((class_name, float(conf), [x1, y1, x2, y2]))

    elif getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
        class_ids = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()

        for cid, conf, box in zip(class_ids, confs, boxes):
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            class_name = res.names[cid]
            detections.append((class_name, float(conf), [x1, y1, x2, y2]))

    filtered = suppress_duplicates_per_class(detections)

    class_counts = defaultdict(int)
    for class_name, _, _ in filtered:
        class_counts[class_name] += 1

    return filtered, dict(class_counts)


def stable_video_count(all_counts):
    final_counts = {}

    for cls_name, values in all_counts.items():
        nonzero = [v for v in values if v > 0]
        if nonzero:
            nonzero.sort()
            idx = int(0.75 * (len(nonzero) - 1))
            final_counts[cls_name] = int(round(nonzero[idx]))

    return final_counts


def get_class_colors():
    return {
        "Apples": (255, 80, 80),
        "Bread": (255, 220, 0),
        "Chips": (255, 255, 255),
        "Noodles": (160, 255, 0),
        "Oranges": (120, 70, 20),
        "Packet Drinks": (220, 80, 220),
        "Soft Drinks": (70, 100, 255),
        "Sweets": (0, 255, 255),
    }


def draw_filtered_boxes(image_rgb, filtered_detections):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    colors = get_class_colors()

    h, w = image_bgr.shape[:2]
    font_scale = max(1.0, min(w, h) / 700)
    thickness = max(2, int(min(w, h) / 350))
    padding = max(4, int(min(w, h) / 250))

    for class_name, conf, box in filtered_detections:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = colors.get(class_name, (0, 255, 0))

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        top = max(0, y1 - th - padding * 2)

        cv2.rectangle(
            image_bgr,
            (x1, top),
            (x1 + tw + padding * 2, top + th + padding * 2),
            color,
            -1
        )

        cv2.putText(
            image_bgr,
            label,
            (x1 + padding, top + th + padding // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def launch_webcam_script():
    script_path = os.path.join(os.getcwd(), WEBCAM_SCRIPT)

    if not os.path.exists(script_path):
        st.error(f"{WEBCAM_SCRIPT} not found in the project folder.")
        return

    try:
        subprocess.Popen([sys.executable, script_path])
        st.success("Live webcam started in a separate window.")
        st.info("Use the OpenCV popup window and press 'q' there to finish.")
    except Exception as e:
        st.error(f"Failed to start webcam script: {e}")


def show_counts(counts):
    st.subheader("Detected Counts")

    if not counts:
        st.warning("No objects detected.")
        return

    total = sum(counts.values())

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Classes Detected", len(counts))
    with c2:
        st.metric("Total Items", total)

    st.markdown("### Breakdown")
    for class_name, count in sorted(counts.items()):
        st.markdown(
            f"""
            <div style="
                padding:12px 16px;
                margin-bottom:10px;
                border-radius:12px;
                background:#f7f7f9;
                border:1px solid #e6e6ea;
                display:flex;
                justify-content:space-between;
                align-items:center;
                font-size:20px;
            ">
                <span><b>{class_name}</b></span>
                <span>{count}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption("These counts are based on filtered boxes, not raw model detections.")


def render_header():
    st.markdown(
        """
        <style>
        .smartcount-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .smartcount-subtitle {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="smartcount-title">SmartCount</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="smartcount-subtitle">Choose an input type for counting.</div>',
        unsafe_allow_html=True
    )


def render(go_to):
    render_header()

    top_cols = st.columns([1, 4])
    with top_cols[0]:
        if st.button("← Back", use_container_width=True):
            go_to("home")
            st.rerun()

    model = load_model()

    st.markdown("### Settings")
    conf_thres = st.slider("Confidence Threshold", 0.05, 0.95, 0.75, 0.05)
    imgsz = st.select_slider("Image Size", options=[320, 512, 640, 800, 960], value=640)

    option = st.radio(
        "Select input type",
        ["Upload Image", "Upload Video", "Webcam Live"],
        horizontal=True
    )

    st.divider()

    if option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=1200)

            image_np = np.array(image)

            res = model.predict(
                image_np,
                conf=conf_thres,
                imgsz=imgsz,
                verbose=False
            )[0]

            filtered, counts = get_detections_and_counts(res, conf_thres)
            annotated = draw_filtered_boxes(image_np, filtered)

            st.image(annotated, caption="Filtered Prediction Result", width=1200)
            show_counts(counts)

    elif option == "Upload Video":
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv"]
        )

        if uploaded_video is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

            st.video(video_path)

            if st.button("Process Video"):
                cap = cv2.VideoCapture(video_path)
                all_counts = defaultdict(list)
                frame_idx = 0

                preview = st.empty()
                progress_text = st.empty()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    if frame_idx % 5 != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    res = model.predict(
                        frame_rgb,
                        conf=conf_thres,
                        imgsz=imgsz,
                        verbose=False
                    )[0]

                    filtered, counts = get_detections_and_counts(res, conf_thres)

                    for cls in set(all_counts.keys()) | set(counts.keys()):
                        all_counts[cls].append(counts.get(cls, 0))

                    annotated = draw_filtered_boxes(frame_rgb, filtered)
                    preview.image(annotated, width=1200)
                    progress_text.write(f"Processed frame: {frame_idx}")

                cap.release()

                final_counts = stable_video_count(all_counts)

                st.success("Video processing completed.")
                show_counts(final_counts)

    elif option == "Webcam Live":
        st.info("This mode launches your existing OpenCV live counting app in a separate window.")
        st.write("Press **q** inside the popup window to finish live counting.")

        if st.button("Start Live Webcam"):
            launch_webcam_script()