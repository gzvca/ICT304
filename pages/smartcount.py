import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
import numpy as np
import tempfile
import cv2
import subprocess
import sys
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

MODEL_PATH = "my_model.pt"
WEBCAM_SCRIPT = "smartcount.py"
HISTORY_CSV = "smartcount_history.csv"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f4f8fb 0%, #eef4f9 100%);
        }

        .block-container {
            max-width: 1200px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .hero-wrap {
            background: linear-gradient(135deg, #0B2A4A 0%, #1e4f82 50%, #2F6FA3 100%);
            border-radius: 28px;
            padding: 34px 36px;
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 16px 35px rgba(11, 42, 74, 0.18);
            position: relative;
            overflow: hidden;
        }

        .hero-wrap::after {
            content: "";
            position: absolute;
            right: -60px;
            top: -60px;
            width: 220px;
            height: 220px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        .hero-subtitle {
            font-size: 1.06rem;
            line-height: 1.7;
            opacity: 0.95;
            max-width: 780px;
            position: relative;
            z-index: 1;
        }

        .panel {
            background: rgba(255,255,255,0.92);
            border: 1px solid #d7e4ef;
            border-radius: 24px;
            padding: 22px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }

        .panel-title {
            font-size: 1.22rem;
            font-weight: 800;
            color: #0B2A4A;
            margin-bottom: 12px;
        }

        .stat-card {
            border-radius: 20px;
            padding: 18px;
            color: white;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            margin-bottom: 10px;
        }

        .stat-blue {
            background: linear-gradient(135deg, #2F6FA3 0%, #4f8fc4 100%);
        }

        .stat-dark {
            background: linear-gradient(135deg, #0B2A4A 0%, #234e7b 100%);
        }

        .stat-label {
            font-size: 0.92rem;
            opacity: 0.92;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.05;
        }

        .count-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(180deg, #ffffff 0%, #f5f9fd 100%);
            border: 1px solid #dce8f2;
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.03);
        }

        .count-name {
            font-weight: 700;
            color: #0B2A4A;
        }

        .count-value {
            background: #e9f2fa;
            color: #2F6FA3;
            font-weight: 800;
            padding: 6px 12px;
            border-radius: 999px;
            min-width: 42px;
            text-align: center;
        }

        .alert-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #9a3412;
            margin-bottom: 10px;
        }

        .alert-box {
            background: linear-gradient(180deg, #fff7ed 0%, #ffedd5 100%);
            border: 1px solid #fdba74;
            color: #9a3412;
            padding: 13px 14px;
            border-radius: 14px;
            margin-bottom: 9px;
            font-weight: 600;
            line-height: 1.5;
        }

        .ok-box {
            background: linear-gradient(180deg, #ecfdf5 0%, #dcfce7 100%);
            border: 1px solid #86efac;
            color: #166534;
            padding: 14px;
            border-radius: 14px;
            font-weight: 700;
        }

        .soft-box {
            background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #93c5fd;
            color: #1d4ed8;
            padding: 16px;
            border-radius: 16px;
            line-height: 1.6;
        }

        .note {
            color: #64748B;
            font-size: 0.94rem;
            margin-top: 8px;
            line-height: 1.6;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2F6FA3 0%, #3f82bb 100%);
            color: white;
            border-radius: 14px;
            font-weight: 800;
            border: none;
            padding: 0.78rem 1rem;
            box-shadow: 0 8px 18px rgba(47, 111, 163, 0.22);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #0B2A4A 0%, #1e4f82 100%);
            color: white;
        }

        .stButton > button:focus:not(:active) {
            color: white;
            border-color: transparent;
        }

        div[role="radiogroup"] label {
            background: white;
            border: 1px solid #d8e4ee;
            padding: 8px 14px;
            border-radius: 999px;
        }

        div[role="radiogroup"] label:hover {
            border-color: #2F6FA3;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def polygon_bbox(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def ensure_obb_result(res):
    if getattr(res, "obb", None) is None:
        raise RuntimeError("This model output does not contain oriented detections.")


def get_detections_and_counts(res, conf_threshold=0.70):
    """
    Use raw OBB detections only.
    No extra duplicate suppression.
    """
    ensure_obb_result(res)

    detections = []
    class_counts = defaultdict(int)

    if len(res.obb) > 0:
        class_ids = res.obb.cls.cpu().numpy().astype(int)
        confs = res.obb.conf.cpu().numpy()
        corners = res.obb.xyxyxyxy.cpu().numpy().astype(np.float32)

        for cid, conf, pts in zip(class_ids, confs, corners):
            if conf < conf_threshold:
                continue

            class_name = res.names[cid]
            detections.append((class_name, float(conf), pts.copy()))
            class_counts[class_name] += 1

    return detections, dict(class_counts)


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


def draw_filtered_detections(image_rgb, detections):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    colors = get_class_colors()

    h, w = image_bgr.shape[:2]
    font_scale = max(0.6, min(w, h) / 900)
    thickness = max(2, int(min(w, h) / 350))
    padding = max(4, int(min(w, h) / 250))

    for class_name, conf, poly in detections:
        pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        color = colors.get(class_name, (0, 255, 0))

        cv2.polylines(image_bgr, [pts], isClosed=True, color=color, thickness=thickness)

        x1, y1, _, _ = polygon_bbox(poly)
        x1 = int(x1)
        y1 = int(y1)

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
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Detected Counts</div>', unsafe_allow_html=True)

    if not counts:
        st.warning("No objects detected.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    total = sum(counts.values())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card stat-dark">
                <div class="stat-label">Classes Detected</div>
                <div class="stat-value">{len(counts)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card stat-blue">
                <div class="stat-label">Total Items</div>
                <div class="stat-value">{total}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="panel-title" style="margin-top:12px;">Breakdown</div>', unsafe_allow_html=True)

    for class_name, count in sorted(counts.items()):
        st.markdown(
            f"""
            <div class="count-row">
                <span class="count-name">{class_name}</span>
                <span class="count-value">{count}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="note">These counts are based on raw detections after confidence filtering.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


def check_alerts(counts):
    alerts = []

    for item, count in counts.items():
        if count < 3:
            alerts.append(f"⚠️ {item}: Low stock (restock soon) — only {count} left")

    return alerts


def show_alerts(counts):
    alerts = check_alerts(counts)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Inventory Alerts</div>', unsafe_allow_html=True)

    if alerts:
        st.markdown('<div class="alert-title">Attention needed</div>', unsafe_allow_html=True)
        for a in alerts:
            st.markdown(f'<div class="alert-box">{a}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">✅ Stock levels are sufficient</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def save_history(counts, source_type):
    total_items = sum(counts.values())
    low_stock_items = [item for item, count in counts.items() if count < 3]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "total_items": total_items,
        "classes_detected": len(counts),
        "low_stock_items": ", ".join(low_stock_items) if low_stock_items else "None",
        "counts_json": "; ".join([f"{k}: {v}" for k, v in sorted(counts.items())]),
    }

    file_exists = os.path.exists(HISTORY_CSV)

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def show_counts_chart(counts):
    if not counts:
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Counts Chart</div>', unsafe_allow_html=True)

    items = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(items, values)
    ax.set_ylabel("Count")
    ax.set_xlabel("Item")
    ax.set_title("Detected Item Counts")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


def read_history_rows():
    if not os.path.exists(HISTORY_CSV):
        return []

    rows = []
    with open(HISTORY_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def show_history_table():
    rows = read_history_rows()
    if not rows:
        return

    rows = list(reversed(rows))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Scan History</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        source_filter = st.selectbox(
            "Filter by Source",
            ["All", "Image", "Video", "Webcam"],
            index=0
        )

    with c2:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "OK", "Low Stock"],
            index=0
        )

    with c3:
        search_text = st.text_input("Search item in Count Breakdown", "")

    enriched_rows = []
    for row in rows:
        low_stock_items = row.get("low_stock_items", "None")
        counts_json = row.get("counts_json", "")
        status = "Low Stock" if low_stock_items != "None" else "OK"

        enriched_rows.append({
            "timestamp": row.get("timestamp", ""),
            "source": row.get("source", ""),
            "classes_detected": row.get("classes_detected", ""),
            "counts_json": counts_json,
            "total_items": row.get("total_items", ""),
            "low_stock_items": low_stock_items,
            "status": status,
        })

    filtered_rows = []
    for row in enriched_rows:
        if source_filter != "All" and row["source"] != source_filter:
            continue

        if status_filter != "All" and row["status"] != status_filter:
            continue

        if search_text.strip() and search_text.lower() not in row["counts_json"].lower():
            continue

        filtered_rows.append(row)

    total_scans = len(filtered_rows)
    latest_source = filtered_rows[0]["source"] if filtered_rows else "-"
    latest_total = filtered_rows[0]["total_items"] if filtered_rows else "-"
    low_stock_scans = sum(1 for r in filtered_rows if r["status"] == "Low Stock")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
            <div class="stat-card stat-dark">
                <div class="stat-label">Total Scans</div>
                <div class="stat-value">{total_scans}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="stat-card stat-blue">
                <div class="stat-label">Latest Source</div>
                <div class="stat-value" style="font-size:1.4rem;">{latest_source}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div class="stat-card stat-dark">
                <div class="stat-label">Latest Total Items</div>
                <div class="stat-value">{latest_total}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s4:
        st.markdown(
            f"""
            <div class="stat-card stat-blue">
                <div class="stat-label">Low Stock Scans</div>
                <div class="stat-value">{low_stock_scans}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    row_html = ""
    for row in filtered_rows[:20]:
        status_badge = (
            '<span style="background:#fee2e2;color:#b91c1c;padding:6px 12px;border-radius:999px;font-weight:700;">Low Stock</span>'
            if row["status"] == "Low Stock"
            else
            '<span style="background:#dcfce7;color:#166534;padding:6px 12px;border-radius:999px;font-weight:700;">OK</span>'
        )

        counts_pretty = row["counts_json"].replace(";", "<br>")

        row_html += f"""
            <tr>
                <td>{row["timestamp"]}</td>
                <td>{row["source"]}</td>
                <td>{row["classes_detected"]}</td>
                <td>{counts_pretty}</td>
                <td>{row["total_items"]}</td>
                <td>{row["low_stock_items"]}</td>
                <td>{status_badge}</td>
            </tr>
        """

    table_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: Arial, sans-serif;
            }}

            .history-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                overflow: hidden;
                border-radius: 14px;
                margin-top: 12px;
            }}

            .history-table th {{
                background: linear-gradient(90deg, #1e3c72, #2a5298);
                color: white;
                text-align: center;
                padding: 12px;
                font-size: 0.95rem;
                position: sticky;
                top: 0;
            }}

            .history-table td {{
                padding: 12px;
                border-top: 1px solid #e5edf5;
                color: #334155;
                font-size: 0.94rem;
                text-align: center;
                background: white;
                vertical-align: middle;
            }}

            .history-table tr:nth-child(even) td {{
                background: #fafcff;
            }}

            .history-table tr:hover td {{
                background: #f5f7fa;
            }}

            .history-table td:nth-child(4) {{
                text-align: left;
                white-space: normal;
                word-break: break-word;
                min-width: 230px;
                line-height: 1.6;
            }}

            .history-table td:nth-child(6) {{
                white-space: normal;
                word-break: break-word;
                min-width: 180px;
            }}

            .history-table td:nth-child(7) {{
                min-width: 120px;
            }}
        </style>
    </head>
    <body>
        <table class="history-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Source</th>
                    <th>Class Detected</th>
                    <th>Count Breakdown</th>
                    <th>Total Items</th>
                    <th>Low Stock</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {row_html}
            </tbody>
        </table>
    </body>
    </html>
    """

    table_height = 120 + min(len(filtered_rows[:20]), 20) * 58
    components.html(table_html, height=table_height, scrolling=True)

    with open(HISTORY_CSV, "rb") as f:
        st.download_button(
            "Download History CSV",
            data=f.read(),
            file_name="smartcount_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


def render_header():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">SmartCount</div>
            <div class="hero-subtitle">
                AI-powered inventory counting for images, uploaded videos, and live webcam monitoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def run_prediction(model, image_rgb, conf_thres, imgsz):
    res = model.predict(
        image_rgb,
        conf=conf_thres,
        imgsz=imgsz,
        verbose=False
    )[0]

    ensure_obb_result(res)
    return res


def render(go_to):
    inject_css()
    render_header()

    top_cols = st.columns([2, 4, 2])
    with top_cols[0]:
        if st.button("← Back", width='stretch'):
            go_to("home")
            st.rerun()
            
    with top_cols[2]:
        if st.button("Go to SmartCast →", width='stretch'):
            go_to("smartcast")
            st.rerun()

    model = load_model()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Settings</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        conf_thres = st.slider("Confidence Threshold", 0.05, 0.95, 0.40, 0.05)
    with c2:
        imgsz = st.select_slider("Image Size", options=[320, 512, 640, 800, 960], value=640)

    st.markdown(
        '<div class="note">For better results, try increasing confidence threshold and reducing image size.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Input Type</div>', unsafe_allow_html=True)

    option = st.radio(
        "Select input type",
        ["Upload Image", "Upload Video", "Webcam Live"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if option == "Upload Image":
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=1200)

                image_np = np.array(image)
                res = run_prediction(model, image_np, conf_thres, imgsz)

                detections, counts = get_detections_and_counts(res, conf_thres)
                annotated = draw_filtered_detections(image_np, detections)

                st.image(annotated, caption="Prediction Result", width=1200)
                show_counts(counts)
                show_alerts(counts)
                show_counts_chart(counts)
                save_history(counts, "Image")
            except Exception as e:
                st.error(f"Processing failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    elif option == "Upload Video":
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Video</div>', unsafe_allow_html=True)

        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv"],
            label_visibility="collapsed"
        )

        if uploaded_video is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_video.read())
            video_path = temp_file.name

            st.video(video_path)

            if st.button("Process Video", use_container_width=True):
                try:
                    cap = cv2.VideoCapture(video_path)
                    all_counts = defaultdict(list)
                    frame_idx = 0

                    preview = st.empty()
                    progress_bar = st.progress(0, text="Processing video...")

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_frames = max(total_frames, 1)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = run_prediction(model, frame_rgb, conf_thres, imgsz)

                        detections, counts = get_detections_and_counts(res, conf_thres)

                        for cls in set(all_counts.keys()) | set(counts.keys()):
                            all_counts[cls].append(counts.get(cls, 0))

                        annotated = draw_filtered_detections(frame_rgb, detections)
                        preview.image(annotated, width=1200)

                        progress_value = min(frame_idx / total_frames, 1.0)
                        progress_bar.progress(progress_value, text=f"Processing frame {frame_idx}...")

                    cap.release()

                    final_counts = stable_video_count(all_counts)
                    progress_bar.empty()

                    st.success("Video processing completed.")
                    show_counts(final_counts)
                    show_alerts(final_counts)
                    show_counts_chart(final_counts)
                    save_history(final_counts, "Video")
                except Exception as e:
                    st.error(f"Video processing failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    elif option == "Webcam Live":
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Live Webcam</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="soft-box">
                This mode launches your existing OpenCV live counting app in a separate window.<br>
                Press <b>q</b> inside the popup window to finish live counting.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Start Live Webcam", use_container_width=True):
            launch_webcam_script()

        st.markdown('</div>', unsafe_allow_html=True)

    show_history_table()