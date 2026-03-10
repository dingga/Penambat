import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os

# Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Penambat Rel ITB", layout="wide")

st.title("🔍 Sistem Deteksi Penambat Rel (YOLOv8)")
st.write("Aplikasi ini mendeteksi jenis penambat dan menghitung komponen yang hilang secara real-time.")

# Sidebar untuk Pengaturan
st.sidebar.header("Konfigurasi Model")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
model_file = st.sidebar.file_uploader("Upload Model (.pt)", type=['pt'])
video_file = st.sidebar.file_uploader("Upload Video Rel (.mp4, .avi)", type=['mp4', 'avi'])

if model_file and video_file:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    # Load Model
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # ROI & Garis Hitung (Sesuai Logika Anda)
    y_atas, y_bawah = int(0.10 * height), int(0.70 * height)
    roi_points = np.array([
        [int(0.35 * width), y_bawah], [int(0.40 * width), y_atas],
        [int(0.60 * width), y_atas], [int(0.65 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.5 * height)

    # Inisialisasi State
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()
    
    # UI Kolom: Video & Statistik
    col1, col2 = st.columns([3, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    if st.button("Mulai Proses Deteksi"):
        # Stream processing
        results = model.track(source=video_path, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Visualisasi ROI & Garis
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Update Tampilan Video (Konversi BGR ke RGB)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Update Statistik di Sidebar/Kolom 2
            with stats_placeholder.container():
                st.subheader("📊 Hasil Perhitungan")
                for cls_name in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                    st.metric(label=cls_name, value=summary_counts[cls_name])
                st.write(f"Total Aset: {sum(summary_counts.values())}")

        st.success("Analisis Video Selesai!")
        
    cap.release()
    os.unlink(video_path)
    os.unlink(model_path)
else:
    st.info("Silakan unggah file model .pt dan video rel pada sidebar untuk memulai.")
