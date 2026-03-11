import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Penambat Rel ITB", 
    page_icon="🔍", 
    layout="wide"
)

# Inisialisasi Session State untuk Galeri
if 'gallery' not in st.session_state:
    st.session_state.gallery = []

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Sistem Deteksi Penambat Rel (YOLOv8)")
st.write("Aplikasi tesis untuk deteksi jenis penambat dan identifikasi kerusakan/komponen hilang secara real-time.")

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/ae/Logo_Institut_Teknologi_Bandung.svg/1200px-Logo_Institut_Teknologi_Bandung.svg.png", width=80)
st.sidebar.header("⚙️ Konfigurasi")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
model_file = st.sidebar.file_uploader("Upload Model YOLO (.pt)", type=['pt'])
video_file = st.sidebar.file_uploader("Upload Video Rekaman (.mp4, .avi)", type=['mp4', 'avi'])

# Tombol Reset Galeri di Sidebar
if st.sidebar.button("🗑️ Reset Galeri Temuan"):
    st.session_state.gallery = []
    st.sidebar.success("Galeri telah dibersihkan!")
    st.rerun()

# --- LOGIKA UTAMA ---
if model_file and video_file:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    # Load Model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Hitung (Koordinat Standar Jalur Rel)
    y_atas, y_bawah = int(0.25 * height), int(0.75 * height)
    roi_points = np.array([
        [int(0.30 * width), y_bawah], [int(0.40 * width), y_atas],
        [int(0.60 * width), y_atas], [int(0.70 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.50 * height) 

    # Inisialisasi State Per-Proses
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter({"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0})
    
    col1, col2 = st.columns([3, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    if st.button("🚀 Mulai Analisis Video"):
        results = model.track(source=video_path, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for res in results:
            frame = res.orig_img
            
            # Visualisasi ROI & Garis Pemicu
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.25*width), y_ref), (int(0.75*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)

                        # Deteksi saat melewati garis
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            # JIKA HILANG: Ambil Screenshot untuk Galeri
                            if final_label == "Hilang":
                                padding = 30
                                # Pastikan koordinat crop tidak keluar dari ukuran frame
                                y1_c, y2_c = max(0, int(y1)-padding), min(height, int(y2)+padding)
                                x1_c, x2_c = max(0, int(x1)-padding), min(width, int(x2)+padding)
                                
                                crop_img = frame[y1_c:y2_c, x1_c:x2_c]
                                if crop_img.size > 0:
                                    img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                                    st.session_state.gallery.append({
                                        "id": tid,
                                        "image": img_rgb,
                                        "time": time.strftime("%H:%M:%S")
                                    })

                        # Bounding Box Visual
                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            with stats_placeholder.container():
                st.subheader("📊 Statistik")
                for cls_name in ["DE CLIP", "E Clip", "KA Clip", "Hilang"]:
                    st.metric(label=cls_name, value=summary_counts[cls_name])
                st.write(f"**Total Objek:** {len(counted_ids)}")

        st.success("✅ Analisis Selesai!")
        
        # Download Laporan
        df_result = pd.DataFrame([summary_counts])
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Laporan (CSV)", data=csv, file_name='hasil_deteksi.csv', mime='text/csv')
        
    cap.release()
    os.unlink(video_path)
    os.unlink(model_path)

    # --- BAGIAN GALERI TEMUAN ---
    if st.session_state.gallery:
        st.divider()
        st.subheader("📸 Galeri Temuan Penambat Hilang")
        st.write("Potongan gambar otomatis untuk objek yang terdeteksi 'Hilang'.")
        
        # Tampilkan galeri dalam grid
        cols = st.columns(4)
        for idx, item in enumerate(reversed(st.session_state.gallery)): # Tampilkan yang terbaru di depan
            with cols[idx % 4]:
                st.image(item["image"], use_container_width=True, caption=f"ID: {item['id']} | {item['time']}")

else:
    st.info("👋 Silakan unggah model dan video pada sidebar untuk memulai analisis.")
