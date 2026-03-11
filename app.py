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

# Custom CSS untuk tampilan lebih profesional
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Sistem Deteksi Penambat Rel (YOLOv8)")
st.write("Aplikasi cerdas untuk mendeteksi jenis penambat dan mengidentifikasi komponen yang hilang pada jalur kereta api.")

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/ae/Logo_Institut_Teknologi_Bandung.svg/1200px-Logo_Institut_Teknologi_Bandung.svg.png", width=100)
st.sidebar.header("⚙️ Pengaturan Model")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, help="Semakin tinggi, semakin ketat deteksinya.")
model_file = st.sidebar.file_uploader("1. Upload Model YOLO (.pt)", type=['pt'])
video_file = st.sidebar.file_uploader("2. Upload Video Rekaman Rel (.mp4, .avi)", type=['mp4', 'avi'])

# --- LOGIKA UTAMA ---
if model_file and video_file:
    # Simpan file sementara untuk dibaca OpenCV & YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model:
        tmp_model.write(model_file.read())
        model_path = tmp_model.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    # Load Model dengan Error Handling
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Hitung (Bisa disesuaikan dengan sudut pandang kamera Anda)
    y_atas, y_bawah = int(0.20 * height), int(0.80 * height)
    roi_points = np.array([
        [int(0.30 * width), y_bawah], [int(0.40 * width), y_atas],
        [int(0.60 * width), y_atas], [int(0.70 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.55 * height) # Garis pemicu hitungan

    # Inisialisasi State
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter({"DE CLIP": 0, "E Clip": 0, "KA Clip": 0, "Hilang": 0})
    
    # UI Kolom
    col1, col2 = st.columns([3, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    if st.button("🚀 Mulai Analisis Video"):
        st.toast("Memulai pemrosesan frame...")
        
        # Generator deteksi untuk efisiensi memori
        results = model.track(
            source=video_path, 
            persist=True, 
            imgsz=640, 
            stream=True, 
            conf=conf_threshold
        )

        for res in results:
            frame = res.orig_img
            
            # Visualisasi ROI (Hijau) & Garis Pemicu (Biru)
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.25*width), y_ref), (int(0.75*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Cek apakah objek berada di dalam ROI
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        # Logika Hitung saat melewati garis horizontal
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            # Label final diambil dari mayoritas deteksi selama tracking
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                        # Gambar Bounding Box
                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Update Tampilan Video
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Update Statistik Real-time
            with stats_placeholder.container():
                st.subheader("📊 Statistik")
                for cls_name, count in summary_counts.items():
                    st.metric(label=cls_name, value=count)
                st.write(f"**Total Objek Terhitung:** {len(counted_ids)}")

        st.success("✅ Analisis Selesai!")
        
        # Fitur Export Data ke CSV
        df_result = pd.DataFrame([summary_counts])
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Laporan Hasil Deteksi (CSV)",
            data=csv,
            file_name='hasil_deteksi_penambat.csv',
            mime='text/csv',
        )
        
    cap.release()
    # Bersihkan file sementara
    os.unlink(video_path)
    os.unlink(model_path)

else:
    # Tampilan awal saat belum upload file
    st.info("👋 Selamat datang! Silakan unggah file model `.pt` dan video rekaman rel pada sidebar untuk memulai proses analisis.")
    
    # Penjelasan singkat alur ROI & Tracking
    st.subheader("Cara Kerja Sistem:")
    st.write("""
    1. **Region of Interest (ROI)**: Sistem hanya mendeteksi objek di dalam area poligon hijau untuk akurasi maksimal.
    2. **Tracking**: Menggunakan ID unik untuk setiap objek agar tidak terhitung dua kali.
    3. **Line Counting**: Objek baru akan masuk ke statistik setelah melewati garis referensi biru.
    """)

