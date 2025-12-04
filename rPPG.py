import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from collections import deque
import time # Untuk potensi peningkatan kinerja di masa depan, saat ini tidak digunakan

# ========================================================
# KONFIGURASI DAN PARAMETER
# ========================================================
FPS = 30 # Asumsi Frame Rate Kamera
# Jendela Pemrosesan POS: 1.6 detik (standar rPPG)
WINDOW_LEN_SEC = 1.6
WINDOW_LEN_FRAMES = int(WINDOW_LEN_SEC * FPS) # Contoh: 48 frames
# Panjang data yang digunakan untuk plot, BPM, dan buffering (misal: 10 detik)
DISPLAY_LEN_FRAMES = int(10 * FPS) 
# Panjang buffer minimal untuk memulai perhitungan BPM (minimal 5 detik untuk FFT yang stabil)
MIN_BPM_FRAMES = int(5 * FPS) 
# Panjang minimum sinyal untuk filtfilt (harus > 21)
MIN_FILTER_LEN = 30 


# ========================================================
# 1. POS ALGORITHM (Diubah untuk menerima buffer fixed-length dan mengembalikan 1 nilai)
# ========================================================
def pos_projection(rgb_series):
    """
    Menghitung POS pada buffer RGB dengan panjang tetap.
    Mengembalikan 1 nilai sinyal rPPG (H) untuk frame terakhir.
    """
    rgb_series = np.array(rgb_series, dtype=np.float32)

    # Ambil hanya frame yang diperlukan (sesuai WINDOW_LEN_FRAMES)
    # Karena kita menggunakan deque dengan maxlen=DISPLAY_LEN_FRAMES, 
    # kita ambil 1.6 detik terakhir dari buffer tersebut.
    chunk = rgb_series[-WINDOW_LEN_FRAMES:]
    
    # 1. Centering
    rgb_centered = chunk - np.mean(chunk, axis=0)
    
    # 2. Normalisasi Temporal
    eps = 1e-8
    mean_chunk = np.mean(chunk, axis=0, keepdims=True) + eps
    chunk_norm = rgb_centered / mean_chunk
    
    # Matriks Proyeksi P
    P = np.array([[0, 1, -1],
                  [-2, 1, 1]], dtype=np.float32)

    # 3. Proyeksi S
    S = P @ chunk_norm.T
    X, Y = S[0], S[1]

    # 4. Ekstraksi Sinyal H
    scale = np.std(X) / (np.std(Y) + eps)
    H = X - scale * Y
    
    # 5. Detrending (Pusatkan sinyal)
    H = H - np.mean(H)
    
    # Kunci pengurangan delay: HANYA kembalikan nilai terakhir dari perhitungan jendela
    return H[-1] 


# ========================================================
# 2. BANDPASS FILTER
# ========================================================
def bandpass(signal, fs, low=0.67, high=4.0):
    """Menerapkan filter Butterworth orde 3 bandpass."""
    # PENGECEKAN PANJANG SINYAL untuk menghindari filtfilt ValueError
    if len(signal) < MIN_FILTER_LEN: 
        # Jika terlalu pendek, kembalikan sinyal apa adanya
        return signal
        
    nyquist = fs * 0.5
    b, a = butter(3, [low/nyquist, high/nyquist], btype="band")
    # filtfilt: Menerapkan filter maju dan mundur (menghilangkan phase shift)
    return filtfilt(b, a, signal)


# ========================================================
# 3. BPM CALCULATOR
# ========================================================
def estimate_bpm(signal, fs):
    """Menghitung BPM menggunakan FFT pada sinyal yang sudah difilter."""
    n = len(signal)
    
    # Cek apakah sinyal cukup panjang untuk FFT
    if n < MIN_BPM_FRAMES:
        return 0.0
        
    freqs = rfftfreq(n, 1/fs)
    mag = np.abs(rfft(signal))

    # Masking (Rentang 40 BPM - 240 BPM atau 0.67 Hz - 4.0 Hz)
    mask = (freqs >= 0.67) & (freqs <= 4.0)
    if not np.any(mask):
        return 0.0

    dominant = freqs[mask][np.argmax(mag[mask])]
    return dominant * 60


# ========================================================
# 4. MEDIAPIPE SETUP
# ========================================================
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# ========================================================
# 5. MATPLOTLIB LIVE PLOTS (Ditambahkan subplot FFT)
# ========================================================
plt.ion()
# Tiga subplot: RGB, POS (Filtered), dan FFT Spectrum
fig, (ax_rgb, ax_pos, ax_fft) = plt.subplots(3, 1, figsize=(10, 10))

# Plot RGB
(ax_r,) = ax_rgb.plot([], [], "r-", label="R")
(ax_g,) = ax_rgb.plot([], [], "g-", label="G")
(ax_b,) = ax_rgb.plot([], [], "b-", label="B")
ax_rgb.legend()
ax_rgb.set_title(f"RGB Signal (Last {DISPLAY_LEN_FRAMES/FPS}s)")
ax_rgb.set_ylim(0, 255)

# Plot POS
(pos_line,) = ax_pos.plot([], [], "m-", label="POS Filtered")
ax_pos.set_title(f"POS Output (Last {DISPLAY_LEN_FRAMES/FPS}s)")
ax_pos.set_ylim(-1, 1)

# Plot FFT
(fft_line,) = ax_fft.plot([], [], 'c-')
ax_fft.set_title("Frequency Spectrum")
ax_fft.set_xlabel("BPM")
ax_fft.set_xlim(40, 240)
ax_fft.set_ylim(0, 1) # Batas Y akan di-update secara dinamis

# ========================================================
# 6. REALTIME CAMERA & BUFFER DEQUE
# ========================================================
cap = cv2.VideoCapture(0)

# Menggunakan deque untuk buffer fixed-length agar tidak ada delay progresif
rgb_buffer = deque(maxlen=DISPLAY_LEN_FRAMES)
# Buffer ini akan menyimpan sinyal POS yang SUDAH DIFILTER
pos_buffer_filtered = deque(maxlen=DISPLAY_LEN_FRAMES) 
# Buffer sementara untuk POS mentah (sebelum filtering)
pos_buffer_raw = deque(maxlen=DISPLAY_LEN_FRAMES)

# Variabel untuk menahan nilai BPM terakhir
bpm_val = 0.0 

# ========================================================
# 7. MAIN LOOP
# ========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)
    
    # Warna kotak default
    box_color = (255, 50, 0) # Warna biru/ungu default

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0]

        # Landmark Pipi (ROI)
        cheek_pts = [234, 93, 132, 58, 172]
        xs = [int(lm.landmark[p].x * w) for p in cheek_pts]
        ys = [int(lm.landmark[p].y * h) for p in cheek_pts]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            b, g, r = np.mean(roi.reshape(-1, 3), axis=0)
            rgb_buffer.append([r, g, b])

            # ======== POS PROCESSING (Hanya jika buffer sudah mencapai panjang jendela) =========
            if len(rgb_buffer) >= WINDOW_LEN_FRAMES:
                
                # 1. Ekstraksi Sinyal POS (Mengembalikan 1 nilai rPPG mentah)
                pos_point = pos_projection(list(rgb_buffer))
                pos_buffer_raw.append(pos_point)
                
                # 2. Filtering dan Estimasi BPM (Hanya dilakukan jika sudah cukup data untuk filter/BPM)
                if len(pos_buffer_raw) >= MIN_BPM_FRAMES:
                    
                    # Lakukan filtering pada data mentah yang sudah pasti panjang (minimal 5 detik)
                    # Catatan: bandpass kini memiliki pengecekan panjang sinyal internal
                    filtered_segment = bandpass(list(pos_buffer_raw), FPS)
                    
                    # Gantikan isi buffer filtered dengan hasil filtering
                    pos_buffer_filtered.clear()
                    pos_buffer_filtered.extend(filtered_segment) 
                    
                    # Hitung BPM dari sinyal yang sudah difilter
                    bpm_val = estimate_bpm(filtered_segment, FPS)
                    
                    # 3. Update Plot FFT
                    n = len(filtered_segment)
                    freqs = rfftfreq(n, 1/FPS)
                    mag = np.abs(rfft(filtered_segment))
                    
                    bpm_freqs = freqs * 60 
                    mask = (bpm_freqs >= 40) & (bpm_freqs <= 240)

                    fft_line.set_data(bpm_freqs[mask], mag[mask])
                    # Update batas Y agar plot FFT terlihat dinamis
                    ax_fft.set_ylim(0, np.max(mag[mask]) * 1.1 + 1e-8) 
                
                # Visualisasi Denyutan ROI
                if len(pos_buffer_filtered) > 0:
                    pulse_val = pos_buffer_filtered[-1]
                    # Normalisasi sinyal (-1 hingga 1) ke skala warna (0 hingga 255)
                    # Semakin tinggi sinyal (puncak), semakin hijau/biru
                    color_r = int(np.clip(127 * (1 - pulse_val), 0, 255))
                    color_g = int(np.clip(255 * (pulse_val + 1) / 2, 0, 255))
                    color_b = 50 
                    box_color = (color_r, color_g, color_b) 

            # Gambar Kotak ROI (dengan warna yang mungkin berdenyut)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)


        # --- Update Display Text ---
        if bpm_val > 40:
            cv2.putText(frame, f"BPM: {bpm_val:.1f}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Collecting signal... ({len(pos_buffer_raw)}/{MIN_BPM_FRAMES})", 
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # --- UPDATE PLOT DATA (Scrolling Effect) ---
    buffer_len = len(rgb_buffer)
    pos_len = len(pos_buffer_filtered)

    # RGB Plot Update
    ax_rgb.set_xlim(max(0, buffer_len - DISPLAY_LEN_FRAMES), buffer_len)
    ax_r.set_data(range(buffer_len), [v[0] for v in rgb_buffer])
    ax_g.set_data(range(buffer_len), [v[1] for v in rgb_buffer])
    ax_b.set_data(range(buffer_len), [v[2] for v in rgb_buffer])

    # POS Plot Update
    ax_pos.set_xlim(max(0, pos_len - DISPLAY_LEN_FRAMES), pos_len)
    pos_line.set_data(range(pos_len), pos_buffer_filtered)
    
    # Update plots
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Show frame
    cv2.imshow("rPPG POS Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
