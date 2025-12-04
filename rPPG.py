import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from collections import deque

# ========================================================
# 1. POS ALGORITHM 
# ========================================================
def pos_projection(rgb_series, fps):
    rgb_series = np.array(rgb_series, dtype=np.float32)
    rgb_centered = rgb_series - np.mean(rgb_series, axis=0)

    C = rgb_centered.T[None, :, :]
    total_frames = C.shape[2]

    window_size = int(1.6 * fps)
    P = np.array([[0, 1, -1],
                  [-2, 1, 1]], dtype=np.float32)

    out = np.zeros(total_frames)
    eps = 1e-8

    for end_idx in range(window_size, total_frames):
        start_idx = end_idx - window_size

        chunk = C[:, :, start_idx:end_idx]
        chunk = chunk / (np.mean(chunk, axis=2, keepdims=True) + eps)

        S = P @ chunk[0]
        X, Y = S[0], S[1]

        scale = np.std(X) / (np.std(Y) + eps)
        H = X - scale * Y
        H = H - np.mean(H)

        out[start_idx:end_idx] += H

    return out


# ========================================================
# 2. BANDPASS FILTER
# ========================================================
def bandpass(signal, fs, low=0.67, high=4.0):
    nyquist = fs * 0.5
    b, a = butter(3, [low/nyquist, high/nyquist], btype="band")
    return filtfilt(b, a, signal)


# ========================================================
# 3. BPM CALCULATOR
# ========================================================
def estimate_bpm(signal, fs):
    n = len(signal)
    freqs = rfftfreq(n, 1/fs)
    mag = np.abs(rfft(signal))

    mask = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(mask):
        return 0

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
# 5. MATPLOTLIB LIVE PLOTS
# ========================================================
plt.ion()
fig, (ax_rgb, ax_pos) = plt.subplots(2, 1, figsize=(10, 8))

(ax_r,) = ax_rgb.plot([], [], "r-", label="R")
(ax_g,) = ax_rgb.plot([], [], "g-", label="G")
(ax_b,) = ax_rgb.plot([], [], "b-", label="B")
ax_rgb.legend()
ax_rgb.set_title("RGB Signal")
ax_rgb.set_ylim(0, 255)

(pos_line,) = ax_pos.plot([], [], "m-", label="POS")
ax_pos.set_title("POS Output")
ax_pos.set_ylim(-1, 1)

# ========================================================
# 6. REALTIME CAMERA
# ========================================================
cap = cv2.VideoCapture(0)
fps = 30

rgb_buffer = []
pos_buffer = deque(maxlen=300)

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

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0]

        cheek_pts = [234, 93, 132, 58, 172]
        xs = [int(lm.landmark[p].x * w) for p in cheek_pts]
        ys = [int(lm.landmark[p].y * h) for p in cheek_pts]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 0), 2)
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            b, g, r = np.mean(roi.reshape(-1, 3), axis=0)
            rgb_buffer.append([r, g, b])

            # UPDATE RGB PLOT
            ax_rgb.set_xlim(0, len(rgb_buffer))
            ax_r.set_data(range(len(rgb_buffer)), [v[0] for v in rgb_buffer])
            ax_g.set_data(range(len(rgb_buffer)), [v[1] for v in rgb_buffer])
            ax_b.set_data(range(len(rgb_buffer)), [v[2] for v in rgb_buffer])

            # ======== POS PROCESSING =========
            if len(rgb_buffer) > int(1.6 * fps):

                pos_sig = pos_projection(rgb_buffer, fps)

                if len(pos_sig) < 30:
                    cv2.putText(frame, "Collecting signal...",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                else:
                    filtered = bandpass(pos_sig, fps)
                    pos_buffer.append(filtered[-1])

                    # Update POS graph
                    pos_line.set_data(range(len(pos_buffer)), pos_buffer)
                    ax_pos.set_xlim(0, len(pos_buffer))

                    bpm_val = estimate_bpm(filtered, fps)
                    cv2.putText(frame, f"BPM: {bpm_val:.1f}",
                                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

    # Update plots
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Show frame
    cv2.imshow("rPPG POS Cheek - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

