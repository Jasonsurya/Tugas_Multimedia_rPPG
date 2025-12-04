import time
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import rfft, rfftfreq

CAM_INDEX = 0
WINDOW_SECONDS = 15
MIN_HZ = 0.67
MAX_HZ = 4.0
BPM_SMOOTHING = 3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

def get_right_cheek_roi(landmarks, image_w, image_h):
    indices = [234, 93, 132, 58, 172]
    xs, ys = [], []
    for idx in indices:
        lm = landmarks.landmark[idx]
        xs.append(int(lm.x * image_w))
        ys.append(int(lm.y * image_h))
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x1 = x_min
    y1 = y_min
    x2 = x_max
    y2 = y_max
    offset_x = 10
    offset_y = 10
    adjust_top = 15
    adjust_bottom = 10
    x1 = x1 + offset_x
    x2 = x2 + offset_x
    y1 = y1 + offset_y + adjust_top
    y2 = y2 + offset_y + adjust_bottom
    return x1, y1, x2, y2

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        return
    prev_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    buffer_size = int(WINDOW_SECONDS * fps) + 5
    green_buffer = deque(maxlen=buffer_size)
    time_buffer = deque(maxlen=buffer_size)
    bpm_history = deque(maxlen=10)
    b, a = butter_bandpass(MIN_HZ, MAX_HZ, fs=fps, order=4)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            fs = 1.0 / dt
        else:
            fs = fps
        try:
            b, a = butter_bandpass(MIN_HZ, MAX_HZ, fs=fs, order=4)
        except:
            b, a = butter_bandpass(MIN_HZ, MAX_HZ, fs=fps, order=4)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            x1, y1, x2, y2 = get_right_cheek_roi(face_landmarks, w, h)
            if x2 - x1 > 5 and y2 - y1 > 5:
                roi = frame[y1:y2, x1:x2]
                green_mean = np.mean(roi[:, :, 1])
                green_buffer.append(green_mean)
                time_buffer.append(curr_time)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                if len(green_buffer) > 0:
                    green_buffer.append(green_buffer[-1])
                    time_buffer.append(curr_time)
        else:
            if len(green_buffer) > 0:
                green_buffer.append(green_buffer[-1])
                time_buffer.append(curr_time)
        if len(green_buffer) >= int(3 * fs):
            sig = np.array(green_buffer)
            t = np.array(time_buffer)
            sig_detrended = detrend(sig - np.mean(sig))
            duration = t[-1] - t[0]
            if duration > 0:
                try:
                    sig_filtered = filtfilt(b, a, sig_detrended)
                except:
                    sig_filtered = sig_detrended
                dt_resample = np.median(np.diff(t))
                yf = np.abs(rfft(sig_filtered))
                xf = rfftfreq(len(sig_filtered), d=dt_resample)
                idx_band = np.where((xf >= MIN_HZ) & (xf <= MAX_HZ))[0]
                if len(idx_band) > 0:
                    xf_band = xf[idx_band]
                    yf_band = yf[idx_band]
                    peak_idx = np.argmax(yf_band)
                    peak_freq = xf_band[peak_idx]
                    bpm = peak_freq * 60.0
                    bpm_history.append(bpm)
                    bpm_smooth = np.mean(list(bpm_history)[-BPM_SMOOTHING:])
                else:
                    bpm_smooth = 0.0
                cv2.putText(frame, f"BPM: {bpm_smooth:.1f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fs)}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("rPPG pipi kanan", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
