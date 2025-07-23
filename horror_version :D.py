import threading
import time
import winsound  # On non‑Windows systems swap with playsound/simpleaudio

import cv2
import imutils
import numpy as np

# ───────────────────────────────────────── Camera Setup ─────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for lower latency on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("[ERROR] Cannot access the webcam.")

frame = imutils.resize(frame, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)

# ───────────────────────────────────────── Parameters ──────────────────────────────────────────
POS_TH = 15
NEG_TH = -15
ALARM_EVENT_COUNT = 25000
ALARM_PERSIST = 20

alarm_active = False
alarm_mode = False
alarm_counter = 0
show_events = True

# Event accumulation setup
event_buffer = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
DECAY_RATE = 0.9  # 0 < decay < 1

# ───────────────────────────────────────── Feedback: Buzzer ─────────────────────────────────────
def beep_alarm():
    global alarm_active
    for _ in range(6):
        if not alarm_mode:
            break
        winsound.Beep(2000, 600)
        time.sleep(0.05)
    alarm_active = False

# ───────────────────────────────────────── Main Loop ────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # ----- Simulate DVS events -----
    diff = cv2.subtract(gray, prev_gray).astype(np.int16)
    pos_events = (diff > POS_TH).astype(np.uint8) * 255
    neg_events = (diff < NEG_TH).astype(np.uint8) * 255

    event_img = np.zeros((*pos_events.shape, 3), dtype=np.uint8)
    event_img[..., 1] = pos_events
    event_img[..., 2] = neg_events

    # ----- Accumulate events over time -----
    float_events = event_img.astype(np.float32)
    event_buffer = cv2.addWeighted(event_buffer, DECAY_RATE, float_events, 1 - DECAY_RATE, 0)
    accumulated_events = np.clip(event_buffer, 0, 255).astype(np.uint8)

    # ----- Display -----
    if show_events:
        cv2.imshow("Event Cam (accumulated)", accumulated_events)
    else:
        cv2.imshow("Camera", frame)

    # ----- Motion‑alarm logic based purely on events -----
    if alarm_mode:
        event_count = pos_events.sum() // 255 + neg_events.sum() // 255
        if event_count > ALARM_EVENT_COUNT:
            alarm_counter += 1
        else:
            alarm_counter = max(alarm_counter - 1, 0)

        if alarm_counter > ALARM_PERSIST and not alarm_active:
            alarm_active = True
            threading.Thread(target=beep_alarm, daemon=True).start()

    prev_gray = gray

    # ----- Keyboard Controls -----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        alarm_mode = not alarm_mode
        alarm_counter = 0
        print(f"[INFO] Alarm mode {'ON' if alarm_mode else 'OFF'}")
    elif key == ord('e'):
        show_events = not show_events
    elif key == ord('q'):
        break

# ───────────────────────────────────────── Cleanup ────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
