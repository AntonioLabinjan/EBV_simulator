import threading
import time
import winsound
import cv2
import imutils
import numpy as np
from collections import deque


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("[ERROR] Cannot access the webcam.")

frame = imutils.resize(frame, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)

# Parameters
POS_TH, NEG_TH = 15, -15
ALARM_EVENT_COUNT, ALARM_PERSIST = 25000, 20
DECAY_RATE = 0.9
ALARM_RESET_TIME = 10  # sec

# State
alarm_active = False
alarm_mode = False
alarm_counter = 0
show_events = True
last_alarm_time = 0
fps_deque = deque(maxlen=10)
event_history = deque(maxlen=150)

event_buffer = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)

# Alarm sound
def beep_alarm():
    global alarm_active, last_alarm_time
    last_alarm_time = time.time()
    for _ in range(6):
        if not alarm_mode:
            break
        winsound.Beep(2000, 500)
        time.sleep(0.1)
    alarm_active = False

# Main loop
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # DVS events
    diff = cv2.subtract(gray, prev_gray).astype(np.int16)
    pos_events = (diff > POS_TH).astype(np.uint8) * 255
    neg_events = (diff < NEG_TH).astype(np.uint8) * 255

    event_img = np.zeros((*pos_events.shape, 3), dtype=np.uint8)
    event_img[..., 1] = pos_events
    event_img[..., 2] = neg_events

    float_events = event_img.astype(np.float32)
    event_buffer = cv2.addWeighted(event_buffer, DECAY_RATE, float_events, 1 - DECAY_RATE, 0)
    accumulated_events = np.clip(event_buffer, 0, 255).astype(np.uint8)

    # Motion detection logic
    event_count = pos_events.sum() // 255 + neg_events.sum() // 255
    event_history.append(event_count)

    if alarm_mode:
        if event_count > ALARM_EVENT_COUNT:
            alarm_counter += 1
        else:
            alarm_counter = max(alarm_counter - 1, 0)

        if alarm_counter > ALARM_PERSIST and not alarm_active:
            alarm_active = True
            threading.Thread(target=beep_alarm, daemon=True).start()

        # Auto reset alarm
        if alarm_active and time.time() - last_alarm_time > ALARM_RESET_TIME:
            alarm_active = False
            alarm_counter = 0

    # FPS calc
    fps = 1.0 / (time.time() - start)
    fps_deque.append(fps)
    avg_fps = sum(fps_deque) / len(fps_deque)

    # Visualization
    display_frame = accumulated_events if show_events else frame.copy()

    # Flash effect
    if alarm_active:
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # Draw info text
    cv2.putText(display_frame, f"Events: {event_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, f"Alarm: {'ON' if alarm_mode else 'OFF'}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    # Draw event graph
    for i in range(1, len(event_history)):
        x1 = 500 + (i - 1)
        x2 = 500 + i
        y1 = 480 - min(event_history[i - 1] // 150, 100)
        y2 = 480 - min(event_history[i] // 150, 100)
        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.imshow("Event Vision", display_frame)
    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        alarm_mode = not alarm_mode
        alarm_counter = 0
        print(f"[INFO] Alarm mode {'ON' if alarm_mode else 'OFF'}")
    elif key == ord('e'):
        show_events = not show_events
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
