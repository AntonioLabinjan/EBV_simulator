"""
Simulated Event‑Based Vision + Motion Alarm
-------------------------------------------
This script upgrades your original motion‑detection example to *simulate* a
Dynamic Vision Sensor (DVS) on a normal webcam stream. It produces ON/OFF
"events" for every pixel whose brightness changes beyond threshold values and
feeds those events into a simple motion‑alarm logic.

Key bindings
============
  • **t** – toggle motion‑alarm mode
  • **e** – toggle event (DVS) visualization vs. raw RGB frame
  • **q** – quit the application

Dependencies
============
  pip install opencv‑python imutils numpy  # winsound is built‑in on Windows

Notes
=====
  • For Linux/macOS replace winsound with the *playsound* library or similar.
  • Tune the constants below (POS_TH, NEG_TH, ALARM_EVENT_COUNT, etc.) to match
    your lighting conditions and desired sensitivity.
"""

import threading
import time
import winsound          # On non‑Windows systems swap with playsound/simpleaudio

import cv2
import imutils
import numpy as np

# ───────────────────────────────────────── Camera Setup ─────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)        # Use DirectShow for lower latency on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("[ERROR] Cannot access the webcam.")

# Resize just once to consistent width so math aligns every loop
frame = imutils.resize(frame, width=640)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)

# ───────────────────────────────────────── Parameters ──────────────────────────────────────────
POS_TH = 15              # +ΔI threshold → ON‑event (pixel brighter)
NEG_TH = -15             # −ΔI threshold → OFF‑event (pixel darker)
ALARM_EVENT_COUNT = 25000  # How many events in a single frame before incr. counter
ALARM_PERSIST = 20         # Frames with high activity before siren fires

alarm_active = False       # Are we beeping right now?
alarm_mode = False         # Is motion‑alarm logic enabled?
alarm_counter = 0          # Persists while motion is present
show_events = True         # Toggle between event view & normal RGB

# ───────────────────────────────────────── Feedback: Buzzer ─────────────────────────────────────

def beep_alarm():
    """Background thread: 5 short beeps unless alarm_mode is turned off."""
    global alarm_active
    for _ in range(6):
        if not alarm_mode:
            break
        winsound.Beep(2000, 600)   # freq, duration(ms)
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
    pos_events = (diff > POS_TH).astype(np.uint8) * 255   # ON events → bright green
    neg_events = (diff < NEG_TH).astype(np.uint8) * 255   # OFF events → bright red

    event_img = np.zeros((*pos_events.shape, 3), dtype=np.uint8)
    event_img[..., 1] = pos_events   # Green channel
    event_img[..., 2] = neg_events   # Red channel

    # ----- Display -----
    if show_events:
        cv2.imshow("Event Cam (simulated)", event_img)
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

    prev_gray = gray  # Update reference frame for next iteration

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
