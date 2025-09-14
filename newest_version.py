import argparse
import time
import math
import csv
from collections import deque

import cv2
import numpy as np


class DVSSimulator:
    def __init__(self, shape, thresh=0.2, refractory_ms=5, max_events_per_frame=100000):
        self.h, self.w = shape
        self.thresh = thresh  # log intensity threshold
        self.ref_ms = refractory_ms
        self.last_log = None
        self.last_event_ts = np.zeros((self.h, self.w), dtype=np.float32) - 1e9
        self.max_events_per_frame = max_events_per_frame

    def init_state(self, frame_gray):
        frame = frame_gray.astype(np.float32) / 255.0
        frame[frame <= 1e-6] = 1e-6
        self.last_log = np.log(frame)

    def step(self, frame_gray, t_ms):
        """Process a single grayscale frame arriving at time t_ms (milliseconds).
        Returns list of events: (x, y, t_ms, polarity)
        Polarity: +1 for ON (intensity increase), -1 for OFF (decrease)
        """
        if self.last_log is None:
            self.init_state(frame_gray)
            return []

        frame = frame_gray.astype(np.float32) / 255.0
        frame[frame <= 1e-6] = 1e-6
        logI = np.log(frame)
        dlog = logI - self.last_log

        # Events where absolute change exceeds thresh
        on_mask = dlog >= self.thresh
        off_mask = dlog <= -self.thresh
        mask = on_mask | off_mask

        # Apply refractory per-pixel
        ts_delta = t_ms - self.last_event_ts
        can_fire = ts_delta >= self.ref_ms
        fire_mask = mask & can_fire

        ys, xs = np.nonzero(fire_mask)
        if len(xs) > self.max_events_per_frame:
            idxs = np.random.choice(len(xs), self.max_events_per_frame, replace=False)
            xs = xs[idxs]
            ys = ys[idxs]

        events = []
        for x, y in zip(xs, ys):
            pol = 1 if on_mask[y, x] else -1
            events.append((x, y, t_ms, pol))
            # Update last_log by stepping it by threshold in the direction of change
            self.last_log[y, x] += self.thresh * (1 if pol > 0 else -1)
            self.last_event_ts[y, x] = t_ms

        # A small global drift update to avoid accumulation of rounding errors (optional)
        self.last_log *= 1.0

        return events


def synthetic_frame(t_sec, shape=(240, 320)):
    h, w = shape
    frame = np.zeros((h, w), dtype=np.uint8)
    # moving circle
    cx = int((0.5 + 0.4 * math.sin(2.0 * math.pi * 0.5 * t_sec)) * w)
    cy = int((0.5 + 0.4 * math.cos(2.0 * math.pi * 0.33 * t_sec)) * h)
    r = int(20 + 10 * math.sin(2.0 * math.pi * 0.8 * t_sec))
    cv2.circle(frame, (cx, cy), max(2, r), 255, -1)
    # moving rectangle
    rx = int((0.5 + 0.4 * math.cos(2.0 * math.pi * 0.6 * t_sec)) * w)
    ry = int((0.7 + 0.2 * math.sin(2.0 * math.pi * 0.4 * t_sec)) * h)
    rw, rh = 40, 24
    cv2.rectangle(frame, (rx - rw//2, ry - rh//2), (rx + rw//2, ry + rh//2), 180, -1)
    # textured background noise
    noise = (20 * (0.5 + 0.5 * np.random.randn(h, w))).clip(0, 40).astype(np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def events_to_image(events, shape, decay_ms=80, timestamp_ms=0):
    h, w = shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # We'll draw recent events with colors: ON=green, OFF=red. Older events fade.
    for (x, y, t_ms, p) in events:
        age = timestamp_ms - t_ms
        if age < 0:
            age = 0
        alpha = max(0.0, 1.0 - age / decay_ms)
        if p > 0:
            img[y, x, 1] = min(255, img[y, x, 1] + int(255 * alpha))
        else:
            img[y, x, 2] = min(255, img[y, x, 2] + int(255 * alpha))
    # slight blur for visibility
    img = cv2.blur(img, (3, 3))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "video", "synthetic"], default="synthetic")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--thresh", type=float, default=0.18)
    parser.add_argument("--ref_ms", type=float, default=5.0)
    parser.add_argument("--save_csv", type=str, default=None)
    args = parser.parse_args()

    H, W = args.height, args.width
    sim = DVSSimulator((H, W), thresh=args.thresh, refractory_ms=args.ref_ms)

    if args.mode == "webcam":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    elif args.mode == "video":
        if not args.video:
            raise SystemExit("--video path required for video mode")
        cap = cv2.VideoCapture(args.video)
    else:
        cap = None

    out_events = []
    if args.save_csv:
        csv_file = open(args.save_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y", "t_ms", "polarity"])
    else:
        csv_writer = None

    vis_event_buffer = deque(maxlen=1)  # store events from last frame for visualization

    t0 = time.time()
    frame_idx = 0
    try:
        while True:
            t_now = time.time()
            if args.mode in ("webcam", "video"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (W, H))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t_ms = (time.time() - t0) * 1000.0
            else:
                t_sec = time.time() - t0
                gray = synthetic_frame(t_sec, (H, W))
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                t_ms = (time.time() - t0) * 1000.0

            events = sim.step(gray, t_ms)
            if csv_writer:
                for e in events:
                    csv_writer.writerow(e)
            vis_event_buffer.clear()
            vis_event_buffer.append((events, t_ms))

            # Visualization build
            vis = frame.copy()
            # overlay raw events
            if vis_event_buffer:
                evs, ts = vis_event_buffer[-1]
                ev_img = events_to_image(evs, (H, W), decay_ms=120, timestamp_ms=ts)
                # blend with original
                vis = cv2.addWeighted(vis, 0.6, ev_img, 0.8, 0)

            # small HUD
            cv2.putText(vis, f"Events: {len(events)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(vis, f"Thresh: {sim.thresh:.3f}", (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(vis, f"Ref(ms): {sim.ref_ms:.1f}", (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("DVS Sim", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):
                break
            elif key == ord('=') or key == ord('+'):
                sim.thresh = max(0.02, sim.thresh - 0.01)
            elif key == ord('-'):
                sim.thresh += 0.01
            elif key == ord(']'):
                sim.ref_ms = min(1000.0, sim.ref_ms + 1.0)
            elif key == ord('['):
                sim.ref_ms = max(0.0, sim.ref_ms - 1.0)

            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if args.save_csv:
            csv_file.close()


if __name__ == '__main__':
    main()
