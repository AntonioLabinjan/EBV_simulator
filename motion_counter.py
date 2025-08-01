
import numpy as np
import cv2
import time

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_level = 0

        if self.prev_frame is None:
            self.prev_frame = gray
            return 0

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_level = np.sum(thresh) / 255

        self.prev_frame = gray
        return motion_level

def main():
    cap = cv2.VideoCapture(0)
    detector = MotionDetector()

    count = 0
    threshold = 1000  # Adjust based on camera and lighting
    cooldown = 1.0  # seconds
    last_detected = time.time() - cooldown

    print("Motion counter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion = detector.detect_motion(frame)
        now = time.time()

        if motion > threshold and (now - last_detected) > cooldown:
            count += 1
            last_detected = now
            print(f"Motion: {count}")

        cv2.putText(frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Motion counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
