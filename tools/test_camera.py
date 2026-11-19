"""
Check that the Hikvision camera stream opens, and measure what the PC receives.

    PowerShell:   $env:CAM_URL = "rtsp://USER:PASSWORD@CAMERA_IP:554/Streaming/Channels/102"
                  python test_camera.py

Saves camera_snapshot.jpg (open it to confirm the picture is the right camera and angle).
The URL is never printed, because it contains the password.
"""
import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import sys
import time

import cv2

url = os.environ.get("CAM_URL") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not url:
    sys.exit("Set the CAM_URL environment variable first (see the docstring).")

print("Opening stream (can take up to ~10 s)...")
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    sys.exit(
        "FAILED to open the stream. Check, in order:\n"
        "  1. Can VLC open the same URL?  2. Camera IP reachable (ping)?\n"
        "  3. Username/password right (special characters must be URL-encoded: @ -> %40)?\n"
        "  4. Channel number: .../Channels/101 (main) or 102 (sub)?\n"
        "  5. Camera's RTSP port is 554 and RTSP is enabled in its settings?"
    )

ok, frame = cap.read()
if not ok:
    sys.exit("Stream opened but no frames arrived (try the sub stream, /102, or set H.264 in the camera).")

h, w = frame.shape[:2]
print(f"Resolution: {w}x{h}")
cv2.imwrite("camera_snapshot.jpg", frame)

n, t0 = 0, time.time()
while time.time() - t0 < 5:
    ok, _ = cap.read()
    if not ok:
        print("Dropped frame / stream interrupted")
        break
    n += 1
print(f"Received {n} frames in 5 s  ->  about {n / 5:.1f} FPS")
print("Saved camera_snapshot.jpg")
cap.release()
