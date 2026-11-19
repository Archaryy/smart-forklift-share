"""
Desktop viewer for gate_engine.py: tiled live view of all cameras.
    python viewer.py            (engine must be running)
Keys: 1-9 maximise a camera | 0 grid | F fullscreen | Q quit
"""
import argparse
import json
import math
import threading
import time
import urllib.request

import cv2
import numpy as np

COLORS = {"OPEN": (0, 200, 80), "CLOSED": (0, 40, 230), "LOST": (0, 160, 255),
          "ERROR": (0, 160, 255), "STARTING": (160, 160, 160)}


def fetch_status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return None


class Feed(threading.Thread):
    def __init__(self, port, cid):
        super().__init__(daemon=True)
        self.url, self.frame, self.stamp = f"http://127.0.0.1:{port}/cam/{cid}.mjpg", None, 0.0

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.url)
            while cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                self.frame, self.stamp = f, time.time()
            cap.release()
            time.sleep(1.0)


def letterbox(img, w, h):
    out = np.zeros((h, w, 3), np.uint8)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    out[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    WIN = "Forklift Gate - Monitor"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    feeds, status, last_ok, full, focus = {}, [], 0.0, False, None
    last_poll = 0.0

    while True:
        if time.time() - last_poll > 1.0:
            st = fetch_status(args.port)
            last_poll = time.time()
            if st is not None:
                status, last_ok = st, time.time()
                for s in st:
                    if s["id"] not in feeds:
                        feeds[s["id"]] = Feed(args.port, s["id"])
                        feeds[s["id"]].start()
        W, H = 1600, 900
        canvas = np.zeros((H, W, 3), np.uint8)
        shown = [s for s in status if focus is None or s["id"] == focus] or status
        n = max(1, len(shown))
        cols = 1 if n == 1 else 2
        rows = math.ceil(n / cols)
        tw, th = W // cols, H // rows
        for i, s in enumerate(shown):
            f = feeds.get(s["id"])
            live = f is not None and f.frame is not None and time.time() - f.stamp < 3
            tile = letterbox(f.frame, tw, th) if live else np.zeros((th, tw, 3), np.uint8)
            if not live:
                cv2.putText(tile, "NO SIGNAL", (tw // 2 - 90, th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2)
            col = COLORS.get(s["state"], (160, 160, 160))
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), col, 4)
            cv2.putText(tile, f"{s['name']}  [{s['state']}]  {s['fps']} fps", (12, th - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            r, c = divmod(i, cols)
            canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        if time.time() - last_ok > 3:
            cv2.rectangle(canvas, (0, 0), (W, 50), (0, 0, 160), -1)
            cv2.putText(canvas, "ENGINE NOT RUNNING / NOT REACHABLE", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(WIN, canvas)

        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('f'):
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        elif k == ord('0'):
            focus = None
        elif ord('1') <= k <= ord('9') and int(chr(k)) <= len(status):
            focus = status[int(chr(k)) - 1]["id"]
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
