"""
Draw the loading-bay area(s) for one camera and save them for the engine.

  python tools/draw_bay.py --source CAM_URL --out bay_cam1.json
  python tools/draw_bay.py --source videos/test_site2.MOV --out bay_test.json [--frame 300]

  LEFT CLICK        add a corner (a live line follows the mouse, the shape fills in after 3 corners)
  RIGHT CLICK       finish this bay (it closes itself)  -> click again to start another bay
  drag a corner     move it (works on finished bays too)
  Z / Backspace     undo last corner      |  D = delete last bay  |  C = clear all  |  Q = quit
Everything is saved automatically to --out each time a bay is finished or edited.
Then in config.yaml on that camera:   zone_file: bay_cam1.json
Draw the FLOOR area where a standing person's feet should trigger the alarm.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2
import numpy as np

WIN = "Draw bay"
GREEN, YELLOW = (0, 230, 80), (0, 230, 255)
GRAB = 14                     # px: how close a click must be to grab an existing corner


def screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1600, 900


def grab(src, frame_no):
    src = os.environ.get(src) or src
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        sys.exit(f"cannot open {src}")
    live = src.isdigit() or src.lower().startswith(("rtsp", "http"))
    if not live and frame_no:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    frame = None
    for _ in range(15 if live else 1):          # live: skip a few frames so we get a fresh one
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        sys.exit("no frame received")
    return frame


class Editor:
    def __init__(self, img, out):
        self.img, self.out = img, out
        self.h, self.w = img.shape[:2]
        self.polys, self.cur = [], []           # polygons are lists of [x, y] in 0..1
        self.mouse, self.drag, self.msg = (0, 0), None, "left click = add corner, right click = finish bay"
        if os.path.exists(out):
            self.polys = [[list(p) for p in poly] for poly in json.load(open(out))]
            self.msg = f"loaded {len(self.polys)} bay(s) from {out}"

    def px(self, p):
        return int(p[0] * self.w), int(p[1] * self.h)

    def save(self):
        json.dump([[[round(x, 5), round(y, 5)] for x, y in p] for p in self.polys], open(self.out, "w"))
        self.msg = f"saved {len(self.polys)} bay(s) -> {self.out}"

    def nearest(self, x, y):
        best, hit = GRAB * max(1.0, self.w / 1500) , None
        for pi, poly in enumerate(self.polys):
            for vi, p in enumerate(poly):
                d = np.hypot(self.px(p)[0] - x, self.px(p)[1] - y)
                if d < best:
                    best, hit = d, (pi, vi)
        return hit

    def finish(self):
        if len(self.cur) >= 3:
            self.polys.append(self.cur)
            self.cur = []
            self.save()
        else:
            self.msg = "need at least 3 corners first"

    def on_mouse(self, ev, x, y, *_):
        self.mouse = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            hit = self.nearest(x, y)
            if hit and not self.cur:
                self.drag = hit
            else:
                self.cur.append([x / self.w, y / self.h])
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            pi, vi = self.drag
            self.polys[pi][vi] = [min(1, max(0, x / self.w)), min(1, max(0, y / self.h))]
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            self.drag = None
            self.save()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.finish()

    def on_key(self, k):
        if k in (13, 10):
            self.finish()
        elif k in (8, ord("z"), ord("Z")) and self.cur:
            self.cur.pop()
        elif k in (ord("d"), ord("D")) and self.polys and not self.cur:
            self.polys.pop()
            self.save()
        elif k in (ord("c"), ord("C")):
            self.polys.clear(); self.cur.clear(); self.save()

    def render(self):
        view = self.img.copy()
        t = max(2, self.w // 700)
        shapes = [(p, GREEN, True) for p in self.polys]
        if self.cur:
            shapes.append((self.cur + [[self.mouse[0] / self.w, self.mouse[1] / self.h]], YELLOW, False))
        for poly, col, closed in shapes:
            pts = np.array([self.px(p) for p in poly], np.int32)
            if len(pts) >= 3:
                ov = view.copy()
                cv2.fillPoly(ov, [pts], col)
                cv2.addWeighted(ov, 0.28, view, 0.72, 0, view)
            cv2.polylines(view, [pts], closed or len(pts) >= 3, col, t, cv2.LINE_AA)
            for pt in pts[:-1] if not closed else pts:
                cv2.circle(view, tuple(int(v) for v in pt), t * 3, col, -1, cv2.LINE_AA)
        bar = max(34, self.h // 24)
        cv2.rectangle(view, (0, 0), (self.w, bar), (0, 0, 0), -1)
        fs = bar / 52
        cv2.putText(view, f"bays: {len(self.polys)}  corners: {len(self.cur)}   {self.msg}", (10, int(bar * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, self.h - bar), (self.w, self.h), (0, 0, 0), -1)
        cv2.putText(view, "L-click add corner | R-click finish bay | drag corner to move | Z undo | D delete bay | C clear | Q quit",
                    (10, self.h - int(bar * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (200, 200, 200), 1, cv2.LINE_AA)
        return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file | rtsp URL | webcam number | env variable name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=0, help="video file: frame number to draw on")
    args = ap.parse_args()
    img = grab(args.source, args.frame)
    sw, sh = screen_size()
    s = min((sw - 80) / img.shape[1], (sh - 140) / img.shape[0], 1.0)
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ed = Editor(img, args.out)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, ed.on_mouse)
    while True:
        cv2.imshow(WIN, ed.render())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 0:
            break
        if k != 255:
            ed.on_key(k)
    cv2.destroyAllWindows()
    if ed.cur:
        print("note: an unfinished bay (fewer than finished corners) was not saved")


if __name__ == "__main__":
    main()
