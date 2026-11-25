"""
Forklift Gate Monitor - ONE program: detection engine + live multi-camera viewer.

    python gate_app.py [--config config.yaml]        (or GateMonitor.exe after build_app.bat)

The detection workers (gate_engine.py) run in background threads of this process and hand
their annotated frames straight to the window below - no second program, no HTTP.

Mouse : click a camera (tile or sidebar row) to maximise it, click again / right-click = all cameras
Keys  : 1-9 maximise camera | 0 or Esc all cameras | S sidebar | M mute PC beep | F fullscreen | Q quit
"""
import os
import sys

# A windowed (no console) exe has no stdout/stderr; libraries that print would crash.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import argparse
import ctypes
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

WIN = "Forklift Gate Monitor"
SIDE_W, TOP_H = 290, 44

BG, PANEL, PANEL2 = (22, 20, 18), (38, 34, 31), (52, 47, 43)
TEXT, DIM = (235, 235, 235), (150, 150, 150)
STATE_COL = {"OPEN": (0, 200, 80), "CLOSED": (0, 40, 230), "LOST": (0, 160, 255),
             "ERROR": (0, 160, 255), "STARTING": (160, 160, 160)}
STATE_TXT = {"OPEN": "CLEAR", "CLOSED": "PERSON IN BAY", "LOST": "NO SIGNAL",
             "ERROR": "ERROR", "STARTING": "STARTING"}


def app_dir():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def fatal(msg):
    try:
        ctypes.windll.user32.MessageBoxW(0, msg, WIN, 0x10)
    except Exception:
        pass
    print(msg, file=sys.stderr)


def put(img, s, org, scale=0.55, col=TEXT, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def fit_text(s, max_px, scale=0.55, thick=1):
    """Shorten s with '..' so it fits max_px wide."""
    while len(s) > 3 and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > max_px:
        s = s[:-3] + ".."
    return s


class UI:
    def __init__(self):
        self.focus = None            # camera id shown alone, or None for the grid
        self.show_side = True
        self.fullscreen = False
        self.tiles, self.rows = [], []   # clickable rects: (x, y, w, h, camera id)
        self.events = deque(maxlen=300)  # (time str, camera name, text, colour)
        self.prev_state = {}

    def toggle_focus(self, cid):
        self.focus = None if self.focus == cid else cid


def on_mouse(event, x, y, _flags, ui):
    if event == cv2.EVENT_RBUTTONDOWN:
        ui.focus = None
    elif event == cv2.EVENT_LBUTTONDOWN:
        for rx, ry, rw, rh, cid in ui.rows + ui.tiles:
            if rx <= x < rx + rw and ry <= y < ry + rh:
                ui.toggle_focus(cid)
                return


def best_grid(n, w, h, aspect=16 / 9):
    """cols x rows that gives the biggest 16:9 tiles in a w x h area."""
    best, best_area = (1, n), -1
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        th = min(w / cols / aspect, h / rows)
        if th * th * aspect > best_area:
            best, best_area = (cols, rows), th * th * aspect
    return best


def draw_tile(canvas, ui, wk, x, y, tw, th, now, flash):
    """One camera tile at (x, y) of size tw x th."""
    pad = 3
    ix, iy, iw, ih = x + pad, y + pad, tw - 2 * pad, th - 2 * pad
    st = wk.state
    col = STATE_COL.get(st, DIM)
    tile = np.full((ih, iw, 3), 12, np.uint8)
    frame, stamp = wk.frame
    live = frame is not None and now - stamp < 3.0 and st not in ("LOST", "ERROR")
    if live:
        fh, fw = frame.shape[:2]
        s = min(iw / fw, ih / fh)
        nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
        ox, oy = (iw - nw) // 2, (ih - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    else:
        msg = "STARTING..." if st == "STARTING" else "NO SIGNAL"
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        put(tile, msg, ((iw - sz[0]) // 2, ih // 2), 0.9, DIM, 2)
    # label bar
    bar = 30
    roi = tile[ih - bar:ih]
    roi[:] = (roi * 0.30).astype(np.uint8)
    label = f"{wk.label}   {STATE_TXT.get(st, st)}"
    put(tile, fit_text(label, iw - 150), (10, ih - 10), 0.6, col if st != "OPEN" else TEXT, 1)
    info = f"{wk.fps:.1f} fps"
    put(tile, info, (iw - 90, ih - 10), 0.5, DIM)
    canvas[iy:iy + ih, ix:ix + iw] = tile
    thick = 2
    if st == "CLOSED":
        thick = 6 if flash else 3
    cv2.rectangle(canvas, (ix, iy), (ix + iw - 1, iy + ih - 1), col, thick)


def draw_top(canvas, W, workers, ui, alarm, flash):
    closed = [w for w in workers if w.state == "CLOSED"]
    bad = [w for w in workers if w.state in ("LOST", "ERROR")]
    if closed:
        bg = (0, 0, 200) if flash else (0, 0, 130)
        msg = "ALARM  -  person in bay: " + ", ".join(w.label for w in closed)
    elif bad:
        bg, msg = (0, 90, 170), "WARNING  -  no signal: " + ", ".join(w.label for w in bad)
    else:
        bg, msg = (60, 52, 46), "All clear"
    cv2.rectangle(canvas, (0, 0), (W, TOP_H), bg, -1)
    put(canvas, "FORKLIFT GATE MONITOR", (14, 29), 0.65, TEXT, 2)
    put(canvas, fit_text(msg, W - 640, 0.6), (260, 29), 0.6, TEXT, 1)
    right = time.strftime("%Y-%m-%d  %H:%M:%S")
    if not alarm.enabled:
        right = "BEEP MUTED   " + right
    sz = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    put(canvas, right, (W - sz - 14, 29), 0.55, TEXT, 1)


def draw_side(canvas, H, ui, workers, now):
    cv2.rectangle(canvas, (0, TOP_H), (SIDE_W, H), PANEL, -1)
    put(canvas, "CAMERAS", (14, TOP_H + 26), 0.5, DIM, 1)
    y = TOP_H + 40
    ui.rows = []
    for i, w in enumerate(workers):
        col = STATE_COL.get(w.state, DIM)
        sel = ui.focus == w.id
        cv2.rectangle(canvas, (6, y), (SIDE_W - 6, y + 52), PANEL2 if sel else PANEL, -1)
        cv2.circle(canvas, (24, y + 26), 7, col, -1)
        put(canvas, fit_text(f"{i + 1}  {w.label}", SIDE_W - 60, 0.6), (42, y + 22), 0.6, TEXT, 1)
        put(canvas, f"{STATE_TXT.get(w.state, w.state)}   {w.persons} person   {w.fps:.1f} fps", (42, y + 42), 0.42, col, 1)
        ui.rows.append((6, y, SIDE_W - 12, 52, w.id))
        y += 58
    put(canvas, "EVENTS", (14, y + 22), 0.5, DIM, 1)
    y += 36
    for t, name, txt, col in reversed(ui.events):
        if y > H - 8:
            break
        put(canvas, t, (14, y), 0.42, DIM, 1)
        put(canvas, fit_text(f"{name}: {txt}", SIDE_W - 90, 0.42), (82, y), 0.42, col, 1)
        y += 20


def render(W, H, ui, workers, alarm, now):
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    flash = int(now * 2.5) % 2 == 0
    for w in workers:                                   # log state changes
        if ui.prev_state.get(w.id) != w.state:
            if w.id in ui.prev_state or w.state != "STARTING":
                ui.events.append((time.strftime("%H:%M:%S"), w.label, STATE_TXT.get(w.state, w.state),
                                  STATE_COL.get(w.state, DIM)))
            ui.prev_state[w.id] = w.state
    side = SIDE_W if (ui.show_side and W > 900) else 0
    shown = [w for w in workers if w.id == ui.focus] or workers
    ui.tiles = []
    ax, ay, aw, ah = side, TOP_H, W - side, H - TOP_H
    cols, rows = best_grid(len(shown), aw, ah)
    tw, th = aw // cols, ah // rows
    for i, w in enumerate(shown):
        r, c = divmod(i, cols)
        x, y = ax + c * tw, ay + r * th
        w.want_until = now + 2.0                        # tell the worker to draw frames for us
        draw_tile(canvas, ui, w, x, y, tw, th, now, flash)
        ui.tiles.append((x, y, tw, th, w.id))
    for w in workers:                                   # hidden cameras are still analysed, just not drawn
        if w not in shown:
            w.want_until = 0.0
    if side:
        draw_side(canvas, H, ui, workers, now)
    else:
        ui.rows = []
    draw_top(canvas, W, workers, ui, alarm, flash)
    return canvas


def splash(W, H, msg, now):
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = BG
    put(canvas, "FORKLIFT GATE MONITOR", (W // 2 - 190, H // 2 - 30), 0.9, TEXT, 2)
    put(canvas, msg + "." * (int(now * 2) % 4), (W // 2 - 190, H // 2 + 15), 0.6, DIM, 1)
    return canvas


def window_size():
    try:
        _x, _y, w, h = cv2.getWindowImageRect(WIN)
        if w >= 320 and h >= 240:
            return w, h
    except Exception:
        pass
    return 1400, 800


def main():
    ap = argparse.ArgumentParser(description="Forklift gate monitor (engine + viewer in one window)")
    ap.add_argument("--config", default=None, help="default: config.yaml next to this program")
    ap.add_argument("--snapshot", default=None, help=argparse.SUPPRESS)        # test aid: save a PNG and exit
    ap.add_argument("--snapshot-after", type=float, default=20.0, help=argparse.SUPPRESS)
    ap.add_argument("--print-hwid", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.print_hwid:
        import hwlock
        hwid = hwlock.get_hwid()
        if getattr(sys, "frozen", False):
            ctypes.windll.user32.MessageBoxW(0, hwid, "Machine ID - send this to Wilson", 0x40)
        else:
            print(hwid)
        return 0
    cfg = Path(args.config) if args.config else app_dir() / "config.yaml"
    if not cfg.exists():
        fatal(f"Config file not found:\n{cfg}\n\nCopy config.example.yaml to config.yaml and edit it.")
        return 1

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 800)
    ui = UI()
    cv2.setMouseCallback(WIN, on_mouse, ui)

    boot = {"ge": None, "alarm": None, "err": None}

    def boot_engine():                       # torch/ultralytics import takes seconds: keep the window alive
        try:
            import gate_engine as ge
            boot["alarm"] = ge.start(cfg, serve_http=False)
            boot["ge"] = ge
        except Exception as e:
            import traceback
            boot["err"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
    threading.Thread(target=boot_engine, daemon=True, name="boot").start()

    while boot["ge"] is None and boot["err"] is None:
        W, H = window_size()
        cv2.imshow(WIN, splash(W, H, "Loading detection engine", time.time()))
        if (cv2.waitKey(50) & 0xFF) in (ord('q'), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 0:
            os._exit(0)
    if boot["err"]:
        fatal(f"Could not start the detection engine:\n{boot['err']}\n\nConfig: {cfg}")
        return 1
    ge, alarm = boot["ge"], boot["alarm"]
    workers = list(ge.WORKERS.values())

    t_start = time.time()
    while True:
        now = time.time()
        W, H = window_size()
        canvas = render(W, H, ui, workers, alarm, now)
        cv2.imshow(WIN, canvas)
        for w in workers:
            alarm.set(w.id, w.beep_hz, w.beep and w.state == "CLOSED")

        if args.snapshot and now - t_start > args.snapshot_after:
            cv2.imwrite(args.snapshot, canvas)
            break
        k = cv2.waitKey(15) & 0xFF
        if k in (ord('q'), ord('Q')) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 0:
            break
        elif k in (ord('0'), 27):
            ui.focus = None
        elif k in (ord('s'), ord('S')):
            ui.show_side = not ui.show_side
        elif k in (ord('m'), ord('M')):
            alarm.enabled = not alarm.enabled
        elif k in (ord('f'), ord('F')):
            ui.fullscreen = not ui.fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if ui.fullscreen else cv2.WINDOW_NORMAL)
        elif ord('1') <= k <= ord('9') and int(chr(k)) <= len(workers):
            ui.focus = workers[int(chr(k)) - 1].id

    cv2.destroyAllWindows()
    ge.stop_all(alarm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
