"""
Warehouse Forklift Safety System
─────────────────────────────────
Features:
  • Forklift detection with false-positive suppression
  • ByteTrack tracker – objects must persist MIN_TRACK_AGE frames to count
    (eliminates 2-3 frame truck/trailer false-positive flickers)
  • Person detection + forklift/operator association
  • Loading-bay polygon zone (interactive drawing, persisted to JSON)
  • Gate-control logic – CLOSES when un-paired persons enter the bay

Controls:
  Q           – quit
  Z           – freeze frame, draw zone (click corners, ENTER=confirm, ESC=cancel)
  R           – reset / clear zone
  C           – cycle to next video
  +/-         – raise / lower confidence threshold
  SPACE       – pause / resume
"""

import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import threading
import time
import requests
from requests.auth import HTTPDigestAuth
from urllib.parse import urlparse, unquote
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_PATH  = "runs/forklift/v4_corrections/weights/best.pt"
VIDEO_PATH  = "videos/test_site3.mov"

CLASS_FORKLIFT = 0
CLASS_PERSON   = 1
CLASS_TRUCK    = 2   # drawn for debugging only, never triggers the gate

# Old workaround for truck-vs-forklift confusion. The model now has a truck
# class, so this is off; turn on only if trucks still leak through as forklifts.
USE_SHAPE_FILTER = False

CONFIDENCE     = 0.50
IMGSZ          = 960   # match finetune.py; lower to 640 if the PC is too slow

# Per-class confidence. A missed pedestrian is worse than a false alarm, so the
# person threshold is lower. +/- keys change the person threshold live.
CONF_FORKLIFT  = 0.50
CONF_PERSON    = 0.30
CONF_TRUCK     = 0.50

# Run the detector only on the loading-bay area (zone bounding box + margin).
# People appear larger -> far/small pedestrians are detected better. Key B toggles.
USE_BAY_CROP   = True
BAY_CROP_PAD   = 0.15   # margin around the zone, as a fraction of its size

# Tracker settings
TRACKER        = "bytetrack.yaml"  # or "botsort.yaml"
MIN_TRACK_AGE  = 4   # frames a track must exist before it's acted on

# Forklift false-positive guards
MAX_ASPECT_RATIO    = 1.6    # w/h above this → likely truck/trailer
MAX_FRAME_AREA_FRAC = 0.30   # fraction of frame area above this → too big
MIN_BOX_WIDTH_PX    = 40
MIN_BOX_HEIGHT_PX   = 40

# Operator-association padding around forklift box
ASSOC_PAD_X = 30
ASSOC_PAD_Y = 50

# Display
MAX_WIDTH  = 1280
MAX_HEIGHT = 720

# Zone persistence
ZONE_FILE = "loading_bay_zone.json"

# Gate: how many consecutive frames must agree before gate state flips
SHOW_TRUCK_BOX = False   # draw truck boxes? (T key toggles; trucks are still detected)
SEEK_SECONDS   = 5.0     # arrow keys / J / L jump this many seconds in video files

# ── Automatic bay (derived from the detected truck) ─────────────────
# The bay is a rectangle attached to the truck's REAR side, in image space.
#   rear    : which side of the truck box is the rear: bottom|top|left|right
#   depth   : how far the bay extends past the rear (fraction of truck box size)
#   lateral : extra width each side of the truck (fraction of truck box size)
#   overlap : how far the bay reaches back INTO the truck box (fraction)
# Live keys: S = cycle rear side, [ ] = depth -/+, A = auto/manual mode.
BAY_CFG_FILE   = "bay_config.json"
BAY_CFG        = {"rear": "bottom", "depth": 0.60, "lateral": 0.15, "overlap": 0.05}
BAY_HOLD_FRAMES = 90    # keep a truck's bay this long after the truck is lost
BAY_SMOOTH      = 0.2   # box smoothing (0 = frozen, 1 = no smoothing)
REAR_SIDES      = ["bottom", "top", "left", "right"]

# Gate timing: CLOSE fast (safety), OPEN slowly (avoid chatter).
# Frame counts, so the real delay depends on the video / webcam FPS.
GATE_CLOSE_FRAMES = 5
GATE_OPEN_FRAMES  = 30

# ─────────────────────────────────────────────
# COLOURS (BGR)
# ─────────────────────────────────────────────
COL_FORKLIFT_SOLO   = (0, 200, 255)
COL_FORKLIFT_PAIRED = (0, 255,  80)
COL_PERSON_FREE     = (0,  60, 255)
COL_PERSON_PAIRED   = (200, 200, 200)
COL_ZONE_SAFE       = (0, 255,  80)
COL_ZONE_DANGER     = (0,  60, 255)
COL_GATE_OPEN       = (0, 220,  80)
COL_GATE_CLOSED     = (0,  40, 220)
COL_CROSSHAIR       = (0, 255, 255)
COL_TRUCK           = (150, 150, 150)

# ─────────────────────────────────────────────
# STATE (shared with mouse callback)
# ─────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.display_scale  = 1.0   # set each frame; used to convert mouse → orig coords
        self.drawing_zone   = False
        self.zone_pts_disp  = []    # points in DISPLAY coordinates (for live preview)
        self.mouse_x        = 0
        self.mouse_y        = 0

# Track age counter: track_id → number of consecutive frames seen
track_age: dict[int, int] = {}

state = AppState()


def mouse_callback(event, x, y, flags, param):
    state.mouse_x = x
    state.mouse_y = y
    if state.drawing_zone and event == cv2.EVENT_LBUTTONDOWN:
        state.zone_pts_disp.append((x, y))


# ─────────────────────────────────────────────
# COORDINATE HELPERS
# ─────────────────────────────────────────────

def disp_to_orig(pts_disp, scale):
    """Convert a list of (x,y) in display space → original frame space."""
    s = max(scale, 1e-6)
    return [(int(x / s), int(y / s)) for x, y in pts_disp]


def orig_to_disp(pts_orig, scale):
    """Convert a list of (x,y) in original frame space → display space."""
    return [(int(x * scale), int(y * scale)) for x, y in pts_orig]


# ─────────────────────────────────────────────
# ZONE IO
# ─────────────────────────────────────────────

def load_zone(path):
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text())
        if data:
            return np.array(data, dtype=np.int32)
    return None


def save_zone(path, poly_orig):
    Path(path).write_text(json.dumps(poly_orig.tolist()))


# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────

def point_in_poly(pt, poly):
    if poly is None or len(poly) < 3:
        return False
    return cv2.pointPolygonTest(
        poly, (float(pt[0]), float(pt[1])), False
    ) >= 0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def foot_point(box):
    """Bottom-centre of a person box = where they stand on the floor.
    People inside a truck or on a pallet have their feet above/outside the
    gray loading-bay floor area, so they are not counted."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def bay_crop_rect(zone, frame_w, frame_h, pad=BAY_CROP_PAD):
    """Bounding box of the zone polygon plus margin, clipped to the frame."""
    xs, ys = zone[:, 0], zone[:, 1]
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(frame_w, x1 + px), min(frame_h, y1 + py)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def load_bay_cfg():
    cfg = dict(BAY_CFG)
    p = Path(BAY_CFG_FILE)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text()))
        except Exception:
            pass
    return cfg


def save_bay_cfg(cfg):
    Path(BAY_CFG_FILE).write_text(json.dumps(cfg, indent=2))


def bay_polygon(box, cfg, fw, fh):
    """Rectangle attached to the truck's rear side (image space, clipped)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    lat, d, ov = cfg["lateral"], cfg["depth"], cfg["overlap"]
    rear = cfg["rear"]
    if rear == "bottom":
        r = (x1 - lat * w, y2 - ov * h, x2 + lat * w, y2 + d * h)
    elif rear == "top":
        r = (x1 - lat * w, y1 - d * h, x2 + lat * w, y1 + ov * h)
    elif rear == "right":
        r = (x2 - ov * w, y1 - lat * h, x2 + d * w, y2 + lat * h)
    else:  # left
        r = (x1 - d * w, y1 - lat * h, x1 + ov * w, y2 + lat * h)
    rx1, ry1 = max(0, r[0]), max(0, r[1])
    rx2, ry2 = min(fw - 1, r[2]), min(fh - 1, r[3])
    return np.array([[rx1, ry1], [rx2, ry1], [rx2, ry2], [rx1, ry2]], dtype=np.int32)


class TruckBays:
    """Smooths truck boxes per track id and keeps a truck for a while after it
    is lost (a person standing in front of it can hide it for a few frames)."""

    def __init__(self):
        self.t = {}   # track id -> {"box": [x1,y1,x2,y2], "seen": frame_idx}

    def update(self, trucks, frame_idx):
        for tid, box in trucks:
            if tid < 0:
                continue
            if tid in self.t:
                old = self.t[tid]["box"]
                box = [o + BAY_SMOOTH * (n - o) for o, n in zip(old, box)]
            self.t[tid] = {"box": list(box), "seen": frame_idx}
        for tid in [k for k, v in self.t.items() if frame_idx - v["seen"] > BAY_HOLD_FRAMES]:
            del self.t[tid]

    def polys(self, cfg, fw, fh):
        return [bay_polygon(v["box"], cfg, fw, fh) for v in self.t.values()]

    def clear(self):
        self.t.clear()


def boxes_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


# ─────────────────────────────────────────────
# FILTER: reject truck/trailer-like boxes
# ─────────────────────────────────────────────

def is_valid_forklift(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if not USE_SHAPE_FILTER:
        return True
    if w < MIN_BOX_WIDTH_PX or h < MIN_BOX_HEIGHT_PX:
        return False
    if (w / max(h, 1)) > MAX_ASPECT_RATIO:
        return False   # too wide → truck/trailer
    if (w * h) / max(frame_w * frame_h, 1) > MAX_FRAME_AREA_FRAC:
        return False   # too large → truck
    return True


# ─────────────────────────────────────────────
# ASSOCIATION
# ─────────────────────────────────────────────

def associate(forklift_boxes, person_boxes):
    remaining = list(person_boxes)
    units = []
    for fb in forklift_boxes:
        fx1, fy1, fx2, fy2 = fb
        search = [fx1 - ASSOC_PAD_X, fy1 - ASSOC_PAD_Y,
                  fx2 + ASSOC_PAD_X, fy2 + ASSOC_PAD_Y]
        best_idx = -1
        for idx, pb in enumerate(remaining):
            if boxes_overlap(search, pb):
                best_idx = idx
                break
        if best_idx >= 0:
            op = remaining.pop(best_idx)
            units.append({"box": fb, "has_operator": True,  "operator_box": op})
        else:
            units.append({"box": fb, "has_operator": False, "operator_box": None})
    return units, remaining


# ─────────────────────────────────────────────
# GATE LOGIC
# ─────────────────────────────────────────────

RELAY = None   # GateRelay instance when --relay is given (see main)


def on_gate_change(new_state):
    """Hook for the real gate. The relay itself is driven per frame in main()
    (it also needs a warm-up and a heartbeat), so nothing to do here."""
    pass


class Alarm:
    """Beeps on the PC speakers while a person is in the bay (gate CLOSED).
    Runs in a thread so it never slows detection. M key mutes."""

    def __init__(self, enabled=True):
        self.enabled, self.on = enabled, False
        try:
            import winsound
            self._beep = winsound.Beep
        except ImportError:                       # non-Windows: terminal bell
            self._beep = lambda f, d: print("\a", end="", flush=True) or time.sleep(d / 1000)
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, on):
        self.on = bool(on) and self.enabled

    def _loop(self):
        while True:
            if self.on:
                self._beep(1200, 250)
                time.sleep(0.15)
            else:
                time.sleep(0.05)


class CameraRelay:
    """Drives the camera's alarm output over ISAPI (digest auth).
    HIGH = person in bay, LOW = bay clear (use invert=True to flip).
    The wanted state is re-sent every second, and sent immediately on change."""

    def __init__(self, host, user, pw, output=1, invert=False, port=80):
        self.url = f"http://{host}:{port}/ISAPI/System/IO/outputs/{output}/trigger"
        self.invert = invert
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, pw)
        self.want, self.sent, self.last, self.err_shown = None, None, 0.0, False
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, person_in_bay):
        self.want = "high" if (bool(person_in_bay) != self.invert) else "low"

    def _loop(self):
        while self._run:
            if self.want and (self.want != self.sent or time.time() - self.last > 1.0):
                try:
                    body = ('<IOPortData version="2.0" xmlns="http://www.isapi.org/ver20/XMLSchema">'
                            f'<outputState>{self.want}</outputState></IOPortData>')
                    r = self.session.put(self.url, data=body, timeout=2,
                                         headers={"Content-Type": "application/xml"})
                    r.raise_for_status()
                    if self.want != self.sent:
                        print(f"[RELAY] output -> {self.want.upper()}")
                    self.sent, self.err_shown = self.want, False
                except Exception as e:
                    if not self.err_shown:
                        print(f"[RELAY] FAILED to set output: {str(e)[:120]}")
                        self.err_shown = True
                    self.sent = None
                self.last = time.time()
            time.sleep(0.05)

    def close(self):
        # On exit treat it as "person in bay" so the relay ends in the alarm state.
        self.set(True)
        time.sleep(0.5)
        self._run = False


def on_camera_lost():
    """Called once when a live camera stops delivering frames.
    Fail-safe: relay OFF = gate closed."""
    if RELAY is not None:
        RELAY.fail_closed()


class GateController:
    def __init__(self, close_frames=GATE_CLOSE_FRAMES, open_frames=GATE_OPEN_FRAMES):
        self.history  = []
        self.close_frames = close_frames
        self.open_frames  = open_frames
        self.state    = "OPEN"

    def update(self, free_persons, zones):
        if not zones:
            raw = "OPEN"
            msg = "✓  GATE OPEN – no bay (no truck detected / no zone)"
        else:
            in_bay = [p for p in free_persons
                      if any(point_in_poly(foot_point(p), z) for z in zones)]
            if in_bay:
                n   = len(in_bay)
                raw = "CLOSED"
                msg = f"⚠  GATE CLOSED – {n} person{'s' if n > 1 else ''} in bay"
            else:
                raw = "OPEN"
                msg = "✓  GATE OPEN – bay clear"

        self.history.append(raw)
        keep = max(self.close_frames, self.open_frames)
        if len(self.history) > keep:
            self.history.pop(0)

        h = self.history
        if len(h) >= self.close_frames and all(g == "CLOSED" for g in h[-self.close_frames:]):
            self.state = "CLOSED"
        elif len(h) >= self.open_frames and all(g == "OPEN" for g in h[-self.open_frames:]):
            self.state = "OPEN"
        # else: keep current state during transition

        if self.state != getattr(self, '_last', 'OPEN'):
            print(f"[GATE] -> {self.state}")
            on_gate_change(self.state)
        self._last = self.state

        return self.state, msg


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────

def draw_zone_on_display(disp, zone_orig, gate_state, scale):
    if zone_orig is None or len(zone_orig) < 3:
        return
    col     = COL_ZONE_SAFE if gate_state == "OPEN" else COL_ZONE_DANGER
    pts_d   = np.array(orig_to_disp(zone_orig.tolist(), scale), dtype=np.int32)
    overlay = disp.copy()
    cv2.fillPoly(overlay, [pts_d], col)
    cv2.addWeighted(overlay, 0.15, disp, 0.85, 0, disp)
    cv2.polylines(disp, [pts_d], True, col, 2)


def draw_zone_preview_on_display(disp, pts_disp):
    """Live preview while user is clicking zone corners (all in display coords)."""
    if not pts_disp:
        return
    for pt in pts_disp:
        cv2.circle(disp, pt, 6, COL_CROSSHAIR, -1)
    for i in range(1, len(pts_disp)):
        cv2.line(disp, pts_disp[i - 1], pts_disp[i], COL_CROSSHAIR, 2)
    # Close the polygon preview
    if len(pts_disp) >= 3:
        cv2.line(disp, pts_disp[-1], pts_disp[0], COL_CROSSHAIR, 1)
    # Crosshair at cursor
    mx, my = state.mouse_x, state.mouse_y
    cv2.line(disp, (mx - 12, my), (mx + 12, my), COL_CROSSHAIR, 1)
    cv2.line(disp, (mx, my - 12), (mx, my + 12), COL_CROSSHAIR, 1)
    # Instructions
    n = len(pts_disp)
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(disp,
                f"ZONE DRAWING  ({n} point{'s' if n != 1 else ''})  "
                f"–  click corners  |  ENTER = confirm  |  ESC = cancel",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_CROSSHAIR, 2)


def draw_forklift_unit(disp, unit, conf, track_id, scale):
    x1, y1, x2, y2 = [int(v * scale) for v in unit["box"]]
    col   = COL_FORKLIFT_PAIRED if unit["has_operator"] else COL_FORKLIFT_SOLO
    op_txt = "Forklift+Op" if unit["has_operator"] else "Forklift ⚠no op"
    label = f"{op_txt}  id:{track_id}  {conf:.2f}"
    cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
    tw = len(label) * 9
    cv2.rectangle(disp, (x1, y1 - 22), (x1 + tw, y1), col, -1)
    cv2.putText(disp, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    if unit["operator_box"] is not None:
        px1, py1, px2, py2 = [int(v * scale) for v in unit["operator_box"]]
        cv2.rectangle(disp, (px1, py1), (px2, py2), COL_PERSON_PAIRED, 1)


def draw_truck(disp, box, conf, scale):
    x1, y1, x2, y2 = [int(v * scale) for v in box]
    cv2.rectangle(disp, (x1, y1), (x2, y2), COL_TRUCK, 1)
    cv2.putText(disp, f"truck {conf:.2f}", (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TRUCK, 1)


def draw_free_person(disp, box, scale, conf=None):
    x1, y1, x2, y2 = [int(v * scale) for v in box]
    cv2.rectangle(disp, (x1, y1), (x2, y2), COL_PERSON_FREE, 2)
    cv2.circle(disp, ((x1 + x2) // 2, y2), 5, COL_PERSON_FREE, -1)  # foot point
    label = "PERSON" if conf is None else f"PERSON {conf:.2f}"
    cv2.rectangle(disp, (x1, y1 - 22), (x1 + 12 * len(label), y1), COL_PERSON_FREE, -1)
    cv2.putText(disp, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_hud(disp, forklift_units, free_persons, conf, video_name):
    cv2.rectangle(disp, (0, 0), (340, 70), (0, 0, 0), -1)
    cv2.putText(disp, f"Forklifts : {len(forklift_units)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(disp, f"Free persons: {len(free_persons)}", (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(disp, f"Person conf: {conf:.2f}  |  {Path(video_name).name}", (8, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def draw_gate_banner(disp, gate_state, message, conf):
    h, w = disp.shape[:2]
    col  = COL_GATE_OPEN if gate_state == "OPEN" else COL_GATE_CLOSED
    cv2.rectangle(disp, (0, h - 56), (w, h), (20, 20, 20), -1)
    cv2.rectangle(disp, (0, h - 56), (w, h), col, 3)
    cv2.putText(disp, message, (12, h - 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 2)
    cv2.putText(disp, "+/- conf  A auto  S rear  [ ] depth  , . width  T trucks  J/L seek  Z zone   R reset   C next video   SPACE pause   Q quit",
                (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)


# ─────────────────────────────────────────────
# LIVE CAMERA (webcam / RTSP / Hikvision)
# ─────────────────────────────────────────────

class LiveStream:
    """Reads a live stream in a background thread and always hands out the
    NEWEST frame, so inference never works on stale buffered frames (RTSP lag).
    Reconnects automatically if the stream drops."""

    def __init__(self, src, is_webcam):
        self.src, self.is_webcam = src, is_webcam
        self._lock = threading.Lock()
        self._frame, self._stamp, self._seen = None, 0, 0
        self._run = True
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError("Cannot open camera/stream - check URL, user, password, network")
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _open(self):
        if self.is_webcam:
            cap = cv2.VideoCapture(int(self.src), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        while self._run:
            ok, frame = self._cap.read()
            if not ok:
                self._cap.release()
                time.sleep(2.0)
                self._cap = self._open()
                continue
            with self._lock:
                self._frame, self._stamp = frame, self._stamp + 1

    def isOpened(self):
        return self._cap.isOpened()

    def read(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._stamp != self._seen:
                    self._seen = self._stamp
                    return True, self._frame
            time.sleep(0.003)
        return False, None

    def release(self):
        self._run = False
        self._t.join(timeout=2)
        self._cap.release()


# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="webcam index (e.g. 0), a video file, or a folder of videos "
                         "(default: ./videos)")
    ap.add_argument("--camera", action="store_true",
                    help="use the RTSP URL stored in the CAM_URL environment variable "
                         "(keeps the camera password out of the command line)")
    ap.add_argument("--seek", type=float, default=SEEK_SECONDS,
                    help="seconds to jump with arrow keys / J / L (video files)")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--conf-person", "--conf", dest="conf_person", type=float, default=CONF_PERSON)
    ap.add_argument("--conf-forklift", type=float, default=CONF_FORKLIFT)
    ap.add_argument("--conf-truck", type=float, default=CONF_TRUCK)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--no-auto-bay", action="store_true", help="use the hand-drawn zone (Z) instead of the truck-based bay")
    ap.add_argument("--no-alarm", action="store_true", help="disable the PC beep when a person is in the bay")
    ap.add_argument("--relay", action="store_true",
                    help="drive the camera alarm output: HIGH = person in bay, LOW = clear "
                         "(credentials from CAM_URL with --camera, or CAM_IP/CAM_USER/CAM_PASS)")
    ap.add_argument("--relay-output", type=int, default=1)
    ap.add_argument("--relay-port", type=int, default=80, help="camera HTTP port")
    ap.add_argument("--relay-invert", action="store_true", help="swap HIGH/LOW")
    ap.add_argument("--rear", choices=REAR_SIDES, default=None, help="which side of the truck box is the rear")
    ap.add_argument("--no-crop", action="store_true", help="always infer on the full frame")
    ap.add_argument("--relay", action="store_true",
                    help="drive the camera alarm-output relay (needs CAM_URL for host/credentials). "
                         "Relay ON only while the bay is clear and the system is healthy.")
    ap.add_argument("--relay-mode", choices=["alarm", "permit"], default="alarm",
                    help="alarm (default): relay ON = person in bay (or system not healthy), OFF = bay clear. "
                         "permit: relay ON = bay clear / gate may open, OFF = closed or any failure.")
    ap.add_argument("--save", default=None, help="write annotated output to this .mp4")
    return ap.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global RELAY
    args   = parse_args()
    if args.relay and not os.environ.get("CAM_URL"):
        raise SystemExit("--relay needs CAM_URL (camera host + credentials).")
    cam_url = os.environ.get("CAM_URL") if args.camera else None
    if args.camera and not cam_url:
        raise SystemExit("--camera needs the CAM_URL environment variable (the RTSP URL).")
    src_str = cam_url or args.source
    webcam  = src_str is not None and src_str.isdigit()
    rtsp    = src_str is not None and src_str.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    live    = webcam or rtsp
    exts   = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    if live:
        # never put the URL (it contains the password) in a label
        all_videos = [Path("webcam" if webcam else "ip-camera")]
    elif args.source and Path(args.source).is_file():
        all_videos = [Path(args.source)]
    else:
        video_dir  = Path(args.source) if args.source else Path("videos")
        all_videos = sorted(v for v in video_dir.iterdir() if v.suffix.lower() in exts)
        if not all_videos:
            all_videos = [Path(VIDEO_PATH)]

    video_idx = 0

    model     = YOLO(args.model)
    zone_orig = load_zone(ZONE_FILE)   # polygon in ORIGINAL frame coords
    gate      = GateController()
    if args.relay:
        from gate_relay import GateRelay
        RELAY = GateRelay(invert=(args.relay_mode == "alarm"))            # forces the relay OFF (gate closed) right now
    relay_open = False                 # what we last told the relay
    warm       = 0                     # clean frames since start / camera loss
    conf      = args.conf_person      # live-adjustable with +/-
    bay_crop  = USE_BAY_CROP and not args.no_crop
    auto_bay  = not args.no_auto_bay
    bay_cfg   = load_bay_cfg()
    if args.rear:
        bay_cfg["rear"] = args.rear
    bays      = TruckBays()
    alarm     = Alarm(enabled=not args.no_alarm)
    muted     = False
    relay     = None
    if args.relay:
        rtsp_url = cam_url or (src_str if rtsp else None)
        if rtsp_url:
            u = urlparse(rtsp_url)
            host, user_, pw_ = u.hostname, unquote(u.username or ""), unquote(u.password or "")
        else:
            host, user_, pw_ = (os.environ.get("CAM_IP"), os.environ.get("CAM_USER"),
                                os.environ.get("CAM_PASS"))
        if not (host and user_ and pw_):
            raise SystemExit("--relay needs --camera (CAM_URL) or CAM_IP / CAM_USER / CAM_PASS.")
        relay = CameraRelay(host, user_, pw_, args.relay_output, args.relay_invert, args.relay_port)
        relay.set(True)    # until the first frame is analysed, assume 'person in bay'
    frame_idx = 0
    paused    = False
    stats     = defaultdict(list)      # class id -> confidences (for threshold tuning)
    writer    = None
    names     = {CLASS_FORKLIFT: "forklift", CLASS_PERSON: "person", CLASS_TRUCK: "truck"}

    WIN = "Warehouse Forklift Safety System"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouse_callback)

    def open_cap(path):
        if live:
            return LiveStream(src_str, webcam)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")
        return cap

    cap        = open_cap(all_videos[video_idx])
    last_frame = None   # used when paused / zone-drawing freeze
    cam_lost   = False
    force_read = False
    show_truck = SHOW_TRUCK_BOX
    seekq      = {"req": None, "internal": False, "cur": 0.0}

    def on_track(v):
        if seekq["internal"] or abs(v / 1000.0 - seekq["cur"]) < 0.01:
            return
        seekq["req"] = ("frac", v / 1000.0)

    if not live:
        cv2.createTrackbar("pos", WIN, 0, 1000, on_track)

    def reset_tracking():
        track_age.clear()
        bays.clear()
        gate.history.clear()
        model.predictor = None   # next model.track() starts a fresh tracker


    print("=" * 55)
    print("  Warehouse Forklift Safety System  –  ready")
    print(f"  Zone : {'loaded' if zone_orig is not None else 'NOT SET – press Z'}")
    print("=" * 55)

    while True:
        # ── Frame acquisition ────────────────────────
        if state.drawing_zone:
            # Freeze: keep showing last_frame with zone overlay
            if last_frame is None:
                continue
            disp = last_frame.copy()
            draw_zone_preview_on_display(disp, state.zone_pts_disp)
            cv2.imshow(WIN, disp)

        else:
            if seekq["req"] is not None and not live:
                kind, val = seekq["req"]
                seekq["req"] = None
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps_  = cap.get(cv2.CAP_PROP_FPS) or 25
                cur   = cap.get(cv2.CAP_PROP_POS_FRAMES)
                target = cur + val * fps_ if kind == "sec" else val * total
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 2, target)))
                reset_tracking()
                force_read = True
            if not paused or force_read:
                force_read = False
                ok, frame = cap.read()
                if not ok:
                    if live:
                        # Camera lost: the system cannot see. Keep retrying, tell the hook once.
                        if not cam_lost:
                            print("[WARN] Camera stream lost - retrying. "
                                  "Gate state is UNKNOWN; handle fail-safe in on_camera_lost().")
                            on_camera_lost()
                            if relay:
                                relay.set(True)    # camera blind -> alarm state
                            cam_lost = True
                            relay_open = False
                            warm = 0
                            gate.history.clear()
                        if cv2.waitKey(100) & 0xFF == ord('q'):
                            break
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            # frame is valid (either new or held from pause)
            cam_lost = False
            if not live:
                total_ = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_ > 0:
                    seekq["cur"] = cap.get(cv2.CAP_PROP_POS_FRAMES) / total_
                    seekq["internal"] = True
                    cv2.setTrackbarPos("pos", WIN, int(1000 * seekq["cur"]))
                    seekq["internal"] = False

            frame_h, frame_w = frame.shape[:2]
            scale = min(MAX_WIDTH / frame_w, MAX_HEIGHT / frame_h, 1.0)
            state.display_scale = scale

            # ── Inference + Tracking ──────────────────
            frame_idx += 1
            crop_rect = None
            infer_img = frame
            if bay_crop and not auto_bay and zone_orig is not None:
                crop_rect = bay_crop_rect(zone_orig, frame_w, frame_h)
                if crop_rect:
                    cx0, cy0, cx1, cy1 = crop_rect
                    infer_img = frame[cy0:cy1, cx0:cx1]
            ox, oy = (crop_rect[0], crop_rect[1]) if crop_rect else (0, 0)
            thr = {CLASS_FORKLIFT: args.conf_forklift, CLASS_PERSON: conf,
                   CLASS_TRUCK: args.conf_truck}

            results = model.track(
                source=infer_img,
                conf=min(thr.values()),
                imgsz=args.imgsz,
                tracker=TRACKER,
                persist=True,      # keep tracker state across frames
                verbose=False,
            )

            forklift_boxes = []
            forklift_confs = []
            forklift_ids   = []
            person_boxes   = []
            person_ids     = []
            person_conf    = {}
            truck_boxes    = []

            # Update track ages; decay unseen tracks
            seen_ids = set()
            boxes_raw = results[0].boxes

            for box in boxes_raw:
                tid = int(box.id[0]) if box.id is not None else -1
                if tid >= 0:
                    track_age[tid] = track_age.get(tid, 0) + 1
                    seen_ids.add(tid)

            # Remove tracks not seen this frame
            for tid in list(track_age):
                if tid not in seen_ids:
                    del track_age[tid]

            for box in boxes_raw:
                cls  = int(box.cls[0])
                c    = float(box.conf[0])
                xyxy = list(map(int, box.xyxy[0].tolist()))
                xyxy = [xyxy[0] + ox, xyxy[1] + oy, xyxy[2] + ox, xyxy[3] + oy]
                tid  = int(box.id[0]) if box.id is not None else -1
                age  = track_age.get(tid, 0)
                stats[cls].append(c)
                if c < thr.get(cls, 0.5):
                    continue

                # Only act on tracks that have been stable for MIN_TRACK_AGE frames
                if age < MIN_TRACK_AGE:
                    continue

                if cls == CLASS_FORKLIFT:
                    if is_valid_forklift(xyxy, frame_w, frame_h):
                        forklift_boxes.append(xyxy)
                        forklift_confs.append(c)
                        forklift_ids.append(tid)
                elif cls == CLASS_PERSON:
                    person_boxes.append(xyxy)
                    person_ids.append(tid)
                    person_conf[tuple(xyxy)] = c
                elif cls == CLASS_TRUCK:
                    truck_boxes.append((xyxy, c, tid))

            # ── Association + gate ─────────────────────
            forklift_units, free_persons = associate(forklift_boxes, person_boxes)
            if auto_bay:
                bays.update([(tid, tb) for tb, _c, tid in truck_boxes], frame_idx)
                zones = bays.polys(bay_cfg, frame_w, frame_h)
            else:
                zones = [zone_orig] if zone_orig is not None else []
            gate_state, gate_msg = gate.update(free_persons, zones)
            alarm.set(gate_state == "CLOSED" and not muted)
            if relay:
                relay.set(gate_state == "CLOSED")
            if RELAY is not None:
                RELAY.heartbeat()
                warm += 1
                # permit-to-open only when: gate logic says OPEN, and we have seen
                # a full open-window of frames since start / camera loss
                want_open = (gate_state == "OPEN" and warm >= GATE_OPEN_FRAMES)
                if want_open != relay_open:
                    relay_open = want_open
                    RELAY.set_open(want_open)
                    print(f"[RELAY] {'clear -> ' if want_open else 'CLOSED -> '}relay {RELAY.permit_level.upper() if want_open else RELAY.closed_level.upper()}")

            # ── Build display frame ────────────────────
            if scale < 1.0:
                dw = int(frame_w * scale)
                dh = int(frame_h * scale)
                disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            else:
                disp = frame.copy()

            for z in zones:
                draw_zone_on_display(disp, z, gate_state, scale)

            if crop_rect:
                cv2.rectangle(disp, (int(crop_rect[0] * scale), int(crop_rect[1] * scale)),
                              (int(crop_rect[2] * scale), int(crop_rect[3] * scale)),
                              (120, 120, 120), 1)
            if show_truck:
                for tb, tc, _tid in truck_boxes:
                    draw_truck(disp, tb, tc, scale)
            for i, unit in enumerate(forklift_units):
                draw_forklift_unit(disp, unit, forklift_confs[i],
                                   forklift_ids[i], scale)
            for pb in free_persons:
                draw_free_person(disp, pb, scale, person_conf.get(tuple(pb)))

            draw_hud(disp, forklift_units, free_persons, conf,
                     all_videos[video_idx].name)
            cv2.rectangle(disp, (0, 70), (340, 90), (0, 0, 0), -1)
            mode_txt = (f"Bay: AUTO rear={bay_cfg['rear']} depth={bay_cfg['depth']:.1f} "
                        f"bays={len(zones)}" if auto_bay else "Bay: MANUAL zone (A = auto)")
            cv2.putText(disp, mode_txt, (8, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)
            draw_gate_banner(disp, gate_state, gate_msg, conf)

            last_frame = disp   # save for zone-drawing freeze
            if args.save:
                if writer is None:
                    h_, w_ = disp.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w_, h_))
                writer.write(disp)
            cv2.imshow(WIN, disp)

        # ── Key handling ─────────────────────────────
        raw = cv2.waitKeyEx(1)
        key = (raw & 0xFF) if raw != -1 else 255
        if not live:
            if raw == 2424832 or key == ord('j'):      # left arrow / J
                seekq["req"] = ("sec", -args.seek)
            elif raw == 2555904 or key == ord('l'):    # right arrow / L
                seekq["req"] = ("sec", args.seek)

        if key == ord('q'):
            break

        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")

        elif key == ord('z'):
            # Start zone drawing – freeze the frame
            state.drawing_zone  = True
            state.zone_pts_disp = []
            print("[Z] Zone drawing active – click corners on the image")

        elif key == ord('r'):
            zone_orig = None
            Path(ZONE_FILE).unlink(missing_ok=True)
            state.drawing_zone  = False
            state.zone_pts_disp = []
            print("[R] Zone cleared")

        elif key == 13:  # ENTER – confirm zone
            if state.drawing_zone:
                if len(state.zone_pts_disp) >= 3:
                    # Convert display-space clicks → original frame coords
                    pts_orig = disp_to_orig(state.zone_pts_disp,
                                            state.display_scale)
                    zone_orig = np.array(pts_orig, dtype=np.int32)
                    save_zone(ZONE_FILE, zone_orig)
                    auto_bay = False
                    print(f"[ENTER] Zone saved – {len(zone_orig)} points")
                else:
                    print("[ENTER] Need at least 3 points")
                state.drawing_zone  = False
                state.zone_pts_disp = []

        elif key == 27:  # ESC – cancel zone drawing
            state.drawing_zone  = False
            state.zone_pts_disp = []
            print("[ESC] Zone drawing cancelled")

        elif key == ord('c'):
            video_idx = (video_idx + 1) % len(all_videos)
            cap.release()
            cap = open_cap(all_videos[video_idx])
            reset_tracking()   # fresh tracker/bays for the new video
            print(f"[C] → {all_videos[video_idx].name}")

        elif key == ord('a'):
            auto_bay = not auto_bay
            bays.clear()
            print(f"[A] Bay mode: {'AUTO (from truck)' if auto_bay else 'MANUAL zone'}")

        elif key == ord('s'):
            i = REAR_SIDES.index(bay_cfg["rear"])
            bay_cfg["rear"] = REAR_SIDES[(i + 1) % len(REAR_SIDES)]
            save_bay_cfg(bay_cfg)
            print(f"[S] Truck rear side -> {bay_cfg['rear']}")

        elif key in (ord('['), ord(']')):
            step = 0.1 if key == ord(']') else -0.1
            bay_cfg["depth"] = round(min(2.0, max(0.1, bay_cfg["depth"] + step)), 2)
            save_bay_cfg(bay_cfg)
            print(f"[{chr(key)}] Bay depth -> {bay_cfg['depth']:.1f}")

        elif key == ord('m'):
            muted = not muted
            print(f"[M] Alarm {'MUTED' if muted else 'on'}")

        elif key == ord('t'):
            show_truck = not show_truck
            print(f"[T] Truck boxes {'shown' if show_truck else 'hidden'}")

        elif key in (ord(','), ord('.')):
            step = 0.05 if key == ord('.') else -0.05
            bay_cfg["lateral"] = round(min(1.0, max(0.0, bay_cfg["lateral"] + step)), 2)
            save_bay_cfg(bay_cfg)
            print(f"[{chr(key)}] Bay lateral margin -> {bay_cfg['lateral']:.2f}")

        elif key == ord('b'):
            bay_crop = not bay_crop
            track_age.clear()
            print(f"[B] Bay crop {'ON' if bay_crop else 'OFF'}")

        elif key in (ord('+'), ord('=')):
            conf = min(conf + 0.05, 0.95)
            print(f"[+] Confidence → {conf:.2f}")

        elif key == ord('-'):
            conf = max(conf - 0.05, 0.10)
            print(f"[-] Confidence → {conf:.2f}")

    if RELAY is not None:
        RELAY.close()
    alarm.set(False)
    if relay:
        relay.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\nDetection confidence summary (all frames, before track-age filter):")
    for cls_id, vals in sorted(stats.items()):
        arr = np.array(vals)
        print(f"  {names.get(cls_id, cls_id):9s} n={len(arr):6d}  mean={arr.mean():.2f}  "
              f"p10={np.percentile(arr, 10):.2f}  p50={np.percentile(arr, 50):.2f}")


if __name__ == "__main__":
    main()
