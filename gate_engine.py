"""
Headless multi-camera gate engine (no window).  One thread + one model per camera,
cameras spread over the GPUs.  Per camera: truck-based bay, gate logic, camera
relay (HIGH = person in bay), shared PC beep.  Annotated frames are served on
http://127.0.0.1:<port>/cam/<id>.mjpg for viewer.py; /status gives JSON.

    python gate_engine.py --config config.yaml
Reuses the detection / bay / gate code from test_video.py.
"""
import argparse
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

import contextlib

import cv2
import numpy as np
import torch         # import heavy libraries once, in the main thread,
import torchvision    # BEFORE worker threads start (concurrent first import crashed)
import yaml
from ultralytics import YOLO

import test_video as tv

log = logging.getLogger("engine")
WORKERS = {}
INIT_LOCK = threading.Lock()      # model load + first (warm-up) inference run one camera at a time


class FileLoopStream:
    """Looping video file used as a fake camera for local testing."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video file {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.t = time.time()

    def read(self, timeout=3.0):
        now = time.time()
        steps = max(1, int((now - self.t) * self.fps))
        self.t += steps / self.fps
        for _ in range(steps - 1):
            if not self.cap.grab():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class MultiAlarm:
    """One beeper thread. Every camera with beep enabled has its own tone, so you
    can hear WHICH stream has a person in its bay."""

    def __init__(self, enabled=True):
        self.enabled, self.active = enabled, {}
        try:
            import winsound
            self._beep = winsound.Beep
        except ImportError:
            self._beep = lambda f, d: (print("\a", end="", flush=True), time.sleep(d / 1000))
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, cid, hz, on):
        if on and self.enabled:
            self.active[cid] = hz
        else:
            self.active.pop(cid, None)

    def _loop(self):
        while True:
            tones = list(self.active.values())
            if not tones:
                time.sleep(0.05)
                continue
            for hz in tones:
                self._beep(int(hz), 250)
            time.sleep(0.15)


def load_zones(path):
    """Fixed bay polygons from a JSON file made by tools/draw_bay.py: [[ [x,y],... ], ...] in 0..1 image
    coordinates (a single polygon is accepted too). Returns [] if no file is configured."""
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data and isinstance(data[0][0], (int, float)):
        data = [data]
    return [[(float(x), float(y)) for x, y in poly] for poly in data]


def cam_url(cam):
    return cam.get("url") or os.environ.get(cam.get("url_env", ""), "") or None


class CameraWorker(threading.Thread):
    def __init__(self, cam, eng):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam, self.id = cam, str(cam["id"])
        self.label = cam.get("name", self.id)
        self.gpu = int(cam.get("gpu", 0))
        self.imgsz = int(eng.get("imgsz", 960))
        self.max_fps = float(eng.get("max_fps", 10))
        self.min_age = int(eng.get("min_age", 3))
        self.model_path = eng["model"]
        # precision: engine.quantize (16 = FP16, faster on NVIDIA GPUs; 32 = FP32). Old `half: true/false` still works.
        self.quantize = eng.get("quantize", 16 if eng.get("half", True) else 32)
        self.draw_boxes = bool(cam.get("draw_boxes", eng.get("draw_boxes", False)))   # truck bay zones + forklift boxes
        self.fixed_zones = load_zones(cam.get("zone_file"))   # if set: this fixed bay replaces the truck-based bay
        self.beep = bool(cam.get("beep", True))
        self.beep_hz = int(cam.get("beep_hz", 800))
        self.thr = {tv.CLASS_FORKLIFT: float(cam.get("conf_forklift", tv.CONF_FORKLIFT)),
                    tv.CLASS_PERSON: float(cam.get("conf_person", tv.CONF_PERSON)),
                    tv.CLASS_TRUCK: float(cam.get("conf_truck", tv.CONF_TRUCK))}
        self.bay_cfg = {"rear": cam.get("rear", "bottom"),
                        "depth": float(cam.get("depth", tv.BAY_CFG["depth"])),
                        "lateral": float(cam.get("lateral", tv.BAY_CFG["lateral"])),
                        "overlap": float(cam.get("overlap", tv.BAY_CFG["overlap"]))}
        self.gate = tv.GateController(
            max(1, round(float(eng.get("close_seconds", 0.5)) * self.max_fps)),
            max(1, round(float(eng.get("open_seconds", 3.0)) * self.max_fps)))
        self.bays = tv.TruckBays()
        self.stop = threading.Event()
        self.state, self.persons, self.fps, self.updated = "STARTING", 0, 0.0, 0.0
        self.jpeg, self.want_until = (None, 0.0), 0.0
        self.frame = (None, 0.0)       # latest annotated BGR frame + timestamp (for the in-process viewer)
        self.relay = None
        rc = cam.get("relay") or {}
        # the alarm output can live on another camera/device: relay.url_env names its URL variable
        url = (os.environ.get(rc["url_env"], "") or None) if rc.get("url_env") else cam_url(cam)
        if rc.get("enabled") and url and url.lower().startswith("rtsp"):
            u = urlparse(url)
            self.relay = tv.CameraRelay(u.hostname, unquote(u.username or ""), unquote(u.password or ""),
                                        int(rc.get("output", 1)), bool(rc.get("invert", False)),
                                        int(rc.get("port", 80)))
            self.relay.set(True)           # unknown until first analysed frame

    def _alarm_state(self, why):
        if self.state != "LOST":
            log.warning("[%s] camera LOST (%s)", self.id, why)
            tv.on_camera_lost()
        self.state = "LOST"
        if self.relay:
            self.relay.set(True)

    def _open(self):
        if self.cam.get("source"):
            return FileLoopStream(self.cam["source"])
        url = cam_url(self.cam)
        if not url:
            raise RuntimeError("no url / url_env / source configured")
        return tv.LiveStream(url, url.isdigit())

    def run(self):
        while not self.stop.is_set():
            try:
                self._loop()
            except Exception:
                log.exception("[%s] worker crashed - restarting in 5 s", self.id)
                self.state = "ERROR"
                if self.relay:
                    self.relay.set(True)
                self.stop.wait(5)
        if self.relay:
            self.relay.close()

    def _loop(self):
        dev = self.gpu if (torch.cuda.is_available() and self.gpu < torch.cuda.device_count()) \
            else (0 if torch.cuda.is_available() else "cpu")
        with INIT_LOCK:
            model = YOLO(self.model_path)
        log.info("[%s] started on device %s", self.id, dev)
        stream, track_age, idx, last_ok, t_prev = None, {}, 0, time.time(), time.time()
        conf = self.thr[tv.CLASS_PERSON]

        while not self.stop.is_set():
            t0 = time.time()
            if stream is None:
                try:
                    stream = self._open()
                except Exception as e:
                    self._alarm_state(f"open failed: {str(e)[:80]}")
                    self.stop.wait(5)
                    continue
            ok, frame = stream.read(timeout=3.0)
            if not ok:
                if time.time() - last_ok > 3.0:
                    self._alarm_state("no frames")
                self.stop.wait(0.2)
                continue
            last_ok = time.time()
            idx += 1
            h, w = frame.shape[:2]

            with (INIT_LOCK if idx == 1 else contextlib.nullcontext()):
                res = model.track(source=frame, conf=min(self.thr.values()), imgsz=self.imgsz,
                                  tracker=tv.TRACKER, persist=True, device=dev,
                                  quantize=(self.quantize if dev != "cpu" else 32), verbose=False)
            boxes = res[0].boxes
            seen = set()
            for b in boxes:
                tid = int(b.id[0]) if b.id is not None else -1
                if tid >= 0:
                    track_age[tid] = track_age.get(tid, 0) + 1
                    seen.add(tid)
            for tid in list(track_age):
                if tid not in seen:
                    del track_age[tid]

            f_boxes, f_confs, f_ids, p_boxes, p_conf, t_boxes = [], [], [], [], {}, []
            for b in boxes:
                cls, c = int(b.cls[0]), float(b.conf[0])
                xyxy = list(map(int, b.xyxy[0].tolist()))
                tid = int(b.id[0]) if b.id is not None else -1
                if c < self.thr.get(cls, 0.5) or track_age.get(tid, 0) < self.min_age:
                    continue
                if cls == tv.CLASS_FORKLIFT and tv.is_valid_forklift(xyxy, w, h):
                    f_boxes.append(xyxy); f_confs.append(c); f_ids.append(tid)
                elif cls == tv.CLASS_PERSON:
                    p_boxes.append(xyxy); p_conf[tuple(xyxy)] = c
                elif cls == tv.CLASS_TRUCK:
                    t_boxes.append((xyxy, c, tid))

            units, free_persons = tv.associate(f_boxes, p_boxes)
            self.bays.update([(tid, tb) for tb, _c, tid in t_boxes], idx)
            if self.fixed_zones:
                zones = [np.array([[x * w, y * h] for x, y in p], np.int32) for p in self.fixed_zones]
            else:
                zones = self.bays.polys(self.bay_cfg, w, h)
            gate_state, gate_msg = self.gate.update(free_persons, zones)
            if gate_state != self.state:
                log.info("[%s] -> %s (persons=%d)", self.id, gate_state, len(free_persons))
            self.state, self.persons = gate_state, len(free_persons)
            if self.relay:
                self.relay.set(gate_state == "CLOSED")

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev, self.updated = now, now

            if now < self.want_until:                      # only draw when a viewer is watching
                sc = min(960 / w, 540 / h, 1.0)
                disp = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else frame.copy()
                if self.draw_boxes:
                    for tb, tc, _tid in t_boxes:
                        tv.draw_truck(disp, tb, tc, sc)
                    for i, u in enumerate(units):
                        tv.draw_forklift_unit(disp, u, f_confs[i], f_ids[i], sc)
                if self.draw_boxes or self.fixed_zones:        # a fixed bay is always shown
                    for z in zones:
                        tv.draw_zone_on_display(disp, z, gate_state, sc)
                for pb in free_persons:
                    tv.draw_free_person(disp, pb, sc, p_conf.get(tuple(pb)))
                tv.draw_hud(disp, units, free_persons, conf, self.label)
                tv.draw_gate_banner(disp, gate_state, gate_msg, conf)
                self.frame = (disp, now)
                okj, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if okj:
                    self.jpeg = (buf.tobytes(), now)

            self.stop.wait(max(0.0, 1.0 / self.max_fps - (time.time() - t0)))

        if stream is not None:
            stream.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/status":
            body = json.dumps([{"id": w.id, "name": w.label, "state": w.state,
                                "persons": w.persons, "fps": round(w.fps, 1),
                                "age": round(time.time() - w.updated, 1) if w.updated else None}
                               for w in WORKERS.values()]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/cam/") and self.path.endswith(".mjpg"):
            w = WORKERS.get(self.path[5:-5])
            if not w:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = 0.0
            try:
                while True:
                    w.want_until = time.time() + 2.0
                    jpg, stamp = w.jpeg
                    if jpg is not None and stamp != last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        last = stamp
                    else:
                        time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        else:
            self.send_error(404)


def start(config_path, serve_http=True):
    """Load the config and start every camera worker. Returns the MultiAlarm.
    Used by main() (headless) and by gate_app.py (built-in viewer, serve_http=False)."""
    cfg_path = Path(config_path).resolve()
    os.chdir(cfg_path.parent)                         # relative paths in the config are relative to it
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[RotatingFileHandler("logs/engine.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"),
                                  logging.StreamHandler()])
    eng = cfg.get("engine", {})
    eng.setdefault("model", tv.MODEL_PATH)
    port = int(eng.get("http_port", 8765))

    for i, c in enumerate(cfg["cameras"]):            # must precede CameraWorker(), which reads beep_hz
        c.setdefault("beep_hz", 800 + 300 * i)
    for c in cfg["cameras"]:
        WORKERS[str(c["id"])] = CameraWorker(c, eng)
    alarm = MultiAlarm(enabled=(cfg.get("alarm") or {}).get("pc_beep", True))

    if serve_http:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)   # local only
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    for w in WORKERS.values():
        w.start()
    log.info("engine started: %d cameras%s", len(WORKERS),
             f", viewer feed on 127.0.0.1:{port}" if serve_http else " (built-in viewer)")
    return alarm


def stop_all(alarm):
    log.info("stopping")
    for w in WORKERS.values():
        alarm.set(w.id, 0, False)
        w.stop.set()
    for w in WORKERS.values():
        w.join(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    alarm = start(args.config)

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, lambda *_: stop.set())
    while not stop.is_set():
        for w in WORKERS.values():
            alarm.set(w.id, w.beep_hz, w.beep and w.state == "CLOSED")
        stop.wait(0.2)

    stop_all(alarm)


if __name__ == "__main__":
    main()
