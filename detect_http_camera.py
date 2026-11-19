"""
Detection from an IP camera over HTTP snapshots (the same request that works
with curl), instead of RTSP.  Reuses the detection / bay / gate logic from
test_video.py, so both programs behave the same.

Credentials come from environment variables (never put the password in a file
or on the command line):

    PowerShell:
        $env:CAM_IP   = "192.168.1.64"
        $env:CAM_USER = "gateapp"
        $env:CAM_PASS = "your-password"
        python detect_http_camera.py

Equivalent curl test (should save a JPEG you can open):
    curl.exe --digest -u USER:PASS http://CAM_IP/ISAPI/Streaming/channels/101/picture -o snap.jpg

Keys:  Q quit | +/- person confidence | S rear side | [ ] bay depth | , . bay width | T truck boxes
"""
import argparse
import os
import subprocess
import threading
import time

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO

import test_video as tv


# ─────────────────────────────────────────────
# Snapshot "video stream"
# ─────────────────────────────────────────────

class SnapshotStream:
    """Polls the camera's JPEG snapshot URL in a background thread and always
    hands out the newest frame.  Frame rate is limited by the camera / network
    (typically a few frames per second), it is not true video."""

    def __init__(self, url, user, password, use_curl=False):
        self.url, self.user, self.password = url, user, password
        self.use_curl = use_curl
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(user, password)
        self._lock = threading.Lock()
        self._frame, self._stamp, self._seen = None, 0, 0
        self.last_error = ""
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _fetch(self):
        if self.use_curl:
            out = subprocess.run(
                ["curl.exe", "--digest", "-s", "--max-time", "5", "-u",
                 f"{self.user}:{self.password}", self.url],
                capture_output=True, timeout=8)
            data = out.stdout
        else:
            r = self.session.get(self.url, timeout=5)
            if r.status_code == 401:
                raise RuntimeError("401 Unauthorized - check CAM_USER / CAM_PASS and the user's permissions")
            r.raise_for_status()
            data = r.content
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Camera did not return an image: " + data[:160].decode("utf-8", "replace"))
        return img

    def _loop(self):
        while self._run:
            try:
                img = self._fetch()
                with self._lock:
                    self._frame, self._stamp = img, self._stamp + 1
                self.last_error = ""
            except Exception as e:                     # network, auth, decode ...
                self.last_error = str(e)[:200]
                time.sleep(1.0)

    def read(self, timeout=3.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._stamp != self._seen:
                    self._seen = self._stamp
                    return True, self._frame
            time.sleep(0.005)
        return False, None

    def release(self):
        self._run = False
        self._t.join(timeout=2)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=os.environ.get("CAM_IP"))
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--scheme", default="http", choices=["http", "https"])
    ap.add_argument("--channel", default="102", help="101 = main stream, 102 = sub stream")
    ap.add_argument("--path", default="/ISAPI/Streaming/channels/{channel}/picture",
                    help="snapshot path; {channel} is replaced")
    ap.add_argument("--use-curl", action="store_true", help="fetch each frame with curl.exe instead of requests")
    ap.add_argument("--model", default=tv.MODEL_PATH)
    ap.add_argument("--imgsz", type=int, default=tv.IMGSZ)
    ap.add_argument("--conf-person", type=float, default=tv.CONF_PERSON)
    ap.add_argument("--conf-forklift", type=float, default=tv.CONF_FORKLIFT)
    ap.add_argument("--conf-truck", type=float, default=tv.CONF_TRUCK)
    ap.add_argument("--rear", choices=tv.REAR_SIDES, default=None)
    ap.add_argument("--min-age", type=int, default=2,
                    help="frames a track must exist before it counts (snapshots are slow, keep small)")
    ap.add_argument("--close-frames", type=int, default=3, help="frames with a person before the gate closes")
    ap.add_argument("--open-frames", type=int, default=10, help="clear frames before the gate reopens")
    ap.add_argument("--once", default=None, metavar="OUT.jpg",
                    help="grab ONE frame, run detection, save the annotated image and exit")
    ap.add_argument("--save", default=None, help="write annotated video to this .mp4")
    return ap.parse_args()


def main():
    args = parse_args()
    user, pw = os.environ.get("CAM_USER"), os.environ.get("CAM_PASS")
    if not (args.ip and user and pw):
        raise SystemExit("Set CAM_IP, CAM_USER and CAM_PASS environment variables first (see top of this file).")

    path = args.path.format(channel=args.channel)
    url = f"{args.scheme}://{args.ip}:{args.port}{path}"     # never printed: keep logs clean
    stream = SnapshotStream(url, user, pw, use_curl=args.use_curl)

    model    = YOLO(args.model)
    bay_cfg  = tv.load_bay_cfg()
    if args.rear:
        bay_cfg["rear"] = args.rear
    bays     = tv.TruckBays()
    gate     = tv.GateController(args.close_frames, args.open_frames)
    track_age = {}
    conf     = args.conf_person
    show_truck = tv.SHOW_TRUCK_BOX
    cam_lost, last_warn, frame_idx = False, 0.0, 0
    fps, t_prev, writer = 0.0, time.time(), None

    WIN = "Hikvision HTTP snapshot detection"
    if not args.once:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    print("Waiting for the first frame...")

    try:
        while True:
            ok, frame = stream.read(timeout=3.0)
            if not ok:
                if not cam_lost:
                    print("[WARN] No frames from camera. Gate state is UNKNOWN; handle in on_camera_lost().")
                    tv.on_camera_lost()
                    cam_lost = True
                if stream.last_error and time.time() - last_warn > 5:
                    print("      last error:", stream.last_error)
                    last_warn = time.time()
                if args.once:
                    raise SystemExit("Could not get a frame: " + (stream.last_error or "timeout"))
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
                continue
            cam_lost = False
            frame_idx += 1
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3)) if frame_idx > 1 else 0.0
            t_prev = now

            h, w = frame.shape[:2]
            scale = min(tv.MAX_WIDTH / w, tv.MAX_HEIGHT / h, 1.0)
            thr = {tv.CLASS_FORKLIFT: args.conf_forklift, tv.CLASS_PERSON: conf,
                   tv.CLASS_TRUCK: args.conf_truck}

            res = model.track(source=frame, conf=min(thr.values()), imgsz=args.imgsz,
                              tracker=tv.TRACKER, persist=True, verbose=False)
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

            f_boxes, f_confs, f_ids = [], [], []
            p_boxes, p_conf, t_boxes = [], {}, []
            for b in boxes:
                cls, c = int(b.cls[0]), float(b.conf[0])
                xyxy = list(map(int, b.xyxy[0].tolist()))
                tid = int(b.id[0]) if b.id is not None else -1
                if c < thr.get(cls, 0.5) or track_age.get(tid, 0) < args.min_age:
                    continue
                if cls == tv.CLASS_FORKLIFT and tv.is_valid_forklift(xyxy, w, h):
                    f_boxes.append(xyxy); f_confs.append(c); f_ids.append(tid)
                elif cls == tv.CLASS_PERSON:
                    p_boxes.append(xyxy); p_conf[tuple(xyxy)] = c
                elif cls == tv.CLASS_TRUCK:
                    t_boxes.append((xyxy, c, tid))

            units, free_persons = tv.associate(f_boxes, p_boxes)
            bays.update([(tid, tb) for tb, _c, tid in t_boxes], frame_idx)
            zones = bays.polys(bay_cfg, w, h)
            gate_state, gate_msg = gate.update(free_persons, zones)

            disp = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
            for z in zones:
                tv.draw_zone_on_display(disp, z, gate_state, scale)
            if show_truck:
                for tb, tc, _tid in t_boxes:
                    tv.draw_truck(disp, tb, tc, scale)
            for i, u in enumerate(units):
                tv.draw_forklift_unit(disp, u, f_confs[i], f_ids[i], scale)
            for pb in free_persons:
                tv.draw_free_person(disp, pb, scale, p_conf.get(tuple(pb)))
            tv.draw_hud(disp, units, free_persons, conf, "http-snapshot")
            cv2.rectangle(disp, (0, 70), (340, 90), (0, 0, 0), -1)
            cv2.putText(disp, f"{fps:4.1f} FPS  rear={bay_cfg['rear']} depth={bay_cfg['depth']:.1f} bays={len(zones)}",
                        (8, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tv.draw_gate_banner(disp, gate_state, gate_msg, conf)

            if args.once:
                cv2.imwrite(args.once, disp)
                print(f"Saved {args.once}  (forklifts={len(units)}, free persons={len(free_persons)}, bays={len(zones)})")
                break
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 5,
                                             (disp.shape[1], disp.shape[0]))
                writer.write(disp)
            cv2.imshow(WIN, disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in (ord('+'), ord('=')):
                conf = min(conf + 0.05, 0.95); print(f"person conf -> {conf:.2f}")
            elif key == ord('-'):
                conf = max(conf - 0.05, 0.10); print(f"person conf -> {conf:.2f}")
            elif key == ord('s'):
                i = tv.REAR_SIDES.index(bay_cfg["rear"])
                bay_cfg["rear"] = tv.REAR_SIDES[(i + 1) % len(tv.REAR_SIDES)]
                tv.save_bay_cfg(bay_cfg); print("rear ->", bay_cfg["rear"])
            elif key in (ord('['), ord(']')):
                bay_cfg["depth"] = round(min(2.0, max(0.1, bay_cfg["depth"] + (0.1 if key == ord(']') else -0.1))), 2)
                tv.save_bay_cfg(bay_cfg); print("depth ->", bay_cfg["depth"])
            elif key in (ord(','), ord('.')):
                bay_cfg["lateral"] = round(min(1.0, max(0.0, bay_cfg["lateral"] + (0.05 if key == ord('.') else -0.05))), 2)
                tv.save_bay_cfg(bay_cfg); print("width ->", bay_cfg["lateral"])
            elif key == ord('t'):
                show_truck = not show_truck
    finally:
        stream.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
