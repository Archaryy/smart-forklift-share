"""
Single-input alarm test: one camera / video file, same detection + bay logic as the production
engine (it uses gate_engine.CameraWorker). Alarm = a person standing in the bay.

  python test_single.py --source CAM_URL                 (URL taken from that environment variable)
  python test_single.py --source "rtsp://USER:PASS@192.168.2.77:554/Streaming/Channels/101"
  python test_single.py --source CAM_URL --relay         (also drive the camera's alarm output)

NO timers in this program: the alarm (looping siren/alarm sound + red window + camera relay) is ON
while the engine sees a person in the bay and OFF as soon as it does not - the next analysed frame
(--fps, default 5/s). Delays are left to your relay's hardware timer.
(--close-seconds / --open-seconds bring back a software hold if you ever want one.)
Only the engine's --min-age (a new detection must be tracked for 3 frames) still filters out one-frame ghosts.

The window shows the camera with PERSON boxes only. Truck bays and forklift boxes are hidden;
--show-boxes brings them back.

Keys: Q quit | M mute sound.        Headless run: --no-window --seconds 60
Exit code: 0 = ran fine, 2 = no alarm was raised (only with --expect-alarm).
"""
import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import cv2
import numpy as np

import gate_engine as ge
import test_video as tv


def _default_beep():
    try:
        import winsound
        return winsound.Beep
    except ImportError:
        return lambda hz, ms: (print("\a", end="", flush=True), time.sleep(ms / 1000))


class SirenAlarm:
    """While active, loops: siren (two rising/falling sweeps) -> alarm (burst of 5 high beeps) -> repeat."""

    def __init__(self, enabled=True, beep=None):
        self.enabled, self.active = enabled, False
        self._beep = beep or _default_beep()
        threading.Thread(target=self._loop, daemon=True, name="siren").start()

    def set(self, on):
        self.active = bool(on)

    def _on(self):
        return self.active and self.enabled

    def _loop(self):
        sweep = list(range(600, 1300, 100)) + list(range(1300, 600, -100))
        while True:
            if not self._on():
                time.sleep(0.05)
                continue
            for _ in range(2):                       # siren
                for hz in sweep:
                    if not self._on():
                        break
                    self._beep(hz, 60)
            for _ in range(5):                       # alarm
                if not self._on():
                    break
                self._beep(1800, 150)
                time.sleep(0.1)


def make_cam(args):
    src = args.source
    cam = {"id": "test", "name": args.name, "gpu": args.gpu, "beep": False, "rear": args.rear,
           "depth": args.depth, "lateral": args.lateral, "overlap": args.overlap,
           "conf_person": args.conf_person, "conf_forklift": args.conf_forklift, "conf_truck": args.conf_truck,
           "draw_boxes": args.show_boxes}
    if os.environ.get(src):                                   # name of an environment variable
        cam["url"] = os.environ[src]
    elif src.isdigit() or src.lower().startswith(("rtsp://", "http://", "https://")):
        cam["url"] = src
    elif Path(src).exists():
        cam["source"] = src
    else:
        sys.exit(f"source not found: {src}  (file, rtsp/http URL, webcam number, or env variable name)")
    if args.relay and not cam.get("url", "").lower().startswith("rtsp"):
        sys.exit("--relay needs an rtsp:// camera source (the relay is the camera's own alarm output)")
    if getattr(args, "zone", None):
        cam["zone_file"] = args.zone                      # fixed hand-drawn bay (tools/draw_bay.py)
    return cam      # the relay is driven by this script, not by the worker


def overlay(frame, mode, persons, now, muted):
    """Big status banner around the engine's annotated frame. mode: ALARM / CLEAR / LOST / STARTING"""
    h, w = frame.shape[:2]
    scale = 1280 / w if w < 1280 else 1.0
    img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame.copy()
    h, w = img.shape[:2]
    if mode == "ALARM":
        col = (0, 0, 220) if int(now * 3) % 2 == 0 else (0, 0, 140)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 14)
        txt = f"ALARM - PERSON IN BAY ({persons})"
    elif mode == "LOST":
        col, txt = (0, 140, 255), "NO SIGNAL"
    elif mode == "STARTING":
        col, txt = (120, 120, 120), "STARTING..."
    else:
        col, txt = (0, 160, 60), "CLEAR"
    cv2.rectangle(img, (0, h - 64), (w, h), col, -1)
    cv2.putText(img, txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    if muted:
        cv2.putText(img, "SOUND MUTED", (w - 290, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description="Single-input person-in-bay alarm test (no program timers)")
    ap.add_argument("--source", required=True, help="video file | rtsp/http URL | webcam number | env variable name")
    ap.add_argument("--name", default="Test camera")
    ap.add_argument("--model", default=tv.MODEL_PATH)
    ap.add_argument("--config", help="optional config.yaml to take engine settings from (imgsz, fps, close/open seconds...)")
    ap.add_argument("--show-boxes", action="store_true", help="also draw the truck bay zones and forklift boxes")
    ap.add_argument("--zone", help="fixed bay JSON from tools/draw_bay.py (replaces the truck-based bay)")
    ap.add_argument("--rear", default="bottom", choices=["bottom", "top", "left", "right"], help="rear side of the truck in the image")
    ap.add_argument("--depth", type=float, default=tv.BAY_CFG["depth"])
    ap.add_argument("--lateral", type=float, default=tv.BAY_CFG["lateral"])
    ap.add_argument("--overlap", type=float, default=tv.BAY_CFG["overlap"])
    ap.add_argument("--conf-person", type=float, default=tv.CONF_PERSON)
    ap.add_argument("--conf-forklift", type=float, default=tv.CONF_FORKLIFT)
    ap.add_argument("--conf-truck", type=float, default=tv.CONF_TRUCK)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fps", type=float, default=5, help="max analysed frames per second")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--close-seconds", type=float, default=0.0,
                    help="software hold before 'in bay' (default 0 = next frame; the hardware timer does the waiting)")
    ap.add_argument("--open-seconds", type=float, default=0.0,
                    help="software hold before 'clear' (default 0 = next frame)")
    ap.add_argument("--no-beep", action="store_true", help="no sound")
    ap.add_argument("--relay", action="store_true", help="also drive the camera's alarm output (rtsp source)")
    ap.add_argument("--relay-output", type=int, default=1)
    ap.add_argument("--relay-port", type=int, default=80)
    ap.add_argument("--relay-invert", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0 = until Q)")
    ap.add_argument("--expect-alarm", action="store_true", help="exit code 2 if no alarm was raised")
    ap.add_argument("--snapshot", help="save the first alarm frame (or the last frame) to this PNG")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    ge.log.setLevel(logging.WARNING)

    eng = {"model": args.model, "imgsz": args.imgsz, "max_fps": args.fps, "quantize": 16, "min_age": 3,
           "close_seconds": args.close_seconds, "open_seconds": args.open_seconds}
    if args.config:
        import yaml
        eng.update((yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}).get("engine", {}))
    if not Path(eng["model"]).exists():
        sys.exit(f"model not found: {eng['model']}  (use --model path\\to\\best.pt)")

    cam = make_cam(args)
    relay = None
    if args.relay:
        u = urlparse(cam["url"])
        relay = tv.CameraRelay(u.hostname, unquote(u.username or ""), unquote(u.password or ""),
                               args.relay_output, args.relay_invert, args.relay_port)
        relay.set(True)                                   # unknown until the first analysed frame (fail-safe)
    worker = ge.CameraWorker(cam, eng)
    siren = SirenAlarm(enabled=not args.no_beep)
    worker.start()
    WIN = "Single camera alarm test  (Q quit, M mute)"
    if not args.no_window:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
    print(f"source: {'(' + args.source + ')' if cam.get('url') else args.source}\nmodel : {eng['model']}\n"
          f"program timers: none (hold close {args.close_seconds:g} s / open {args.open_seconds:g} s)   "
          f"relay: {'ON' if relay else 'off'}   sound: {'off' if args.no_beep else 'on'}\n"
          f"loading model, first result in a few seconds...")

    t0 = time.time()
    prev, alarm_since, alarms, alarm_total, snap_saved, last_frame = "", None, 0, 0.0, False, None
    try:
        while True:
            now = time.time()
            worker.want_until = now + 2.0
            st = worker.state
            alarm_on = st == "CLOSED"
            if st != prev:
                stamp = time.strftime("%H:%M:%S")
                if alarm_on:
                    alarms += 1
                    alarm_since = now
                    print(f"[{stamp}] [ALARM] ON - person in bay (persons={worker.persons})")
                else:
                    if prev == "CLOSED":
                        alarm_total += now - alarm_since
                        print(f"[{stamp}] [ALARM] OFF - bay clear (rang {now - alarm_since:.1f} s)")
                    print(f"[{stamp}] state: {st}")
                prev = st
            siren.set(alarm_on)
            if relay:
                relay.set(alarm_on or st in ("STARTING", "LOST", "ERROR"))

            mode = "ALARM" if alarm_on else "LOST" if st in ("LOST", "ERROR") else "STARTING" if st == "STARTING" else "CLEAR"
            frame, stamp_f = worker.frame
            if frame is not None and now - stamp_f < 3.0:
                last_frame = overlay(frame, mode, worker.persons, now, not siren.enabled)
                if args.snapshot and not snap_saved and alarm_on:
                    cv2.imwrite(args.snapshot, last_frame)
                    snap_saved = True
                    print(f"saved alarm frame -> {args.snapshot}")
            if not args.no_window:
                if last_frame is None:
                    wait = np.zeros((720, 1280, 3), np.uint8)
                    cv2.putText(wait, "loading model / waiting for first frame...", (60, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 160, 160), 2, cv2.LINE_AA)
                    cv2.imshow(WIN, wait)
                else:
                    cv2.imshow(WIN, last_frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 0:
                    break
                if k in (ord("m"), ord("M")):
                    siren.enabled = not siren.enabled
            else:
                time.sleep(0.05)
            if args.seconds and now - t0 >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if prev == "CLOSED" and alarm_since:
            alarm_total += time.time() - alarm_since
        siren.set(False)
        worker.stop.set()
        worker.join(timeout=5)
        if relay:
            relay.close()                                  # ends in the alarm (HIGH) state, like the engine
        if args.snapshot and not snap_saved and last_frame is not None:
            cv2.imwrite(args.snapshot, last_frame)
        cv2.destroyAllWindows()

    print(f"\nsummary: {alarms} alarm(s) ({alarm_total:.1f} s in alarm), ran {time.time() - t0:.0f} s, avg {worker.fps:.1f} fps")
    return 2 if (args.expect_alarm and alarms == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
