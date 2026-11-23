"""
Single-input alarm test WITH a software delay (anti false-positive). Same engine/detection as
test_single.py (which has no timers); this version only raises the alarm after a person has been
in the bay for --alarm-delay seconds (default 5) in a row.

  python test_single_delay.py --source CAM_URL
  python test_single_delay.py --source CAM_URL --alarm-delay 10 --relay

  person in bay           -> amber "confirming N s" countdown (no sound, relay LOW)
  still there after 5 s   -> ALARM: looping siren + red window (+ relay HIGH with --relay)
  person gone             -> countdown resets (alarm stops) once the bay has been clear for
                             --dropout seconds (default 1; tolerates a few missed frames so a
                             flickering detection doesn't restart the 5 s). 0 = stop at once.
  gone before 5 s         -> no alarm at all
Stream lost / starting -> relay HIGH immediately (fail-safe), no delay.

Keys: Q quit | M mute.   Same options as test_single.py (--relay, --no-window, --seconds, --snapshot ...).
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import cv2
import numpy as np

import gate_engine as ge
import test_video as tv
from test_single import SirenAlarm, make_cam, overlay


def main():
    ap = argparse.ArgumentParser(description="Single-input person-in-bay alarm test (with delay)")
    ap.add_argument("--source", required=True, help="video file | rtsp/http URL | webcam number | env variable name")
    ap.add_argument("--alarm-delay", type=float, default=5.0, help="seconds a person must stay in the bay (default 5)")
    ap.add_argument("--dropout", type=float, default=1.0, help="seconds the bay may look empty before the countdown resets")
    ap.add_argument("--name", default="Test camera")
    ap.add_argument("--model", default=tv.MODEL_PATH)
    ap.add_argument("--show-boxes", action="store_true")
    ap.add_argument("--zone", help="fixed bay JSON from tools/draw_bay.py (replaces the truck-based bay)")
    ap.add_argument("--rear", default="bottom", choices=["bottom", "top", "left", "right"])
    ap.add_argument("--depth", type=float, default=tv.BAY_CFG["depth"])
    ap.add_argument("--lateral", type=float, default=tv.BAY_CFG["lateral"])
    ap.add_argument("--overlap", type=float, default=tv.BAY_CFG["overlap"])
    ap.add_argument("--conf-person", type=float, default=tv.CONF_PERSON)
    ap.add_argument("--conf-forklift", type=float, default=tv.CONF_FORKLIFT)
    ap.add_argument("--conf-truck", type=float, default=tv.CONF_TRUCK)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--relay", action="store_true")
    ap.add_argument("--relay-output", type=int, default=1)
    ap.add_argument("--relay-port", type=int, default=80)
    ap.add_argument("--relay-invert", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--expect-alarm", action="store_true")
    ap.add_argument("--snapshot")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    ge.log.setLevel(logging.WARNING)
    # engine reports the raw per-frame result (hold 0); the delay is done here
    eng = {"model": args.model, "imgsz": args.imgsz, "max_fps": args.fps, "quantize": 16, "min_age": 3,
           "close_seconds": 0, "open_seconds": 0}
    if not Path(eng["model"]).exists():
        sys.exit(f"model not found: {eng['model']}")
    cam = make_cam(args)
    relay = None
    if args.relay:
        u = urlparse(cam["url"])
        relay = tv.CameraRelay(u.hostname, unquote(u.username or ""), unquote(u.password or ""),
                               args.relay_output, args.relay_invert, args.relay_port)
        relay.set(True)
    worker = ge.CameraWorker(cam, eng)
    siren = SirenAlarm(enabled=not args.no_beep)
    worker.start()
    WIN = "Single camera alarm test - delayed  (Q quit, M mute)"
    if not args.no_window:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)
    print(f"source: {'(' + args.source + ')' if cam.get('url') else args.source}\n"
          f"alarm only after {args.alarm_delay:g} s continuously in bay (dropout tolerance {args.dropout:g} s)   "
          f"relay: {'ON' if relay else 'off'}\nloading model...")

    t0 = time.time()
    t_start = last_seen = alarm_since = None
    alarm_on, alarms, ghosts, alarm_total, snap_saved, last_frame = False, 0, 0, 0.0, False, None
    try:
        while True:
            now = time.time()
            worker.want_until = now + 2.0
            st = worker.state
            if st == "CLOSED":                              # person in bay (raw, this frame)
                last_seen = now
                if t_start is None:
                    t_start = now
                    print(f"[{time.strftime('%H:%M:%S')}] person in bay - alarm in {args.alarm_delay:g} s if they stay")
            elif t_start is not None and now - last_seen > args.dropout:
                if alarm_on:
                    alarm_total += now - alarm_since
                    print(f"[{time.strftime('%H:%M:%S')}] [ALARM] OFF - bay clear (rang {now - alarm_since:.1f} s)")
                else:
                    ghosts += 1
                    print(f"[{time.strftime('%H:%M:%S')}] person gone after {last_seen - t_start:.1f} s - no alarm")
                t_start, alarm_on = None, False

            if t_start is not None and not alarm_on and now - t_start >= args.alarm_delay:
                alarm_on, alarm_since = True, now
                alarms += 1
                print(f"[{time.strftime('%H:%M:%S')}] [ALARM] ON - person in bay for {now - t_start:.1f} s")
            siren.set(alarm_on)
            if relay:
                relay.set(alarm_on or st in ("STARTING", "LOST", "ERROR"))

            mode = ("ALARM" if alarm_on else "LOST" if st in ("LOST", "ERROR")
                    else "STARTING" if st == "STARTING" else "CLEAR")
            frame, stamp_f = worker.frame
            if frame is not None and now - stamp_f < 3.0:
                last_frame = overlay(frame, mode, worker.persons, now, not siren.enabled)
                if t_start is not None and not alarm_on:        # confirming: amber instead of green
                    h, w = last_frame.shape[:2]
                    cv2.rectangle(last_frame, (0, 0), (w - 1, h - 1), (0, 150, 255), 8)
                    cv2.rectangle(last_frame, (0, h - 64), (w, h), (0, 150, 255), -1)
                    cv2.putText(last_frame, f"PERSON IN BAY - confirming {max(0.0, args.alarm_delay - (now - t_start)):.1f} s",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
                if args.snapshot and not snap_saved and alarm_on:
                    cv2.imwrite(args.snapshot, last_frame)
                    snap_saved = True
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
        if alarm_on and alarm_since:
            alarm_total += time.time() - alarm_since
        siren.set(False)
        worker.stop.set()
        worker.join(timeout=5)
        if relay:
            relay.close()
        if args.snapshot and not snap_saved and last_frame is not None:
            cv2.imwrite(args.snapshot, last_frame)
        cv2.destroyAllWindows()
    print(f"\nsummary: {alarms} alarm(s) ({alarm_total:.1f} s ringing), {ghosts} detection(s) ignored "
          f"(gone before {args.alarm_delay:g} s), ran {time.time() - t0:.0f} s")
    return 2 if (args.expect_alarm and alarms == 0) else 0


if __name__ == "__main__":
    sys.exit(main())
