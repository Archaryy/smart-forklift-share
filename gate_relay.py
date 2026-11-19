"""
gate_relay.py - fail-safe control of the Hikvision alarm-output relay.

Meaning: relay HIGH (closed contact) = bay clear + system healthy = gate MAY open.
         relay LOW  = everything else (person in bay, camera lost, startup, error, exit).

Config (environment variables only, nothing secret in files):
  CAM_URL        rtsp://USER:PASS@IP:554/...  (credentials and IP are taken from it)
     or CAM_IP, CAM_USER, CAM_PASS
  GATE_MODE      hold  (default) - relay held HIGH while clear, re-sent every GATE_RESEND_SEC
                 pulse           - one HIGH pulse of GATE_PULSE_SEC each time the bay becomes clear
                                   (re-pulsed every GATE_REPULSE_SEC while clear, 0 = never)
  GATE_RESEND_SEC     1.0   hold mode refresh period
  GATE_PULSE_SEC      0.5   pulse width
  GATE_REPULSE_SEC    0     pulse mode: repeat pulse while clear (0 = off)
  GATE_HEARTBEAT_SEC  2.0   if heartbeat() is not called this often, relay goes LOW
  GATE_DRYRUN         1     log only, never touch the camera
  GATE_EXIT           release (default) | alarm  - alarm polarity only: relay state on normal exit
  GATE_POLARITY       permit (default) | alarm  - alarm: relay HIGH = person in bay / any failure,
                      LOW = bay clear. Failure states (start-up, camera lost, freeze, exit) raise the alarm.

Usage in test_video.py:
    relay = GateRelay()                       # forces relay LOW immediately
    ...each frame, after detection...         relay.heartbeat()
    on_gate_change(is_open):                  relay.set_open(is_open)
    on_camera_lost():                         relay.fail_closed()
    on quit / finally:                        relay.close()

Limits: a software hang is covered by the heartbeat watchdog. A hard crash, power loss
or network loss while the relay is HIGH is NOT covered by software - the camera keeps
the last state. Cover that with a hardware off-delay / watchdog relay (see handoff #3).
"""
import atexit
import logging
import os
import signal
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse, unquote

log = logging.getLogger("gate_relay")


def _creds():
    url = os.environ.get("CAM_URL", "")
    if url:
        p = urlparse(url)
        if p.hostname:
            return p.hostname, unquote(p.username or ""), unquote(p.password or "")
    return (os.environ.get("CAM_IP", ""), os.environ.get("CAM_USER", ""),
            os.environ.get("CAM_PASS", ""))


class GateRelay:
    def __init__(self, base_url=None, mode=None, invert=None):
        self.dry = os.environ.get("GATE_DRYRUN") == "1"
        self.mode = (mode or os.environ.get("GATE_MODE", "hold")).lower()
        if self.mode not in ("hold", "pulse"):
            raise ValueError("GATE_MODE must be hold or pulse")
        # invert=False: relay HIGH = permit open (bay clear), LOW = closed / any failure
        # invert=True : relay HIGH = ALARM (person in bay, or any failure), LOW = clear
        if invert is None:
            invert = os.environ.get("GATE_POLARITY", "permit").lower() == "alarm"
        self.invert = bool(invert)
        if self.invert and self.mode == "pulse":
            raise ValueError("pulse mode is not supported with alarm polarity")
        self.permit_level = "low" if self.invert else "high"
        self.closed_level = "high" if self.invert else "low"
        self.resend = float(os.environ.get("GATE_RESEND_SEC", "1.0"))
        self.pulse = float(os.environ.get("GATE_PULSE_SEC", "0.5"))
        self.repulse = float(os.environ.get("GATE_REPULSE_SEC", "0"))
        self.hb_timeout = float(os.environ.get("GATE_HEARTBEAT_SEC", "2.0"))

        host, user, pw = _creds()
        if base_url:
            self.base = base_url.rstrip("/")
        else:
            if not self.dry and not (host and user and pw):
                raise RuntimeError("Set CAM_URL (or CAM_IP/CAM_USER/CAM_PASS)")
            self.base = f"http://{host}"
        self.endpoint = self.base + "/ISAPI/System/IO/outputs/1/trigger"

        mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        mgr.add_password(None, self.base, user, pw)
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPDigestAuthHandler(mgr))

        self._want_open = False          # what the detector says
        self._last_hb = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._io = threading.Lock()      # serialises HTTP sends
        self._opened_at = 0.0            # pulse mode bookkeeping
        self._pulsed = False
        self.state = None                # last state we successfully sent

        self._send(self.closed_level)                # fail-safe at startup
        self._t = threading.Thread(target=self._loop, daemon=True, name="gate-relay")
        self._t.start()
        atexit.register(self.close)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)

                def handler(s, f, prev=prev):
                    self.close()
                    if callable(prev):
                        prev(s, f)
                    else:
                        raise KeyboardInterrupt
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass                     # not main thread / unsupported

    # ---- public API -------------------------------------------------
    def set_open(self, is_open: bool):
        """Detector verdict: True = bay clear (permit open), False = person in bay."""
        with self._lock:
            was = self._want_open
            self._want_open = bool(is_open)
            if is_open and not was:
                self._pulsed = False
                self._opened_at = time.monotonic()
        if not is_open:
            self._send(self.closed_level)            # act immediately, don't wait for the thread

    def heartbeat(self):
        """Call once per processed frame. Silence => relay goes LOW."""
        self._last_hb = time.monotonic()

    def fail_closed(self):
        """Camera stream lost or any doubt about the detector."""
        with self._lock:
            self._want_open = False
        self._send(self.closed_level)

    def close(self):
        self._stop.set()
        with self._lock:
            self._want_open = False
        if self._t.is_alive() and threading.current_thread() is not self._t:
            self._t.join(timeout=2.5)    # let any in-flight send finish first
        # Alarm polarity: on a normal exit the relay is released (OFF = no alarm) unless
        # GATE_EXIT=alarm. Permit polarity always ends closed (OFF).
        if self.invert and os.environ.get("GATE_EXIT", "release").lower() != "alarm":
            self._send(self.permit_level)
        else:
            self._send(self.closed_level)

    # ---- internals --------------------------------------------------
    def _send(self, level: str) -> bool:
        with self._io:
            return self._send_locked(level)

    def _send_locked(self, level: str) -> bool:
        if self.dry:
            if level != self.state:
                log.info("[dryrun] relay %s", level)
            self.state = level
            return True
        body = (f"<IOPortData><outputState>{level}</outputState></IOPortData>").encode()
        req = urllib.request.Request(self.endpoint, data=body, method="PUT",
                                     headers={"Content-Type": "application/xml"})
        try:
            with self._opener.open(req, timeout=2.0) as r:
                ok = 200 <= r.status < 300
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log.error("relay %s failed: %s", level, e)
            ok = False
        if ok:
            if level != self.state:
                log.info("relay %s", level)
            self.state = level
        else:
            self.state = None            # unknown
        return ok

    def _loop(self):
        while not self._stop.wait(0.1):
            now = time.monotonic()
            with self._lock:
                want = self._want_open
                opened_at = self._opened_at
                pulsed = self._pulsed
            alive = (now - self._last_hb) <= self.hb_timeout
            if not (want and alive):
                if self.state != self.closed_level:  # also retries after a failed LOW
                    self._send(self.closed_level)
                continue
            if self.mode == "hold":
                if self.state != self.permit_level or (now - getattr(self, "_t_sent", 0)) >= self.resend:
                    if self._send(self.permit_level):
                        self._t_sent = now
            else:                        # pulse
                due = (not pulsed) or (self.repulse > 0 and
                                       now - opened_at >= self.repulse)
                if due:
                    with self._lock:
                        self._pulsed = True
                        self._opened_at = now
                    if self._send(self.permit_level):
                        time.sleep(self.pulse)
                        self._send(self.closed_level)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    r = GateRelay()
    if cmd == "off":
        print("relay forced LOW")
    elif cmd == "test":
        print("LOW 2s -> HIGH 3s -> LOW  (watch the multimeter / gate input)")
        time.sleep(2)
        r.heartbeat(); r.set_open(True)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            r.heartbeat(); time.sleep(0.2)
        r.set_open(False)
        time.sleep(1)
    r.close()
