"""
Decodes a lock.key and prints what's in it - the client label, the hwid it was issued
for, the camera cap, and the date. Needs no private key: a lock.key's payload is signed
but NOT secret, by design, so this doubles as a quick watermark lookup. If a copy of
the software turns up somewhere it shouldn't, ask whoever has it for their lock.key
(or pull it from the install folder) and run this on it to see which client it was
issued to.

    python tools\\read_lock.py path\\to\\lock.key
"""
import argparse
import base64
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("lock_path")
args = ap.parse_args()

lines = Path(args.lock_path).read_text(encoding="utf-8").strip().splitlines()
if len(lines) != 2:
    print("not a recognisable lock.key (expected 2 lines)")
    sys.exit(1)

try:
    data = json.loads(base64.b64decode(lines[0]))
except Exception as e:
    print(f"could not decode payload: {e}")
    sys.exit(1)

print("client:      ", data.get("client"))
print("hwid:        ", data.get("hwid"))
print("max_cameras: ", data.get("max_cameras"))
print("issued:      ", data.get("issued"))
print()
print("Note: this only reads the payload. It does NOT check the signature, so it")
print("tells you what the file CLAIMS, not that it's a genuine, unmodified lock -")
print("run hwlock.enforce()/load_lock() on the actual PC for that check.")
