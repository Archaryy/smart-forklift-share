"""
Run on your DEV PC, after you have the client's hwid (from --print-hwid on their PC).
Signs a lock.key for that one PC. Needs dev_private_key.pem (from tools\\keygen.py) in
the repo root - never copy that file anywhere else.

    python tools\\make_lock.py <hwid> "Client Name" --max-cameras 4 --out lock.key
"""
import argparse
import base64
import json
import sys
from datetime import date
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

root = Path(__file__).resolve().parent.parent
keyfile = root / "dev_private_key.pem"

ap = argparse.ArgumentParser()
ap.add_argument("hwid", help="output of --print-hwid on the client PC")
ap.add_argument("client", help="just a label, e.g. the site/branch name")
ap.add_argument("--max-cameras", type=int, default=4)
ap.add_argument("--out", default="lock.key")
args = ap.parse_args()

if not keyfile.exists():
    print(f"{keyfile} not found - run tools\\keygen.py first (once, on this dev PC).")
    sys.exit(1)

private_key = load_pem_private_key(keyfile.read_bytes(), password=None)
if not isinstance(private_key, Ed25519PrivateKey):
    print("dev_private_key.pem is not an Ed25519 key - did keygen.py write it?")
    sys.exit(1)

payload = json.dumps({
    "hwid": args.hwid,
    "client": args.client,
    "max_cameras": min(args.max_cameras, 4),
    "issued": date.today().isoformat(),
}, sort_keys=True).encode()
sig = private_key.sign(payload)

out = Path(args.out)
out.write_text(base64.b64encode(payload).decode() + "\n" + base64.b64encode(sig).decode() + "\n",
                encoding="utf-8")
print(f"Wrote {out} for hwid {args.hwid} ({args.client}, max {min(args.max_cameras, 4)} cameras)")
print("Copy it into that client's install folder, next to GateMonitor.exe, as lock.key")
