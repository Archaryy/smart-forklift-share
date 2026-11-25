"""
Silent per-PC install lock, model encryption, and camera cap.  No UI, no license
text shown to the client - if the signed lock.key next to the program doesn't match
THIS PC's hardware id, or is missing, the program refuses to start.  The model file
is encrypted so a copied folder has no usable weights on another PC either.

Setup (see SETUP.md for the full walkthrough):
  1. tools/keygen.py        - once, on your dev PC only. Never ship or commit its output.
  2. --print-hwid           - on the CLIENT pc, via the built exe, to get its hardware id.
  3. tools/make_lock.py     - on your dev PC, signs a lock.key for that hardware id.
  4. tools/encrypt_model.py - on your dev PC, encrypts best.pt for that hardware id.
  5. Copy lock.key + model.enc into the client's install folder.
"""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ---- replace both of these after running tools/keygen.py -----------------------------
# Public half of the signing keypair (safe to ship / commit - it can only verify, not sign).
PUBLIC_KEY_HEX = "0000000000000000000000000000000000000000000000000000000000000000"
# Secret used (together with each PC's hardware id) to derive the model's encryption key.
# Keep this the same forever once you start shipping - changing it breaks every lock
# and encrypted model already issued. Never changes per client; the hwid does that.
APP_SECRET_HEX = "0000000000000000000000000000000000000000000000000000000000000000"
# ----------------------------------------------------------------------------------------

HARD_CAMERA_CAP = 4   # absolute ceiling, regardless of what a lock file claims


class LockError(RuntimeError):
    pass


def _run(cmd):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def get_hwid():
    """Fingerprint for this PC: BIOS/board UUID + first physical disk's serial, hashed down
    to a short id. Stable across reboots and reinstalls; changes if the client moves the
    software to different hardware (a new PC, a VM, a different disk)."""
    board = _run("(Get-CimInstance Win32_ComputerSystemProduct).UUID")
    disk = _run("(Get-CimInstance Win32_DiskDrive | Select-Object -First 1 -ExpandProperty SerialNumber)")
    raw = f"{board.strip()}|{disk.strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def derive_model_key(hwid):
    """32-byte AES key, unique per PC, from the fixed app secret + that PC's hwid."""
    secret = bytes.fromhex(APP_SECRET_HEX)
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=hwid.encode()).derive(secret)


def _verify_signature(payload_b64, sig_b64):
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEY_HEX))
        payload = base64.b64decode(payload_b64)
        sig = base64.b64decode(sig_b64)
    except (ValueError, TypeError) as e:
        raise LockError(f"malformed lock file ({e})")
    try:
        pub.verify(sig, payload)
    except InvalidSignature:
        raise LockError("signature invalid")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise LockError(f"malformed lock file ({e})")


def load_lock(path):
    """Reads and verifies lock.key. Raises LockError if missing, tampered, or the wrong PC."""
    p = Path(path)
    if not p.exists():
        raise LockError(f"missing {p.name}")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) != 2:
        raise LockError("malformed lock file")
    data = _verify_signature(lines[0], lines[1])
    if data.get("hwid") != get_hwid():
        raise LockError("hardware id mismatch")
    return data


def enforce(app_dir):
    """Call once at startup, before loading any camera or model. Returns the camera count
    to apply. Exits the process (no traceback, no 'license' wording) if the lock is
    missing or doesn't match this PC."""
    try:
        data = load_lock(Path(app_dir) / "lock.key")
    except LockError as e:
        print(f"Startup check failed: {e}", file=sys.stderr)
        sys.exit(1)
    cap = int(data.get("max_cameras", HARD_CAMERA_CAP))
    return max(1, min(cap, HARD_CAMERA_CAP))


def decrypt_model(enc_path, hwid):
    """Returns the plaintext .pt bytes. Only succeeds on the PC the model was encrypted for."""
    raw = Path(enc_path).read_bytes()
    nonce, ct = raw[:12], raw[12:]
    key = derive_model_key(hwid)
    try:
        return AESGCM(key).decrypt(nonce, ct, None)
    except Exception:
        raise LockError("model does not match this PC")


def encrypt_model(plain_path, hwid, out_path):
    """Dev-side helper used by tools/encrypt_model.py."""
    import os
    key = derive_model_key(hwid)
    nonce = os.urandom(12)
    data = Path(plain_path).read_bytes()
    ct = AESGCM(key).encrypt(nonce, data, None)
    Path(out_path).write_bytes(nonce + ct)
