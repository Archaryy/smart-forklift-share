"""
Run this ONCE, on your own dev PC, before shipping the first protected release.

Generates the signing keypair and the app secret that hwlock.py needs. Writes
dev_private_key.pem (keep it safe, back it up, NEVER commit it or send it anywhere)
and prints two lines to paste into hwlock.py.

    python tools\\keygen.py
"""
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

root = Path(__file__).resolve().parent.parent
keyfile = root / "dev_private_key.pem"
if keyfile.exists():
    print(f"{keyfile} already exists - refusing to overwrite. Delete it yourself if you")
    print("really mean to start over (every lock.key and model.enc already issued breaks).")
    sys.exit(1)

key = Ed25519PrivateKey.generate()
pub_hex = key.public_key().public_bytes_raw().hex()
secret_hex = os.urandom(32).hex()

keyfile.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))

print(f"Wrote {keyfile}")
print("Back this file up somewhere safe (outside the repo) and keep it off any client PC.")
print()
print("Now open hwlock.py and replace these two lines:")
print(f'PUBLIC_KEY_HEX = "{pub_hex}"')
print(f'APP_SECRET_HEX = "{secret_hex}"')
