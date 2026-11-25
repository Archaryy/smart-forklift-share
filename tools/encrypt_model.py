"""
Run on your DEV PC, after you have the client's hwid (from --print-hwid on their PC).
Encrypts best.pt so it only decrypts on that one PC's hardware.

    python tools\\encrypt_model.py runs\\forklift\\v4_corrections\\weights\\best.pt <hwid> --out model.enc
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hwlock

ap = argparse.ArgumentParser()
ap.add_argument("model_path")
ap.add_argument("hwid", help="output of --print-hwid on the client PC")
ap.add_argument("--out", default="model.enc")
args = ap.parse_args()

hwlock.encrypt_model(args.model_path, args.hwid, args.out)
print(f"Wrote {args.out} - only decrypts on hwid {args.hwid}")
print("Copy it into that client's install folder (e.g. runs\\model\\model.enc) and point")
print("config.yaml's engine.model at it.")
