# Copy protection — how to use it

This locks a release to one PC's hardware and caps it at 4 cameras, with nothing for
the client to see or configure. Do steps 1–2 once ever. Do steps 3–7 once per client PC.

## 1. Generate your signing key (once, ever)

```
.venv\Scripts\python.exe tools\keygen.py
```

This writes `dev_private_key.pem` in the repo root and prints two lines like:

```
PUBLIC_KEY_HEX = "e646a0e1..."
APP_SECRET_HEX = "697f645f..."
```

Open [hwlock.py](hwlock.py) and paste those two lines in, replacing the two placeholder
`"000...0"` lines near the top.

**Back up `dev_private_key.pem` somewhere safe (a password manager, an encrypted USB
drive) and never commit it or copy it anywhere near a client PC.** It's already in
`.gitignore` so a normal `git add` won't pick it up — double-check before you push.

## 2. Build the generic release (once per software update)

```
build_app.bat
package_release.bat
```

This gives you `release\GateMonitor\` — the exe, viewer, config template. It has no
model and no lock yet, and **will not run yet**. This is the folder you hand to any
client; it's identical for all of them until step 5.

## 3. Get the client PC's hardware id

Copy `release\GateMonitor\` to the client PC and, in that folder, run:

```
GateMonitor.exe --print-hwid
```

A small popup shows a string like `7ccd0a38e092c997ee7c0d9e`. Write it down (or have
them screenshot/text it to you) — that's the hardware id for *that* PC.

## 4. Issue that PC's lock file (back on your dev PC)

```
.venv\Scripts\python.exe tools\make_lock.py 7ccd0a38e092c997ee7c0d9e "Acme Warehouse" --out lock.key
```

Replace the hwid with the one from step 3, and the name with whatever label helps you
remember whose PC this is (it's just for your own records — never shown to them).
This writes `lock.key`. (`--max-cameras` defaults to 4 and can't go higher.)

## 5. Encrypt the model for that PC

```
.venv\Scripts\python.exe tools\encrypt_model.py runs\forklift\v4_corrections\weights\best.pt 7ccd0a38e092c997ee7c0d9e --out model.enc
```

Same hwid again. This writes `model.enc` — unreadable weights unless decrypted on
that one PC.

## 6. Drop both files into the client's folder

Copy `lock.key` and `model.enc` onto the client PC, into:

```
release\GateMonitor\lock.key
release\GateMonitor\runs\model\model.enc
```

`config.yaml` already points `engine.model` at `runs/model/model.enc` (package_release.bat
sets that automatically).

## 7. Set the camera URLs and run it

On the client PC (admin PowerShell, in `release\GateMonitor\`):

```
setx /M CAM1_URL "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

(repeat for CAM2–CAM4), then run `GateMonitor.exe` normally. It now only starts on
this one PC, decrypts the model only in memory, and never analyses more than 4
cameras — all silently, no prompts, no license screen.

## Testing it yourself without a second PC

Steps 3–6 all work on your own dev PC too — `--print-hwid` just prints *your* dev
PC's id, and a lock/model made for your own hwid will run fine here. To see the
*failure* case, make a lock for a fake hwid (e.g. `make_lock.py deadbeef "fake" --out
lock.key`) and it will refuse to start with "Startup check failed: hardware id mismatch".

## If you move the client PC to new hardware (new disk, new motherboard)

The hwid changes, so the old lock/model stop working — re-run steps 3–6. If a client's
PC dies and they rebuild it, or they ask for more cameras, this is working as intended;
just reissue.

## What this doesn't stop

A skilled reverse engineer can still find `hwlock.py`'s logic inside the compiled exe
(it's PyInstaller bytecode, not source, but decompilable) and patch the checks out.
This stops casual copying — zipping the folder to another PC, or editing `config.yaml`
to add a 5th camera — not a determined attacker. Compiling the real entry points with
Nuitka instead of plain PyInstaller would raise that bar further; that's a separate,
optional step if you ever want it.
