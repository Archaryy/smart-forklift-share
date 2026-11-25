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

## 7. Set the camera URLs

You have two ways to do this. Either works; pick one.

**Easiest — bake them into `config.yaml` before you send the folder.** Open
`release\GateMonitor\config.yaml` on your dev PC and, for each camera, replace
`url_env: CAM1_URL` with a direct `url:`, e.g.:

```yaml
- {id: dock1, name: "Dock 1", url: "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101", gpu: 0, rear: bottom, relay: {enabled: true, output: 1}}
```

Do this for all 4 cameras and the folder needs **zero commands run on the client PC** —
copy it over, double-click `GateMonitor.exe`, done. (`config.yaml` is already
per-client and gitignored, same as `lock.key`/`model.enc` — this doesn't change what's
exposed, just where you type it.)

**Original way — leave `url_env` as-is and set a machine env var on the client PC**
(admin PowerShell, in `release\GateMonitor\`):

```
setx /M CAM1_URL "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101"
```

(repeat for CAM2–CAM4). Use this if you'd rather not have the password sitting in a
file you handed someone, even though it's no more or less readable than the
registry/env-var either way.

Either way, once it's set, run `GateMonitor.exe` normally. It now only starts on this
one PC, decrypts the model only in memory, and never analyses more than 4 cameras —
all silently, no prompts, no license screen.

## What to actually hand the client

A single folder, `release\GateMonitor\` — not a single file (see the FAQ below for
why). Once steps 1–7 are done, it should contain:

```
GateMonitor\
  GateMonitor.exe
  _internal\             (DLLs, Python runtime, etc. - don't touch)
  config.yaml             (camera URLs already filled in, if you used the easy way)
  lock.key                (from step 4)
  runs\model\model.enc    (from step 5)
```

Zip that whole folder and send the zip, or copy it over directly (USB drive, remote
session, whatever you'd normally use). Nothing else from the repo needs to go with it
— no source, no training data, no `dev_private_key.pem`.

## FAQ: can this just be one .exe file?

Not a good idea, and here's why: `build_app.bat`'s own comment explains that PyInstaller
is deliberately built as `--onedir` (a folder) rather than `--onefile` (one packed exe),
because a packed exe has to unpack several GB of PyTorch/CUDA into a temp folder
**every time it starts** — turning a ~2 second launch into a slow one, every launch,
forever. That trade was made before any of this copy-protection work and still holds.

What *is* true: the environment variables were never "compiled in" anywhere — they're
just read from the OS at startup, or (per the easy option above) straight out of
`config.yaml`. Either way they live in a plain file/registry entry next to the exe, not
baked into the binary.

If having literally one file matters enough to accept a slower startup, it's possible
to go further and build a separate exe *per client* with that client's `lock.key`,
`model.enc`, and `config.yaml` embedded into a `--onefile` PyInstaller build — true
one-file delivery, at the cost of a ~10 minute rebuild for every client (and a slower
launch each time it runs). Say if you want that and I'll build it out; it's a real
option, just a different trade-off than what's here now.

## Testing it yourself without a second PC

Steps 3–6 all work on your own dev PC too — `--print-hwid` just prints *your* dev
PC's id, and a lock/model made for your own hwid will run fine here. To see the
*failure* case, make a lock for a fake hwid (e.g. `make_lock.py deadbeef "fake" --out
lock.key`) and it will refuse to start with "Startup check failed: hardware id mismatch".

## If you move the client PC to new hardware (new disk, new motherboard)

The hwid changes, so the old lock/model stop working — re-run steps 3–6. If a client's
PC dies and they rebuild it, or they ask for more cameras, this is working as intended;
just reissue.

## If a copy turns up somewhere it shouldn't (watermarking)

Every `lock.key` already carries the client label you gave it in step 4, readable
without your private key (that's intentional — the lock is signed, not secret). If you
ever find a copy of the software running somewhere unexpected, grab its `lock.key` and
run:

```
.venv\Scripts\python.exe tools\read_lock.py path\to\lock.key
```

It prints the client name, hwid, camera cap and issue date straight off the file, so
you know whose install it came from.

## Harder to reverse engineer: Nuitka instead of PyInstaller (shelved)

`build_app.bat` ships your code as PyInstaller bytecode, which decompiles easily.
[build_app_nuitka.bat](build_app_nuitka.bat) was meant to be an alternative that
compiles it to native code instead, making `hwlock.py`'s checks much harder to find
and patch out. **Tried it on this machine and it hit a real toolchain bug**: Nuitka's
auto-downloaded MinGW64/GCC 15.2.0 compiler can't find one of its own header files
while compiling Pillow's native modules (pulled in via `ultralytics`) — the header
genuinely exists on disk, so this is a packaging/compatibility bug in that specific
compiler download, not anything in this project's code. Fixing it for real means
installing a different MinGW-w64 toolchain by hand or installing Visual Studio Build
Tools so Nuitka uses MSVC instead — bigger, slower installs not worth it unless you
specifically want to revisit this. The working PyInstaller build doesn't need any of
it. The lock/model workflow (steps 3–7 above) would be identical either way if you
ever do get it building.

## What this doesn't stop

A skilled reverse engineer can still find `hwlock.py`'s logic inside the compiled exe
and patch the checks out — Nuitka raises that bar, but doesn't remove it. This stops
casual copying — zipping the folder to another PC, or editing `config.yaml` to add a
5th camera — not a determined attacker. [CLIENT_LICENSE_CLAUSE.md](CLIENT_LICENSE_CLAUSE.md)
has a sample license paragraph for the part the technical lock can't cover.
