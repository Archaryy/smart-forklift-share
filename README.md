# Loading-bay pedestrian detection and gate signal

Detects forklifts, trucks and people on IP-camera video (YOLO26) and raises a signal when a
**pedestrian stands in the loading bay** behind a truck. People inside/on a truck or pallet
are ignored (only feet on the floor area count), and forklift operators are paired with their forklift.

> **Status:** prototype. Not safety-rated. If people could be hurt, the gate needs independent
> certified protection (light curtain / safety relay); this system is an extra layer.

## How it works
1. Model classes: `0 forklift`, `1 person`, `2 truck` (`models/best.pt`).
2. A **bay** is a rectangle attached to the rear of each detected truck (`rear`, `depth`, `lateral` settings per camera).
3. A person whose feet are inside a bay, and who is not paired with a forklift, counts as "in bay".
4. After `close_seconds` the camera's alarm output goes **HIGH** (and the PC beeps); after `open_seconds` clear it goes **LOW**.
   Camera lost / start-up / stop also give HIGH. Re-sent every second.

## Layout
| Path | Purpose |
|---|---|
| `test_video.py` | Single-camera/video test app with window (draw zone, auto bay, seek, relay, alarm) |
| `gate_engine.py` | Headless multi-camera engine (1 thread + model per camera, GPUs shared) |
| `viewer.py` | Desktop window: tiled live view of all cameras from the engine |
| `detect_http_camera.py` | Same detection from camera HTTP snapshots (slow, ~5 FPS; testing only) |
| `gate_relay.py` | Alternative fail-safe relay module (permit-to-open polarity); not wired into the engine |
| `config.example.yaml` / `config.local.yaml` | Real 4-camera template / local test with video files |
| `tools/` | Training, fine-tuning, dataset scripts, camera test |
| `setup_client.bat`, `install_service.ps1`, `build_exe.bat`, `package_release.bat` | Client install / packaging |

## Quick start (development PC, Windows, NVIDIA GPU)
```
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Local test with video files as fake cameras (two terminals):
```
python gate_engine.py --config config.local.yaml
python viewer.py
```
Single camera with a window:
```
$env:CAM_URL = "rtsp://USER:PASSWORD@CAMERA_IP:554/Streaming/Channels/102"
python tools\test_camera.py            # check the stream first
python test_video.py --camera --relay
```
`test_video.py` keys: `S` rear side, `[ ]` depth, `, .` width, `A` auto/manual bay, `T` truck boxes, `M` mute,
`+/-` person confidence, `J/L` or arrows seek (files), `Z` draw manual zone, `Q` quit.

## Install on the client PC
1. Copy `config.example.yaml` to `config.yaml`; set each camera's `gpu`, `rear` and relay.
2. Store camera URLs as machine variables (admin PowerShell), never in files:
   `setx /M CAM1_URL "rtsp://USER:PASSWORD@IP:554/Streaming/Channels/101"` (CAM2..CAM4 likewise).
3. `powershell -ExecutionPolicy Bypass -File install_service.ps1` registers a Task Scheduler task
   (start at boot, auto-restart). Use `-AtLogon` if CUDA fails under the SYSTEM account.
4. Run `python viewer.py` (or `viewer.exe`) for the monitor window. Logs: `logs/engine.log`.

Executables (experimental): `build_exe.bat`, then `package_release.bat`, copy `release\` to the client.

## Camera alarm output
Triggered over ISAPI: `PUT /ISAPI/System/IO/outputs/1/trigger` with `<IOPortData><outputState>high|low</outputState></IOPortData>`
(HTTP digest auth, operator-level user). Set the output to manual mode in the camera. Check that your camera model
has an alarm output and that the relay rating suits the gate controller (use an interposing relay).
If the program crashes the output keeps its last state: add a hardware timeout/watchdog for real use.

## Training
Dataset: Kaggle + Ultralytics subset + site frames labelled in Roboflow (`pedestrian` -> `person`).
`python tools\finetune.py <run_name>` fine-tunes from the current weights (imgsz 960, site frames oversampled).
Known weak spot: small/far people in red uniforms next to red trucks; collect more site-camera person labels.
On Windows, `WinError 1455` means the paging file is too small: raise it or lower dataloader workers.

## Not in the repo
Datasets, videos, runs, `config.yaml`, logs, site-specific zone/bay files (see `.gitignore`).
