@echo off
REM Assembles the GENERIC folder to copy to the client PC:  release\
REM This build has NO model and NO lock.key yet - it is the same for every client.
REM After copying it over, get that PC's hwid (GateMonitor.exe --print-hwid) and run
REM tools\make_lock.py + tools\encrypt_model.py on YOUR dev PC, then drop the two files
REM they produce (lock.key, model.enc) into release\GateMonitor\ before the client uses it.
REM See SETUP.md for the full walkthrough.
cd /d "%~dp0"
if not exist dist\GateMonitor\GateMonitor.exe ( echo Run build_app.bat first & exit /b 1 )
if exist release rmdir /s /q release
mkdir release\GateMonitor\runs\model
REM legacy headless engine + separate viewer, if you still build those (build_exe.bat)
if exist dist\gate_engine\gate_engine.exe xcopy /e /i /q dist\gate_engine release\gate_engine
if exist dist\viewer\viewer.exe xcopy /e /i /q dist\viewer release\viewer
xcopy /e /i /q dist\GateMonitor release\GateMonitor
copy install_service.ps1 release\ >nul
copy config.example.yaml release\GateMonitor\config.yaml >nul
REM point the release config at the (not yet present) per-client encrypted model
powershell -NoProfile -Command "(Get-Content release\GateMonitor\config.yaml) -replace 'runs/forklift/v4_corrections/weights/best.pt','runs/model/model.enc' | Set-Content release\GateMonitor\config.yaml"
echo.
echo release\GateMonitor\ is ready but will NOT start yet - it has no lock.key or model yet.
echo.
echo 1. Copy release\GateMonitor\ to the client PC.
echo 2. On that PC, run:  GateMonitor.exe --print-hwid   (a popup shows the hardware id)
echo 3. Back on this dev PC:
echo      python tools\make_lock.py ^<hwid^> "client name" --out release\GateMonitor\lock.key
echo      python tools\encrypt_model.py runs\forklift\v4_corrections\weights\best.pt ^<hwid^> --out release\GateMonitor\runs\model\model.enc
echo 4. Copy those two new files (lock.key, runs\model\model.enc) onto the client PC,
echo    into the same GateMonitor folder, then run GateMonitor.exe normally.
echo.
echo Camera URLs still go in MACHINE env vars on the client PC, e.g.:
echo   setx /M CAM1_URL "rtsp://user:pass@IP:554/Streaming/Channels/101"   (and CAM2..CAM4)
