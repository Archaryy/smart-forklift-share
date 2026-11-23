@echo off
REM Assembles the folder to copy to the client PC:  release\
cd /d "%~dp0"
if not exist dist\gate_engine\gate_engine.exe ( echo Run build_exe.bat first & exit /b 1 )
if exist release rmdir /s /q release
mkdir release\runs\model
xcopy /e /i /q dist\gate_engine release\gate_engine
xcopy /e /i /q dist\viewer release\viewer
REM optional one-program monitor (build_app.bat): GateMonitor.exe = engine + viewer in one window
if exist dist\GateMonitor\GateMonitor.exe xcopy /e /i /q dist\GateMonitor release\GateMonitor
copy runs\v2_with_truck-2\weights\best.pt release\runs\model\best.pt
copy install_service.ps1 release\
copy config.example.yaml release\config.yaml
REM point the release config at the copied model
powershell -NoProfile -Command "(Get-Content release\config.yaml) -replace 'runs/v2_with_truck-2/weights/best.pt','runs/model/best.pt' | Set-Content release\config.yaml"
echo.
echo release\ is ready. On the client PC (admin PowerShell, in the release folder):
echo   setx /M CAM1_URL "rtsp://user:pass@IP:554/Streaming/Channels/101"   (and CAM2..CAM4)
echo   powershell -ExecutionPolicy Bypass -File install_service.ps1 -ExePath "%%CD%%\gate_engine\gate_engine.exe"
echo   then run release\viewer\viewer.exe
