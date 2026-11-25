@echo off
REM Builds the ONE-program monitor (detection engine + viewer):  dist\GateMonitor\GateMonitor.exe
REM It is a folder (GateMonitor.exe + _internal\) rather than a single packed file on purpose:
REM a packed single file would unpack several GB of PyTorch/CUDA into %TEMP% on every start.
REM Takes ~10+ minutes, output is several GB.
cd /d "%~dp0"
REM call the venv's python directly: 'activate' is broken (venv was created at an old path)
set PY=.venv\Scripts\python.exe
%PY% -m pip install pyinstaller || goto :err

%PY% -m PyInstaller --noconfirm --onedir --noconsole --name GateMonitor ^
  --collect-all ultralytics --collect-all torchvision --collect-all cryptography --copy-metadata ultralytics ^
  --hidden-import lap --hidden-import gate_engine --hidden-import hwlock gate_app.py || goto :err

echo.
echo Built:  dist\GateMonitor\GateMonitor.exe
echo Put next to it:  config.yaml  and the model file named in it (e.g. runs\model\best.pt)
echo Test:   dist\GateMonitor\GateMonitor.exe --config "%CD%\config.local.yaml"
exit /b 0
:err
echo BUILD FAILED
exit /b 1
