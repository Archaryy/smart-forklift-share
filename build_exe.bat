@echo off
REM Builds folder-style executables (NOT one single file: a single file would unpack
REM several GB of PyTorch/CUDA on every start).  Takes ~10+ minutes, output is several GB.
cd /d "%~dp0"
call .venv\Scripts\activate || goto :err
pip install pyinstaller || goto :err

pyinstaller --noconfirm --onedir --name gate_engine ^
  --collect-all ultralytics --copy-metadata ultralytics ^
  --hidden-import lap gate_engine.py || goto :err
pyinstaller --noconfirm --onedir --name viewer viewer.py || goto :err

echo.
echo Built:  dist\gate_engine\gate_engine.exe   dist\viewer\viewer.exe
echo Test:   dist\gate_engine\gate_engine.exe --config config.local.yaml
echo Then:   package_release.bat
exit /b 0
:err
echo BUILD FAILED
exit /b 1
