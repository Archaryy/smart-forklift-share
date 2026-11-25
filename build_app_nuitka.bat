@echo off
REM EXPERIMENTAL alternative to build_app.bat. Compiles gate_app.py (and everything it
REM imports, including hwlock.py) with Nuitka instead of plain PyInstaller.
REM
REM Why: PyInstaller ships your .py files as ordinary Python bytecode inside the exe -
REM readable with a decompiler in minutes. Nuitka translates it to C and compiles that,
REM which is much harder to turn back into readable source, so the hardware-lock checks
REM in hwlock.py are harder to find and patch out.
REM
REM Trade-offs vs build_app.bat:
REM   - First build is much slower (expect 20-60+ min with torch/ultralytics; a plain
REM     PyInstaller build is ~10 min per build_app.bat's own comment).
REM   - Bigger download the first time (Nuitka fetches a C compiler if you don't have one).
REM   - Large ML libraries (torch, ultralytics, cv2) sometimes need extra --include /
REM     --nofollow-import-to flags to compile cleanly. If this fails, the error names the
REM     missing module - add --include-module=<name> (or --nofollow-import-to=<name> for
REM     something it shouldn't try to compile) and rerun. Expect to iterate once or twice
REM     the first time you try this on this project.
REM   - Not yet actually run end-to-end in this repo - try it on a machine you can watch,
REM     before relying on it for a client build.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
%PY% -m pip install nuitka || goto :err

%PY% -m nuitka --standalone --windows-console-mode=disable --output-dir=dist_nuitka ^
  --mingw64 --assume-yes-for-downloads --disable-cache=ccache ^
  --include-module=hwlock --include-module=gate_engine ^
  --include-package=ultralytics --include-package=torchvision --include-package=cryptography ^
  --include-package-data=ultralytics ^
  gate_app.py || goto :err

echo.
echo Built:  dist_nuitka\gate_app.dist\gate_app.exe
echo Rename/copy that folder the same way package_release.bat does for dist\GateMonitor,
echo then follow SETUP.md from step 3 - the lock/model workflow is identical either way.
exit /b 0
:err
echo BUILD FAILED - see the error above. Large ML packages often need one more
echo --include-module or --nofollow-import-to flag; add it and rerun.
exit /b 1
