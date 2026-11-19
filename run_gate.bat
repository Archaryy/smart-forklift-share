@echo off
REM Starts the forklift/pedestrian gate system with the IP camera.
REM The RTSP URL (with password) comes from the CAM_URL environment variable.
REM Set it once for the Windows user with:
REM   setx CAM_URL "rtsp://USER:PASSWORD@CAMERA_IP:554/Streaming/Channels/102"
REM then open a NEW terminal / re-login.
cd /d "%~dp0"
call .venv\Scripts\activate
python test_video.py --camera --model runs\v2_with_truck-2\weights\best.pt
pause
