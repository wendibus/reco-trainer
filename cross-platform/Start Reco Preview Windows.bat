@echo off
cd /d "%~dp0"
if not exist "ui\index.html" (
  echo The interface ^(ui folder^) is missing. Please download the release package again.
  pause
  exit /b 1
)
where py >nul 2>nul
if not errorlevel 1 (
  set "RECO_PYTHON=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3 was not found. Install Python 3.10 or newer ^(3.12 recommended^) and start again.
    pause
    exit /b 1
  )
  set "RECO_PYTHON=python"
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg was not found. Install FFmpeg before preparing videos.
)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8766/"
echo Reco Trainer runs locally at http://localhost:8766/ - close this window to stop it.
%RECO_PYTHON% local_worker.py --port 8766
