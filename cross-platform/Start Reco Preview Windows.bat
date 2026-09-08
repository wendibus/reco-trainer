@echo off
cd /d "%~dp0"
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js/npm was not found. Install Node.js and start again.
  pause
  exit /b 1
)
where py >nul 2>nul
if not errorlevel 1 (
  set "RECO_PYTHON=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3 was not found. Install Python 3.11 or 3.12 and start again.
    pause
    exit /b 1
  )
  set "RECO_PYTHON=python"
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg was not found. Install FFmpeg before preparing videos.
)
%RECO_PYTHON% -c "import sys; v = sys.version_info; sys.exit(0 if v.major == 3 and v.minor in [11, 12] else 1)" >nul 2>nul
if errorlevel 1 echo Warning: this Python is not 3.11 or 3.12. Uploading and browsing videos will still work, but the Set up ML step needs 3.11 or 3.12 and may fail otherwise.
if not exist node_modules call npm install
start "Reco Local Worker" /min cmd /c "%RECO_PYTHON% local_worker.py --port 8766"
start "" cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8765/"
call npm run dev -- --host localhost --port 8765
