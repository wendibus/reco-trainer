@echo off
rem Tests RF-DETR training on an AMD graphics card (ROCm). Installs into a separate folder only.
where py >nul 2>nul
if %errorlevel%==0 (py -3 "%~dp0amd_training_probe.py") else (python "%~dp0amd_training_probe.py")
pause
