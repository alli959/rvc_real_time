@echo off
:: WSL2 Port Forwarding for MorphVox
:: Double-click to run (will prompt for admin)

echo Requesting administrator privileges...
powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0wsl_port_forward.ps1\"' -Verb RunAs"
