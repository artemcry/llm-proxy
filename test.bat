@echo off
@REM https://api.minimax.io/v1
@REM http://localhost:8087

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8087 ^| findstr LISTENING') do taskkill /PID %%a /F 2>nul
start "" pythonw d:\Projects\LLMProxy\server_bg.py

pause