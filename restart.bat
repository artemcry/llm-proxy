@echo off
set PORT=8087
for /f "tokens=1,2 delims==" %%a in (.env) do if "%%a"=="SERVER_PORT" set PORT=%%b
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT% ^| findstr LISTENING') do taskkill /PID %%a /F 2>nul
wmic process where "commandline like '%%server_bg.py%%'" call terminate 2>nul
echo LLMProxy stopped.


timeout /t 1 /nobreak >nul

start "" pythonw d:\Projects\LLMProxy\server_bg.py
timeout /t 4 /nobreak >nul
echo "Exit 4s.."
exit

