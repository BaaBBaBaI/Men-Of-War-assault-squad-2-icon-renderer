@echo off
cd /d "%~dp0"
py -3.9 app.py %*
if errorlevel 1 python app.py %*
