@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0dashboard\app.py" --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 pause
