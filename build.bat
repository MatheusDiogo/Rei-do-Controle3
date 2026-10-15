@echo off
REM Gera o executavel dist\ReiDoControle.exe
python -m pip install -r requirements.txt
python -m PyInstaller --noconfirm --onefile --windowed --name ReiDoControle app.py
echo.
echo Executavel gerado em dist\ReiDoControle.exe
pause
