@echo off
echo Building Heart Disease Prediction System Executable...
python -m pip install pyinstaller
python -m PyInstaller --clean --noconfirm --onefile --windowed --name=HeartDiseasePrediction heart_disease_app.py
echo.
echo If the build was successful, the executable can be found in the 'dist' folder.
pause 