@echo off
echo ========================================
echo Enhanced YOLO Detection System
echo ========================================
echo.
echo Checking for running Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
echo.
echo Starting system...
cd enhanced_yolo_system
python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt --camera 0
cd ..
pause
