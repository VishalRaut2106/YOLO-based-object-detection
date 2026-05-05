@echo off
echo ========================================
echo Ollama Universal Detector (Local AI)
echo Detects: Glasses, Headphones, Pens, etc.
echo ========================================
echo.
echo Make sure Ollama is running: ollama serve
echo.
cd experimental_detectors
python ollama_universal_detector.py
cd ..
