# Enhanced YOLO Detection System Launcher
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Enhanced YOLO Detection System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if camera is available
Write-Host "Checking camera availability..." -ForegroundColor Yellow

# Try to release any stuck camera processes
$pythonProcesses = Get-Process | Where-Object {$_.ProcessName -like "*python*"}
if ($pythonProcesses) {
    Write-Host "Found running Python processes. Stopping them..." -ForegroundColor Yellow
    $pythonProcesses | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Write-Host "Starting Enhanced YOLO System..." -ForegroundColor Green
Write-Host ""

Set-Location enhanced_yolo_system
python enhanced_detection_system.py -c ../yolov3.cfg -w ../yolov3.weights -cl ../yolov3.txt --camera 0
Set-Location ..

Write-Host ""
Write-Host "System stopped." -ForegroundColor Yellow
