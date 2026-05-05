# Camera Troubleshooting Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Camera Troubleshooting Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill all Python processes
Write-Host "[1/4] Stopping all Python processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Write-Host "      Done!" -ForegroundColor Green

# Step 2: Check for camera apps
Write-Host "[2/4] Checking for camera applications..." -ForegroundColor Yellow
$cameraApps = @("Camera", "Skype", "Teams", "Zoom", "Discord")
foreach ($app in $cameraApps) {
    $process = Get-Process | Where-Object {$_.ProcessName -like "*$app*"}
    if ($process) {
        Write-Host "      Found: $($process.ProcessName) - Stopping..." -ForegroundColor Yellow
        $process | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "      Done!" -ForegroundColor Green

# Step 3: Wait for camera to be released
Write-Host "[3/4] Waiting for camera to be released..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Write-Host "      Done!" -ForegroundColor Green

# Step 4: Test camera
Write-Host "[4/4] Testing camera..." -ForegroundColor Yellow
Write-Host ""

$testScript = @"
import cv2
import sys

cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print('SUCCESS: Camera is working!')
        print(f'Resolution: {frame.shape[1]}x{frame.shape[0]}')
        sys.exit(0)
    else:
        print('ERROR: Camera opened but cannot read frames')
        sys.exit(1)
else:
    print('ERROR: Cannot open camera')
    print('Try:')
    print('  1. Check if camera is connected')
    print('  2. Close other apps using camera (Zoom, Teams, etc.)')
    print('  3. Try camera ID 1: --camera 1')
    sys.exit(1)
cap.release()
"@

$testScript | python -

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Troubleshooting Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If camera still doesn't work:" -ForegroundColor Yellow
Write-Host "  1. Close Zoom, Teams, Skype, or other video apps" -ForegroundColor White
Write-Host "  2. Try: python enhanced_detection_system.py ... --camera 1" -ForegroundColor White
Write-Host "  3. Restart your computer" -ForegroundColor White
Write-Host ""
