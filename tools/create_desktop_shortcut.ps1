# PowerShell script to create a Windows Desktop Shortcut for Tooth Segmentation app
$ErrorActionPreference = "Stop"

# Determine project root directory relative to script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = (Resolve-Path (Join-Path $ScriptDir "..")).Path

$VbsPath = Join-Path $ProjectDir "run_app.vbs"
$IconPath = Join-Path $ProjectDir "assets\tooth_segmentation.ico"

# Verify VBS launcher exists
if (-not (Test-Path $VbsPath)) {
    Write-Error "Target launcher script run_app.vbs not found at: $VbsPath"
    exit 1
}

# Get current user's Desktop directory
$DesktopDir = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$ShortcutPath = Join-Path $DesktopDir "Automated Tooth Segmentation.lnk"

# Create COM WScript.Shell instance
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)

# Configure shortcut settings
$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"$VbsPath`""
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description = "Automated Tooth Segmentation Streamlit Application"

# Apply project icon if present
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = "$IconPath, 0"
}

$Shortcut.Save()
Write-Host "Desktop shortcut created successfully!"
Write-Host "Shortcut Path: $ShortcutPath"
Write-Host "Target: $VbsPath"
