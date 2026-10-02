Option Explicit

Dim fso, wshShell, scriptPath, scriptDir, pythonPath, appPath, startScriptPath

Set fso = CreateObject("Scripting.FileSystemObject")
Set wshShell = CreateObject("WScript.Shell")

' 1. Determine script directory and set current working directory
scriptPath = WScript.ScriptFullName
scriptDir = fso.GetParentFolderName(scriptPath)
wshShell.CurrentDirectory = scriptDir

' 2. Verify virtual environment Python
pythonPath = fso.BuildPath(scriptDir, ".venv\Scripts\python.exe")
If Not fso.FileExists(pythonPath) Then
    MsgBox "Virtual environment Python not found!" & vbCrLf & vbCrLf & _
           "Expected at: " & pythonPath & vbCrLf & vbCrLf & _
           "Please create the .venv environment before launching.", vbCritical, "Automated Tooth Segmentation"
    WScript.Quit 1
End If

' 3. Verify app.py
appPath = fso.BuildPath(scriptDir, "app.py")
If Not fso.FileExists(appPath) Then
    MsgBox "Application entry point app.py not found!" & vbCrLf & vbCrLf & _
           "Expected at: " & appPath, vbCritical, "Automated Tooth Segmentation"
    WScript.Quit 1
End If

' 4. Verify tools\start_app.py
startScriptPath = fso.BuildPath(scriptDir, "tools\start_app.py")
If Not fso.FileExists(startScriptPath) Then
    MsgBox "Launcher script tools\start_app.py not found!" & vbCrLf & vbCrLf & _
           "Expected at: " & startScriptPath, vbCritical, "Automated Tooth Segmentation"
    WScript.Quit 1
End If

' 5. Check if Port 8501 is occupied
Dim checkPortCmd, portExitCode
checkPortCmd = """" & pythonPath & """ -c ""import socket, sys; s=socket.socket(); res=s.connect_ex(('127.0.0.1', 8501)); s.close(); sys.exit(0 if res==0 else 1)"""
portExitCode = wshShell.Run(checkPortCmd, 0, True)

If portExitCode = 0 Then
    MsgBox "Port 8501 is already in use by another process or application!" & vbCrLf & vbCrLf & _
           "Streamlit cannot start on default port 8501." & vbCrLf & vbCrLf & _
           "Please stop the process running on port 8501 before launching.", vbExclamation, "Automated Tooth Segmentation - Port Conflict"
    WScript.Quit 1
End If

' 6. Launch start_app.py in background without showing CMD window (0 = hidden, False = do not wait)
Dim launchCmd
launchCmd = """" & pythonPath & """ """ & startScriptPath & """"
wshShell.Run launchCmd, 0, False
