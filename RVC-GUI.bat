@echo off

REM Get the path of the script's directory
set "scriptDir=%~dp0"

REM Set the path to the Python runtime folder
set "runtimeFolder=%scriptDir%runtime"

REM Check if the runtime folder exists

REM Check if the runtime folder exists
if exist "%runtimeFolder%\python.exe" (
    REM Runtime folder exists, so run the file using the runtime Python
    echo Running with the runtime Python, Please wait.
    msg * "RVC GUI aciliyor, lutfen biraz bekleyin.."
     "runtime/python.exe" rvcgui.py --pycmd "runtime/python.exe"
      exit
) else (
    REM Runtime folder does not exist, so run the file using the system Python
    echo Running with the system Python.
    msg * "RVC GUI aciliyor, lutfen biraz bekleyin.."
    python.exe rvcgui.py --pycmd python.exe
exit
)