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
    msg * "Running with the runtime Python, Please wait."
     "runtime/python.exe" rvcgui-en.py --pycmd "runtime/python.exe"
      pause
) else (
    REM Runtime folder does not exist, so run the file using the system Python
    echo Running with the system Python.
    msg * "Running with the system Python."
    python.exe rvcgui-en.py --pycmd python.exe
pause
)