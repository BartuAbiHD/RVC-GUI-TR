::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAnk
::fBw5plQjdG8=
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSTk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCyDJGyX8VAjFDlVXhCXH2KsB6YgzO3o5P6IsnEvYsFyX7ry5oa4A60032yqcI4otg==
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
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