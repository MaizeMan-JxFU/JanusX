@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "MODULE_DIR=%SCRIPT_DIR%module"
set "VENV_PYLAUNCH=%SCRIPT_DIR%.venv\Scripts\python.exe"

set "MODULE_NAME=%~1"
set "MODULE=%MODULE_NAME:.py=%"

type "%SCRIPT_DIR%\fig\logo.txt"

if "%~1"=="" goto show_help
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help

if exist "%MODULE_DIR%\%MODULE%.py" (
    REM loop for obtain parameter
    set "NEWCLI="
    :loop
    if "%~1"=="" goto :done
    if defined NEWCLI set "NEWCLI=%NEWCLI% "
    set "NEWCLI=%NEWCLI%%2"
    shift
    goto :loop
    :done
    REM loop finished!
    "%VENV_PYLAUNCH%" -u "%MODULE_DIR%\%MODULE%.py" %NEWCLI%
    exit /b 1
) else (
    echo Unknown module: %MODULE%
    echo Installed modules:
    for %%f in ("%MODULE_DIR%\*.py") do (
        set "filename=%%~nf"
        echo !filename!
    )
    exit /b 1
)

exit /b 0

:show_help
echo Usage: %~nx0 ^<module^> [parameter]
echo Modules:
set "output="
for %%f in ("%MODULE_DIR%\*.py") do (
    set "modname=%%~nf"
    if not "!modname:~0,1!"=="_" (
        set "output=!output! !modname!"
    )
)
echo !output!
exit /b 0