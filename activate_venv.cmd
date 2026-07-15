@echo off
set "VENV_DIR=%~dp0.venv\Scripts"
if exist "%VENV_DIR%\activate.bat" (
    call "%VENV_DIR%\activate.bat"
) else (
    echo Virtual environment not found at %VENV_DIR%
    exit /b 1
)
