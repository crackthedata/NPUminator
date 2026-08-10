@echo off
echo Starting NPUminator Transcriber...

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found in "venv" folder.
    echo Please follow the README instructions to setup the project first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python transcriber.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Transcriber exited with an error.
    pause
) else (
    echo.
    echo Transcription complete!
    pause
)
