@echo off
echo Starting NPUminator Folder Transcriber...

if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found in "venv" folder.
    echo Please follow the README instructions to setup the project first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python transcribe_folder.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Folder transcription exited with an error.
    pause
) else (
    echo.
    echo Folder transcription complete!
    pause
)
