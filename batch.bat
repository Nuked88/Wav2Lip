@echo off


IF NOT EXIST venv (
    python -m venv venv
    call %~dp0\venv\Scripts\activate.bat
) ELSE (
    echo venv folder already exists, skipping creation...
)
call %~dp0\venv\Scripts\activate.bat
set PYTHON="%~dp0\venv\Scripts\Python.exe"

echo venv %PYTHON%
%PYTHON% -m pip install -r cuda_requirements.txt
%PYTHON% -m pip install -r requirements.txt

%PYTHON% "%~dp0\download_models.py"

IF "%~1"=="" (
    echo "Enter the path to your video and audio files (e.g., C:\path\to\your\videos-and-audio):"
    set /p FOLDER_PATH=
) ELSE (
    set FOLDER_PATH=%~1
)

echo Path: %FOLDER_PATH%

%PYTHON% batch_inference.py --checkpoint_path "%~dp0\checkpoints\wav2lip_gan.pth" --folder_path "%FOLDER_PATH%"

echo.
echo Exiting.
pause
