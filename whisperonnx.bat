@echo off
title Whisper ONNX Transcriber
echo =======================================================
echo        Starting Whisper ONNX Transcriber...
echo =======================================================
echo.

:: Step 1: Tell the script to go to your specific folder first
cd /d C:\Yourfolder

:: Step 2: Activate your virtual environment
call venv\Scripts\activate

:: Step 3: Run the Python script
python 31.py

:: Step 4: Automatically open your default web browser to the Gradio UI
start http://127.0.0.1:7860

:: Step 5: Keep the window open if there's an error so you can read it

pause
