# NPUminator
Locally deployed video transcription tool using OpenVINO, tailored to use Intel NPU for low power, high performance transcription and diarization.

## Introduction

**transcriber.py** is the main script: it transcribes video (or audio) into text with **speaker labels**. You pick a video file and an output path; the script converts the track to 16 kHz WAV, runs **speaker diarization** to determine “who spoke when,” then transcribes each segment.

- **Transcription** is done by a **local Whisper model** (OpenVINO) running on the **Intel NPU**, so inference stays on-device and power-efficient.
- **Speaker diarization** (“who spoke when”) is handled by **Pyannote** and runs on the **CPU**.

The result is a timestamped transcript saved as a `.txt` file (e.g. `[0.0s - 5.2s] SPEAKER_00: Hello everyone.`).

## Required setup

Before running NPUminator, complete these steps:

1. **Ensure hardware is NPU-enabled**  
   The app targets Intel® Core™ Ultra series processors (and compatible NPU-enabled hardware). Confirm your system has an NPU and that it is enabled in BIOS/firmware if applicable.

2. **Install Intel NPU Driver for Windows** (if not already installed)  
   Download and install from:  
   [Intel® NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)

3. **Install Intel Graphics Driver** (if not already installed)  
   Download and install from:  
   [Intel® Arc™ Graphics - Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html)

4. **Install Microsoft Visual C++ Redistributable** (if not already installed)  
   Download and install the latest supported version for your architecture (x64 recommended):  
   [Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

5. **Install FFmpeg and add it to PATH**  
   - Download FFmpeg from [FFmpeg download](https://www.ffmpeg.org/download.html) (e.g. a Windows build from gyan.dev or BtbN).  
   - Unzip it to a local folder (e.g. `C:\ffmpeg`).  
   - Add that folder’s `bin` directory to your system **PATH** environment variable.  
   - A reboot may be required for PATH changes to take effect.

## Clone, setup environment, and download model

1. **Clone the repository**

   **Linux (bash):**
   ```
   git clone https://github.com/crackthedata/NPUminator.git
   cd NPUminator
   ```

   **Windows (PowerShell or CMD):**
   ```
   git clone https://github.com/crackthedata/NPUminator.git
   cd NPUminator
   ```

2. **Create a virtual environment**  
   ```
   python -m venv venv
   ```

3. **Activate the environment**

   **Linux (bash):**
   ```
   source venv/bin/activate
   ```

   **Windows (PowerShell or CMD):**
   ```
   venv\Scripts\activate
   ```

4. **Install dependencies**  
   Run with the virtual environment activated:
   ```
   pip install -r requirements.txt
   ```

5. **Download the required Whisper OpenVINO model**  
   Run with the virtual environment activated:
   ```
   optimum-cli export openvino --model openai/whisper-base --trust-remote-code whisper-base-ov
   ```
   This exports the `openai/whisper-base` model to the `whisper-base-ov` directory for use with OpenVINO/NPU.
