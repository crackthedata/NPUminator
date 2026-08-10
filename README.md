# NPUminator
Locally deployed video transcription with **speaker diarization**. It runs Whisper for text and **Pyannote** for “who spoke when,” and picks a sensible accelerator automatically: **NVIDIA / AMD (CUDA)**, **Apple Silicon (MPS)**, **Intel NPU / OpenVINO GPU**, or **CPU**.

## Introduction

**transcriber.py** transcribes video (or audio) into a **timestamped transcript with speaker labels**. You choose a file and an output `.txt` path; the script converts audio to 16 kHz WAV, runs **diarization**, then transcribes each speech segment.

- **Transcription (Whisper)**  
  - **Default (`WHISPER_BACKEND=auto`)**: If **CUDA** or **Apple MPS** is available, Whisper runs through **PyTorch + Hugging Face** on the GPU (model from the Hub, default `openai/whisper-small`). Otherwise it uses **OpenVINO GenAI** from the local **`whisper/`** export and tries devices in order **NPU → GPU → CPU** (Intel-focused path).  
  - You can force **OpenVINO** (`WHISPER_BACKEND=openvino`) or **PyTorch** (`WHISPER_BACKEND=torch`) — see [Environment variables](#environment-variables) below.

- **Diarization (Pyannote)** uses **CUDA** if available, else **MPS** on Mac, else **CPU**.

You can set the number of speakers in the dialog (default **2**; range **1–50**). Cancel leaves it empty so Pyannote **auto-detects** speaker count.

Output example: `[0.0s - 5.2s] SPEAKER_00: Hello everyone.`

## Required setup (all machines)

1. **Install FFmpeg and add it to PATH**  
   - [FFmpeg download](https://www.ffmpeg.org/download.html) (Windows builds: e.g. gyan.dev or BtbN).  
   - Unzip and add the `bin` folder to **PATH**; reboot if needed.

2. **Microsoft Visual C++ Redistributable** (Windows, if installers prompt):  
   [Latest VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

3. **GPU with NVIDIA CUDA (optional)**  
   Install a **CUDA-enabled PyTorch** build (see [pytorch.org](https://pytorch.org/)) so `torch.cuda.is_available()` is true. The generic `pip install -r requirements.txt` wheel is often CPU-only; for GPU transcription, install the matching CUDA wheel first, then the rest of the requirements.

## Optional: Intel NPU / OpenVINO GPU (Windows)

Only needed if you rely on the **OpenVINO** Whisper path (no CUDA/MPS, or `WHISPER_BACKEND=openvino`):

1. **Intel NPU** (Core™ Ultra, etc.): enable in firmware if applicable.  
2. [Intel® NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)  
3. [Intel® Arc™ Graphics - Windows](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html) (for OpenVINO **GPU** plugin on Intel graphics)

## Clone, setup environment, and download model

1. **Clone the repository**

   **Linux (bash):**
   ```
   git clone https://github.com/crackthedata/NPUminator.git
   cd NPUminator
   ```

   **Windows (CMD):**
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

   **Windows (CMD):**
   ```
   venv\Scripts\activate
   ```

4. **Install dependencies**  
   Run with the virtual environment activated:
   ```
   pip install -r requirements.txt
   ```

5. **Whisper model(s)**  

   - **PyTorch path (CUDA / MPS, or `WHISPER_BACKEND=torch`)**  
     No OpenVINO export required. The first run downloads **`WHISPER_HF_MODEL`** (default `openai/whisper-small`) from Hugging Face. Use a larger checkpoint if you want (e.g. `openai/whisper-medium`).

   - **OpenVINO path (`WHISPER_BACKEND=openvino`, or `auto` on a machine without CUDA/MPS)**  
     Export into a folder named **`whisper`** in the project root (or set `WHISPER_MODEL_PATH`):
     ```
     optimum-cli export openvino --model openai/whisper-small --trust-remote-code whisper
     ```
     Larger model example:
     ```
     optimum-cli export openvino --model openai/whisper-medium --trust-remote-code whisper-medium
     ```
     Then set `WHISPER_MODEL_PATH=whisper-medium` in `.env` if you did not use `whisper`.

     OpenVINO tries **`WHISPER_DEVICE`** first if set (`NPU`, `GPU`, or `CPU`), then falls through the rest of **NPU → GPU → CPU** until one works.

     **Windows:** If `optimum-cli` is not found, use e.g.  
     `.\venv\Scripts\optimum-cli.exe export openvino --model openai/whisper-small --trust-remote-code whisper`

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `WHISPER_BACKEND` | `auto` | `auto` → PyTorch if CUDA or MPS, else OpenVINO. `torch` / `openvino` forces that stack. |
| `WHISPER_HF_MODEL` | `openai/whisper-small` | Hugging Face model id for **PyTorch** Whisper. |
| `WHISPER_MODEL_PATH` | `whisper` | Folder with **OpenVINO** export. |
| `WHISPER_DEVICE` | *(empty)* | OpenVINO only: prefer `NPU`, `GPU`, or `CPU` before trying other devices. |
| `HF_TOKEN` | — | Required for Pyannote (set in `.env`). |

6. **Create a Hugging Face access token and accept Pyannote license terms**  
   The transcriber uses Pyannote for speaker diarization; Pyannote models require a Hugging Face token and accepted license.

   - Go to [Hugging Face → Access Tokens](https://huggingface.co/settings/tokens) and create a token (read access is enough).
   - Open the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model page and **Accept** the license terms if you haven’t already. Do the same for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) if the pipeline prompts you to.
   - In the project root, create or edit a `.env` file (you can copy `.env.example`) and add your token:
     ```
     HF_TOKEN=your_token_here
     ```
     Replace `your_token_here` with your actual token. The script loads this via `python-dotenv` and uses it for the Pyannote pipeline. You can also configure other settings in this file; check `.env.example` for details.

## Run the transcriber

There are a few ways to run the transcription pipeline:

### 1. Using the Single Video Batch File (Windows)
Double-click `run_transcriber_single.bat`. It will automatically activate the virtual environment and run `transcriber.py`.
The following will happen:
1. An explorer box will open to select the video file to transcribe.
2. An explorer box will open to save the transcript. It defaults to the video's filename appended with `.txt`, which you can change.
3. A dialog box will open for you to select how many speakers should be identified. If you don't know, leave it empty/cancel, and the pipeline will try to identify the speaker count automatically.

### 2. Batch Processing a Folder of Videos (Windows)
Double-click `run_transcriber_folder.bat`. It will:
1. Prompt you to select a folder containing `.mp4` video files.
2. Ask for a delay (in seconds) to wait between processing each video (useful to prevent your machine from overheating).
3. Automatically process every `.mp4` file in the folder, using auto-detection for the number of speakers and saving the output as `<video_name>.txt`.

### 3. Command Line Execution (Advanced)
If you want to bypass the graphical prompts, you can provide arguments directly to the script:
```bash
# Activate your environment first, then:
python transcriber.py --video "my_video.mp4" --output "my_video.mp4.txt" --speakers "auto"
```
Arguments:
- `--video`: Path to the input video.
- `--output`: Path to the output transcript text file.
- `--speakers`: Integer number of speakers, or `auto` to let the model decide.

## Future work

1. Optional ROCm / explicit device overrides for edge cases.  
2. Evaluate other local ASR models alongside Whisper.
