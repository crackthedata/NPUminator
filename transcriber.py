import os
from typing import Any, Callable, List, Optional, Tuple

import librosa
import torch
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
import soundfile as sf

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Configuration Variables (set in .env or the shell; see README)
#
# HF_TOKEN
#   Required. Hugging Face read token for Pyannote diarization.
#
# WHISPER_MODEL_PATH
#   Folder with an OpenVINO Whisper export (optimum-cli). Default: "whisper"
#   Example: WHISPER_MODEL_PATH=C:\models\whisper-medium-ov
#
# WHISPER_DEVICE  (OpenVINO only; ignored for PyTorch Whisper)
#   Prefer one device, then script falls back through the rest:
#     NPU | GPU | CPU
#   Empty / unset: try NPU → GPU → CPU in order.
#
# WHISPER_BACKEND
#   auto     — PyTorch Whisper if CUDA or Apple MPS exists, else OpenVINO (default)
#   torch    — always Hugging Face + PyTorch (CUDA / MPS / CPU)
#   openvino — always local OpenVINO export at WHISPER_MODEL_PATH
#
# WHISPER_HF_MODEL  (PyTorch / WHISPER_BACKEND=torch or auto with GPU)
#   Any Hub ASR model id, e.g.:
#     openai/whisper-tiny, openai/whisper-base, openai/whisper-small,
#     openai/whisper-medium, openai/whisper-large-v2, openai/whisper-large-v3,
#     distil-whisper/distil-small.en, …
#   Default below: openai/whisper-small
#
# WHISPER_OV_RELOAD_EVERY  (OpenVINO only)
#   The NPU plugin's memory pool degrades over many varying-shape inference
#   calls, eventually throwing "bad allocation". To work around this, the
#   OpenVINO pipeline is reloaded from scratch every N segments. Set to 0 to
#   disable. Default: 50

TEMP_WAV_PATH = "temp_meeting_clean.wav"
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "whisper")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "").strip() or None
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "auto").strip().lower()
WHISPER_HF_MODEL = os.getenv("WHISPER_HF_MODEL", "openai/whisper-small")
WHISPER_OV_RELOAD_EVERY = int(os.getenv("WHISPER_OV_RELOAD_EVERY", "50") or 0)

def _diarization_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _openvino_device_order(preferred: Optional[str]) -> List[str]:
    order = ["NPU", "GPU", "CPU"]
    if not preferred:
        return order
    p = preferred.strip().upper()
    if p not in order:
        return order
    return [p] + [d for d in order if d != p]


def _load_whisper_openvino(model_path: str, device_hint: Optional[str]) -> Tuple[Any, str]:
    import openvino_genai as ov_genai_local

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"OpenVINO Whisper folder not found: {model_path!r}. "
            "Export with optimum-cli (see README) or set WHISPER_MODEL_PATH."
        )
    last_err: Optional[Exception] = None
    for dev in _openvino_device_order(device_hint):
        try:
            wp = ov_genai_local.WhisperPipeline(model_path, dev)
            return wp, dev
        except Exception as e:
            last_err = e
            print(f"   OpenVINO Whisper on {dev} failed: {e}")
    raise RuntimeError(f"OpenVINO Whisper could not load on any device. Last error: {last_err}")


def _pipeline_device_arg(dev: torch.device):
    if dev.type == "cuda":
        return 0
    if dev.type == "mps":
        return "mps"
    return -1


def _load_whisper_torch(model_id: str) -> Tuple[Any, torch.device]:
    dev = _diarization_torch_device()
    torch_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=_pipeline_device_arg(dev),
    )
    return pipe, dev


def _transcribe_openvino(pipe: Any, audio: np.ndarray) -> str:
    return str(pipe.generate(audio)).strip()


class _OpenVinoWhisperTranscriber:
    """Wraps an OpenVINO WhisperPipeline and periodically reloads it.

    The NPU plugin's memory pool degrades across many inference calls with
    varying input shapes, eventually raising "bad allocation" from
    infer_request.cpp. Reloading the pipeline from scratch every N segments
    resets that state before it gets bad enough to fail.
    """

    def __init__(self, model_path: str, device_hint: Optional[str], reload_every: int):
        self.model_path = model_path
        self.device_hint = device_hint
        self.reload_every = reload_every
        self.segment_count = 0
        self.pipe, self.device = _load_whisper_openvino(model_path, device_hint)

    def _reload(self) -> None:
        print(f"   Reloading OpenVINO Whisper pipeline on {self.device} "
              f"(every {self.reload_every} segments) to reset device memory...")
        self.pipe = None
        try:
            self.pipe, self.device = _load_whisper_openvino(self.model_path, self.device_hint)
        except Exception as e:
            print(f"   Reload failed, falling back through device order: {e}")
            self.pipe, self.device = _load_whisper_openvino(self.model_path, None)

    def __call__(self, audio: np.ndarray) -> str:
        if self.reload_every and self.segment_count > 0 and self.segment_count % self.reload_every == 0:
            self._reload()
        self.segment_count += 1
        return _transcribe_openvino(self.pipe, audio)


def _transcribe_torch(pipe: Any, audio: np.ndarray) -> str:
    out = pipe(audio, sampling_rate=16000)
    if isinstance(out, dict):
        return (out.get("text") or "").strip()
    return str(out).strip()


def _resolved_whisper_backend() -> str:
    if WHISPER_BACKEND in ("openvino", "torch"):
        return WHISPER_BACKEND
    if WHISPER_BACKEND != "auto":
        print(f"   Unknown WHISPER_BACKEND={WHISPER_BACKEND!r}, using auto.")
    if torch.cuda.is_available():
        return "torch"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "torch"
    return "openvino"


def _build_whisper_transcriber() -> Tuple[Callable[[np.ndarray], str], str]:
    mode = _resolved_whisper_backend()
    if mode == "torch":
        print(f"\n Loading Whisper (PyTorch) — {WHISPER_HF_MODEL} ...")
        pipe, dev = _load_whisper_torch(WHISPER_HF_MODEL)
        label = f"PyTorch / {dev.type.upper()}"
        print(f"   -> Success! Whisper on {label}")

        def fn(audio: np.ndarray) -> str:
            return _transcribe_torch(pipe, audio)

        return fn, label

    print(f"\n Loading Whisper (OpenVINO) from '{MODEL_PATH}' ...")
    ov_transcriber = _OpenVinoWhisperTranscriber(MODEL_PATH, WHISPER_DEVICE, WHISPER_OV_RELOAD_EVERY)
    label = f"OpenVINO / {ov_transcriber.device}"
    print(f"   -> Success! Whisper on {label}")
    if WHISPER_OV_RELOAD_EVERY:
        print(f"   -> Will reload pipeline every {WHISPER_OV_RELOAD_EVERY} segments to avoid NPU memory exhaustion")

    return ov_transcriber, label


if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file!")

# --- File selection dialogs (explorer / save-as) ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="Path to video file")
parser.add_argument("--output", type=str, help="Path to output .txt file")
parser.add_argument("--speakers", type=str, help="Number of speakers or 'auto'")
args, _ = parser.parse_known_args()

if args.video and args.output and args.speakers:
    VIDEO_PATH = args.video
    OUTPUT_FILE = args.output
    if args.speakers.lower() == "auto":
        NUM_SPEAKERS = None
    else:
        NUM_SPEAKERS = int(args.speakers)
    if NUM_SPEAKERS is not None:
        print(f"   -> Diarization will use num_speakers={NUM_SPEAKERS}")
else:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    print("Select a video file to transcribe...")
    VIDEO_PATH = filedialog.askopenfilename(
        title="Select video file to transcribe",
        filetypes=[
            ("Video files", ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.webm", "*.flv", "*.wmv", "*.m4v")),
            ("All files", "*.*"),
        ],
    )
    if not VIDEO_PATH:
        raise SystemExit("No video file selected. Exiting.")
    
    print("Choose where to save the transcript and enter the output .txt file name...")
    default_filename = os.path.basename(VIDEO_PATH) + ".txt"
    OUTPUT_FILE = filedialog.asksaveasfilename(
        title="Save transcript as",
        initialfile=default_filename,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not OUTPUT_FILE:
        raise SystemExit("No output file chosen. Exiting.")
    
    NUM_SPEAKERS = simpledialog.askinteger(
        "Number of speakers",
        "How many speakers are in the conversation?\n(Leave empty or Cancel for auto-detect.)",
        initialvalue=2,
        minvalue=1,
        maxvalue=50,
        parent=root,
    )
    # None means user cancelled → let Pyannote auto-detect; otherwise use the chosen value
    if NUM_SPEAKERS is not None:
        print(f"   -> Diarization will use num_speakers={NUM_SPEAKERS}")
    
    root.destroy()

print(f"Configuration loaded (WHISPER_BACKEND={WHISPER_BACKEND!r}).")

# --- 2. PRE-PROCESSING ---

print(f"\n Step 0: Converting '{VIDEO_PATH}' to clean WAV format...")
# Pyannote reads a 16kHz WAV file
try:
    # Load the audio from the video to handle decoding 
    audio_data, samplerate = librosa.load(VIDEO_PATH, sr=16000)
    
    # Save it as a clean 16kHz WAV file
    sf.write(TEMP_WAV_PATH, audio_data, 16000)
    print(f"   -> Saved temporary file: {TEMP_WAV_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to convert video to WAV: {e}")

# --- 3. LOAD MODELS ---

transcribe_segment, whisper_runtime_label = _build_whisper_transcriber()

print("\n Loading Pyannote Pipeline (Speaker ID)...")
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN,
    )
    d_dev = _diarization_torch_device()
    diarization_pipeline.to(d_dev)
    print(f"   -> Diarization: {d_dev.type.upper()}")
except Exception as e:
    raise RuntimeError(f"Pyannote Error: {e}")

# --- 4. PROCESSING ---

print("\n  Step 1: Analyzing Speakers (Diarization)...")

if NUM_SPEAKERS is not None:
    diarization_result = diarization_pipeline(TEMP_WAV_PATH, num_speakers=NUM_SPEAKERS)
else:
    diarization_result = diarization_pipeline(TEMP_WAV_PATH)

print(f"\n Step 2: Transcribing segments ({whisper_runtime_label})...")
final_transcript = []

# Check if the result is wrapped in the new 'DiarizeOutput'
if hasattr(diarization_result, "speaker_diarization"):
    # Extract the actual timeline data from the wrapper
    annotation = diarization_result.speaker_diarization
else:
    # Fallback for older Pyannote versions
    annotation = diarization_result

# Use extracted 'annotation' variable instead of 'diarization_result'
for turn, _, speaker in annotation.itertracks(yield_label=True):
    start_sec = turn.start
    end_sec = turn.end
    duration = end_sec - start_sec
    
    if duration < 0.5:
        continue

    # Convert seconds to array indices (16000 samples per second)
    start_sample = int(start_sec * 16000)
    end_sample = int(end_sec * 16000)
    
    # Slice the audio from pre-loaded array
    speaker_audio = audio_data[start_sample:end_sample]

    try:
        text = transcribe_segment(speaker_audio)
        
        if text:
            # Format: [00:15 - 00:20] SPEAKER_00: Hello.
            timestamp_str = f"[{start_sec:.1f}s - {end_sec:.1f}s]"
            formatted_line = f"{timestamp_str} {speaker}: {text}"
            print(formatted_line)
            final_transcript.append(formatted_line)
            
    except Exception as e:
        print(f" Error transcribing segment {start_sec}-{end_sec}: {e}")
        
# --- 5. CLEANUP & SAVING ---

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(final_transcript))

# Optional: Delete the temp wav file to save space
if os.path.exists(TEMP_WAV_PATH):
    os.remove(TEMP_WAV_PATH)

print(f"\n Transcription Complete! Saved to {OUTPUT_FILE}")
