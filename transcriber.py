import os
import librosa
import torch
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from dotenv import load_dotenv
from pyannote.audio import Pipeline
import openvino_genai as ov_genai 
import soundfile as sf

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables from .env file
load_dotenv()

# Configuration Variables
TEMP_WAV_PATH = "temp_meeting_clean.wav"  
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = "whisper-base-ov"
DEVICE_WHISPER = "NPU"
DEVICE_DIARIZATION = "cpu"      

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file!")

# --- File selection dialogs (explorer / save-as) ---
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
OUTPUT_FILE = filedialog.asksaveasfilename(
    title="Save transcript as",
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

print(f"Configuration loaded. Target: {DEVICE_WHISPER}")

# --- 2. PRE-PROCESSING (The Fix) ---

print(f"\n Step 0: Converting '{VIDEO_PATH}' to clean WAV format...")
# This fixes the "ValueError: requested chunk..." crash by ensuring
# Pyannote reads a perfect 16kHz WAV file, not a messy MP4.
try:
    # Load the audio from the video (this handles the decoding)
    audio_data, samplerate = librosa.load(VIDEO_PATH, sr=16000)
    
    # Save it as a clean 16kHz WAV file
    sf.write(TEMP_WAV_PATH, audio_data, 16000)
    print(f"   -> Saved temporary file: {TEMP_WAV_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to convert video to WAV: {e}")

# --- 3. LOAD MODELS ---

print(f"\n Loading WhisperPipeline from '{MODEL_PATH}' to {DEVICE_WHISPER}...")
try:
    whisper_pipe = ov_genai.WhisperPipeline(MODEL_PATH, DEVICE_WHISPER)
    print(f"   -> Success! Whisper running on {DEVICE_WHISPER}")
except Exception as e:
    print(f"  NPU Error: {e}")
    print("   -> Falling back to CPU...")
    whisper_pipe = ov_genai.WhisperPipeline(MODEL_PATH, "CPU")

print("\n🎤 Loading Pyannote Pipeline (Speaker ID)...")
try:
    # Updated to use 'token=' instead of 'use_auth_token=' for newer versions
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        token=HF_TOKEN 
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
        print("   -> Diarization: GPU (CUDA)")
    else:
        print("   -> Diarization: CPU")
except Exception as e:
    raise RuntimeError(f"Pyannote Error: {e}")

# --- 4. PROCESSING ---

print("\n🕵️  Step 1: Analyzing Speakers (Diarization)...")
# CRITICAL CHANGE: We pass the CLEAN WAV file, not the MP4
# NUM_SPEAKERS from tkinter dialog (None = auto-detect)
if NUM_SPEAKERS is not None:
    diarization_result = diarization_pipeline(TEMP_WAV_PATH, num_speakers=NUM_SPEAKERS)
else:
    diarization_result = diarization_pipeline(TEMP_WAV_PATH)

print("\n Step 2: Transcribing Segments (Whisper GenAI)...")
final_transcript = []

# --- NEW: Handle the new Pyannote v4.x output format ---
# We check if the result is wrapped in the new 'DiarizeOutput' object
if hasattr(diarization_result, "speaker_diarization"):
    # Extract the actual timeline data from the wrapper
    annotation = diarization_result.speaker_diarization
else:
    # Fallback for older Pyannote versions
    annotation = diarization_result

# We use our newly extracted 'annotation' variable here instead of 'diarization_result'
for turn, _, speaker in annotation.itertracks(yield_label=True):
    start_sec = turn.start
    end_sec = turn.end
    duration = end_sec - start_sec
    
    if duration < 0.5:
        continue

    # Convert seconds to array indices (16000 samples per second)
    start_sample = int(start_sec * 16000)
    end_sample = int(end_sec * 16000)
    
    # Slice the audio from our pre-loaded array
    speaker_audio = audio_data[start_sample:end_sample]

    try:
        # Pass the raw numpy array directly to the C++ pipeline
        res = whisper_pipe.generate(speaker_audio)
        text = str(res).strip()
        
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

print(f"\n✅ Transcription Complete! Saved to {OUTPUT_FILE}")
