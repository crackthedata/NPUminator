import os
import time
import subprocess
from tkinter import Tk, filedialog, simpledialog

def main():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Select a folder containing .mp4 videos...")
    folder_path = filedialog.askdirectory(title="Select folder with .mp4 files")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    delay = simpledialog.askinteger(
        "Cooldown Delay",
        "Enter delay in seconds between transcriptions (to allow cooldown):",
        initialvalue=10,
        minvalue=0,
        parent=root
    )
    if delay is None:
        print("No delay entered. Exiting.")
        return

    root.destroy()

    mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
    if not mp4_files:
        print(f"No .mp4 files found in {folder_path}.")
        return

    print(f"Found {len(mp4_files)} .mp4 files. Starting batch transcription...")

    for i, file_name in enumerate(mp4_files):
        video_path = os.path.join(folder_path, file_name)
        output_file = video_path + ".txt"
        
        print(f"\n[{i+1}/{len(mp4_files)}] Processing: {file_name}")
        
        cmd = [
            "python", "transcriber.py",
            "--video", video_path,
            "--output", output_file,
            "--speakers", "auto"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_name}: {e}")
            continue
            
        if i < len(mp4_files) - 1:
            print(f"\nWaiting {delay} seconds before next video...")
            time.sleep(delay)

    print("\nBatch transcription complete!")

if __name__ == "__main__":
    main()
