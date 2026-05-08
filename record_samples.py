import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

FS=16000
DURATION=2
SAVE_DIR="data/positives/manual"

if not os.path.exists(SAVE_DIR):
  os.makedirs(SAVE_DIR)

print("--- 'Hello Manas' Recorder ---")
print("Instructions: Press Enter, say the wakeword, wait for the beep.")

count=0
try:
  while True:
    input(f"\n[{count+1}] Press Enter to start recording (or Ctrl+C to stop)...")

    print("Recording...")
    recording=sd.rec(int(DURATION*FS), samplerate=FS, channels=1)
    sd.wait()

    filename = f"{SAVE_DIR}/manas_voice_{count:03d}.wav"
    write(filename, FS, recording)

    print(f"Saved: {filename}")
    count+=1
except KeyboardInterrupt:
  print(f"\nStopped! You recorded {count} samples.")