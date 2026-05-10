import os
import librosa
import soundfile as sf
import numpy as np

DIRECTORIES_TO_CLEAN=[
  "data/positives/tts",
  "data/positives/manual"
]

TARGET_SR=16000
TARGET_AUDIO_DURATION=2.0 # SEC
TARGET_SAMPLES=int(TARGET_SR*TARGET_AUDIO_DURATION) # Exactly 32,000 samples

print("Sweeping base directories and containerizing raw files...")

for directory in DIRECTORIES_TO_CLEAN:
  count=0
  if not os.path.exists(directory):
    continue

  for filename in os.listdir(directory):
    if filename.endswith(".wav"):
      file_path=os.path.join(directory, filename)

      samples, sample_rate=librosa.load(file_path, sr=TARGET_SR)
      current_len=len(samples)

      if current_len==TARGET_SAMPLES:
        continue
      
      if current_len<TARGET_SAMPLES:
        pad_total=TARGET_SAMPLES-current_len
        pad_front=pad_total//2
        pad_back=pad_total-pad_front
        samples=np.pad(samples, (pad_front, pad_back), mode="constant")

      elif current_len>TARGET_SAMPLES:
        samples=samples[:TARGET_SAMPLES]
      
      sf.write(file_path, samples, sample_rate)
      count+=1

  print(f"Standardized {count} files in {directory}") 

print("All base files are now perfectly 2.0 seconds. Ready for training.")