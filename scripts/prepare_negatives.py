import os
import librosa
import random
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
NEGATIVES_DIRECTORY = "./data/negatives"
NOISE_DIRECTORY = "./data/background_noise"
FINAL_NEGATIVES_DIRECTORY = "./data/negatives/final_negatives"

TARGET_SR=16000
TARGET_AUDIO_DURATION=2.0
TARGET_SAMPLES=int(TARGET_SR*TARGET_AUDIO_DURATION) # Exactly 32,000 samples (2.0 seconds)

os.makedirs(FINAL_NEGATIVES_DIRECTORY, exist_ok=True)

print("Cataloging background noises...")
noise_files=[os.path.join(NOISE_DIRECTORY, filename) for filename in os.listdir(NOISE_DIRECTORY) if filename.endswith(".wav")]

if not noise_files:
  print("ERROR: No noise files found in data/background_noise.")
  exit()

def get_random_noise():
  noise_path=random.choice(noise_files)
  noise, _=librosa.load(noise_path, sr=TARGET_SR)

  while len(noise)<TARGET_SAMPLES:
    noise=np.tile(noise, 2)

  max_start=len(noise)-TARGET_SAMPLES
  start_index=random.randint(0, max_start)
  return noise[start_index:start_index+TARGET_SAMPLES]

print(f"Containerizing and mixing noise for Negatives...")

negative_files=[filename for filename in os.listdir(NEGATIVES_DIRECTORY) if filename.endswith(".wav")]

count=0
for filename in tqdm(negative_files, desc="Processing Negatives...."):
  file_path=os.path.join(NEGATIVES_DIRECTORY, filename)

  audio, _=librosa.load(file_path, sr=TARGET_SR)

  current_len=len(audio)
  if current_len<TARGET_SAMPLES:
    pad_total=TARGET_SAMPLES-current_len
    pad_front=random.randint(0, pad_total)
    pad_back=pad_total-pad_front
    audio=np.pad(audio, (pad_front, pad_back), mode="constant")
  elif current_len>TARGET_SAMPLES:
    audio=audio[:TARGET_SAMPLES]
  
  noise_layer=get_random_noise()
  noise_volumne=random.uniform(0.1, 0.4)

  mixed_audio=audio+(noise_layer*noise_volumne)

  mixed_audio=np.clip(mixed_audio, -1.0, 1.0)

  dest_path=os.path.join(FINAL_NEGATIVES_DIRECTORY, filename)
  sf.write(dest_path, mixed_audio, TARGET_SR)
  count+=1

print(f"\n{count} Negatives have been containerized, noise-layered, and saved to {FINAL_NEGATIVES_DIRECTORY}.")