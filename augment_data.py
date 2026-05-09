import os
import soundfile as sf
import librosa
import numpy as np
import random
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift

INPUT_DIR="./data/positives/tts"
OUTPUT_DIR="./data/positives/augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

augment_wave=Compose([
  PitchShift(min_semitones=-5, max_semitones=5, p=0.8), # Changes vocal cord size
  TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8), # Changes speaking speed
])

augment_noise=Compose([
  AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5) # Adds cheap mic static
])

VARIATIONS_PER_FILE=20
TARGET_SR=16000
TARGET_AUDIO_DURATION=2.0 # SEC
TARGET_SAMPLES=int(TARGET_SR*TARGET_AUDIO_DURATION)

print("Applying the multiplier effect inside a secure 2-second container...")

count=0
for filename in os.listdir(INPUT_DIR):
  if filename.endswith(".wav"):
    file_path=os.path.join(INPUT_DIR, filename)

    samples, sample_rate=librosa.load(file_path, sr=TARGET_SR)

    for i in range(VARIATIONS_PER_FILE):
      augmented_samples=augment_wave(samples=samples, sample_rate=sample_rate)

      current_len=len(augmented_samples)

      if current_len<TARGET_SAMPLES:
        pad_total=TARGET_SAMPLES-current_len

        pad_front=random.randint(0, pad_total)
        pad_back=pad_total-pad_front

        augmented_samples=np.pad(augmented_samples, (pad_front, pad_back), mode="constant")
      elif current_len>TARGET_SAMPLES:
        augmented_samples=augmented_samples[:TARGET_SAMPLES]
      
      final_augmented_samples=augment_noise(samples=augmented_samples, sample_rate=sample_rate)

      new_filename=f"aug_{i}_{filename}"
      save_path=os.path.join(OUTPUT_DIR, new_filename)
      sf.write(save_path, final_augmented_samples, sample_rate)
      count+=1

print(f"Generated {count} flawlessly containerized samples in {OUTPUT_DIR}.")