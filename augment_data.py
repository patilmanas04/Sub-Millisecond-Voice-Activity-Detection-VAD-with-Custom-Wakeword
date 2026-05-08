import os
import soundfile as sf
import librosa
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise, Shift

INPUT_DIR="./data/positives/tts"
OUTPUT_DIR="./data/positives/augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

augment=Compose([
  PitchShift(min_semitones=-5, max_semitones=5, p=0.8), # Changes vocal cord size
  TimeStretch(min_rate=0.8, max_rate=1.25, p=0.8), # Changes speaking speed
  AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5), # Adds cheap mic static
  Shift(p=0.5) # Randomly shifts the audio forward/backward in time
])

VARIATIONS_PER_FILE=20

print("Applying the multiplier effect.")

count=0
for filename in os.listdir(INPUT_DIR):
  if filename.endswith(".wav"):
    file_path=os.path.join(INPUT_DIR, filename)

    samples, sample_rate=librosa.load(file_path, sr=16000)

    for i in range(VARIATIONS_PER_FILE):
      augmented_samples=augment(samples=samples, sample_rate=sample_rate)

      new_filename=f"aug_{i}_{filename}"
      save_path=os.path.join(OUTPUT_DIR, new_filename)
      sf.write(save_path, augmented_samples, sample_rate)
      count+=1

print(f"Absolute W. Generated {count} new augmented samples in {OUTPUT_DIR}.")