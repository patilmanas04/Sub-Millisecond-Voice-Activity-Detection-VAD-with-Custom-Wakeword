import os
import librosa
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
POSITIVES_DIRECTORY = "./data/positives/final_positives"
NEGATIVES_DIRECTORY = "./data/negatives/final_negatives"
OUTPUT_DIRECTORY = "./data/features"

# Audio parameters
SAMPLE_RATE=16000
N_MELS = 40 # Number of frequency bands (Height of our image)
N_FFT = 480 # Window size for the Fourier Transform
HOP_LENGTH = 160 # How many steps to slide the window (Width of our image)
TARGET_SAMPLES = 32000

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

def process_audio_folder(folder_path, label_value):
  features=[]
  labels=[]

  files=[filename for filename in os.listdir(folder_path) if filename.endswith(".wav")]

  if not files:
    print(f"Warning: No files found in {folder_path}")
    return features, labels
  
  for filename in tqdm(files, desc=f"Converting {os.path.basename(folder_path)}"):
    file_path=os.path.join(folder_path, filename)

    audio, _=librosa.load(file_path, sr=SAMPLE_RATE)

    if len(audio)!=TARGET_SAMPLES:
      continue

    # Create the Mel Spectrogram
    mel_spectrogram=librosa.feature.melspectrogram(
      y=audio,
      sr=SAMPLE_RATE,
      n_mels=N_MELS,
      n_fft=N_FFT,
      hop_length=HOP_LENGTH
    )

    # Convert to Decibels (Log-Scale) - mimics how human ears hear volume
    log_mel_spectrogram=librosa.power_to_db(mel_spectrogram, ref=np.max)

    features.append(log_mel_spectrogram)
    labels.append(label_value)
  
  return features, labels

if __name__=="__main__":
  print("--- EXTRACTING FEATURES ---")

  # Process Positives (Label=1)
  positive_features, positive_labels=process_audio_folder(POSITIVES_DIRECTORY, label_value=1)

  # Process Negatives (Label=0)
  negative_features, negative_labels=process_audio_folder(NEGATIVES_DIRECTORY, label_value=0)

  # Combine lists and convert to high-performance NumPy arrays
  X=np.array(positive_features+negative_features, dtype=np.float32)
  y=np.array(positive_labels+negative_labels, dtype=np.int8)

  # CRITICAL STEP for CNNs: We must add a "Channel" dimension.
  # Currently X is shaped (Number of Samples, Height, Width).
  # PyTorch CNNs expect (Number of Samples, Channels, Height, Width).
  # Since spectrograms are "grayscale" (1 channel), we add a dummy dimension.
  X=np.expand_dims(X, axis=1)

  print(f"Final Input Data Shape (X): {X.shape}")
  print(f"Final Label Data Shape (y): {y.shape}")

  print("\nSaving raw tensors to disk...")
  np.save(os.path.join(OUTPUT_DIRECTORY, "X_data.npy"), X)
  np.save(os.path.join(OUTPUT_DIRECTORY, "y_labels.npy"), y)

  print(f"Feature extraction completed. Files saved to {OUTPUT_DIRECTORY}.")