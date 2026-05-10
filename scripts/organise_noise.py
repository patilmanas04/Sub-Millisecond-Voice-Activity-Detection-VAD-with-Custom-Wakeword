import os
import shutil

KAGGLE_MUSAN_NOISE_DIRECTORY="./downloads/musan/noise"
DESTINATION_DIRECTORY="./data/background_noise"

os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)

print(f"Flattening MUSAN noise data into {DESTINATION_DIRECTORY}...")

count=0
for root, dirs, files in os.walk(KAGGLE_MUSAN_NOISE_DIRECTORY):
  for filename in files:
    if filename.endswith(".wav"):
      src_path=os.path.join(root, filename)
      
      subfolder=os.path.basename(root)
      safe_filename=f"{subfolder}_{filename}"
      dest_path=os.path.join(DESTINATION_DIRECTORY, safe_filename)

      shutil.copy2(src_path, dest_path)
      count+=1

print(f"Successfully moved {count} background noise files into the pipeline.")