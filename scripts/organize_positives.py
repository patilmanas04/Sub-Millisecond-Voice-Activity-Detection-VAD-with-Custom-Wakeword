import os
import shutil

SOURCE_DIRECTORIES=[
  "data/positives/augmented",
  "data/positives/augmented_manual",
  "data/positives/manual",
  "data/positives/tts"
]

FINAL_DIRECTORY="data/positives/final_positives"

os.makedirs(FINAL_DIRECTORY, exist_ok=True)
print(f"Consolidating all positive samples into {FINAL_DIRECTORY}...")

total_copied=0
for directory in SOURCE_DIRECTORIES:
  if not os.path.exists(directory):
    print(f"Skipping {directory} - folder not found.")
    continue

  count=0
  for filename in os.listdir(directory):
    if filename.endswith(".wav"):
      folder_name=os.path.basename(directory)
      safe_filename=f"{folder_name}_{filename}"

      src_path=os.path.join(directory, filename)
      dest_path=os.path.join(FINAL_DIRECTORY, safe_filename)

      shutil.copy2(src_path, dest_path)
      count+=1
      total_copied+=1
  
  print(f"Copied {count} files from {directory}")

print(f"\nSuccessfully aggregated {total_copied} perfectly sized 2.0s samples into {FINAL_DIRECTORY}.")