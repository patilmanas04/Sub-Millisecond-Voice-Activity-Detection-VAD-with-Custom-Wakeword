import os
import shutil
from tqdm import tqdm

KAGGLE_SPEECH_DIRECTORY = "./downloads/speech_commands" 
DESTINATION_DIRECTORY = "./data/negatives"

TARGET_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)
print(f"Filtering specific negative words from {KAGGLE_SPEECH_DIRECTORY}...")

total_copied = 0

# Check if the directory actually exists first
if not os.path.exists(KAGGLE_SPEECH_DIRECTORY):
  print(f"ERROR: Cannot find the folder {KAGGLE_SPEECH_DIRECTORY}.")
else:
  for word in TARGET_WORDS:
    word_dir=os.path.join(KAGGLE_SPEECH_DIRECTORY, word)

    if not os.path.exists(word_dir):
      print(f"Warning: Couldn't find folder for '{word}' - skipping.")
      continue

    print(f"Copying files for '{word}'...")

    files = [f for f in os.listdir(word_dir) if f.endswith(".wav")]

    for filename in tqdm(files, desc=word):
      src_path=os.path.join(word_dir, filename)

      new_filename = f"{word}_{filename}"
      dest_path = os.path.join(DESTINATION_DIRECTORY, new_filename)
      
      shutil.copy2(src_path, dest_path)
      total_copied += 1

  print(f"\nSuccessfully locked in {total_copied} perfectly isolated negative samples into {DESTINATION_DIRECTORY}.")