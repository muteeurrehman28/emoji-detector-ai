import os

# Path to the folder you want to inspect
folder_path = '/content/drive/MyDrive/emoji_dataset/1_annotation'

# List only files (not sub‑directories)
files = [f for f in os.listdir(folder_path)
         if os.path.isfile(os.path.join(folder_path, f))]

print(f"Total files in '{folder_path}': {len(files)}")
