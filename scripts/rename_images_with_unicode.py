import os

# Set your folder and labels.txt path
folder_path = 'D:\\University Work\\Backup\\emoji-detector-ai\\emoji_data\\cropped_emojis'
labels_file_path = 'D:\\University Work\\Backup\\emoji-detector-ai\\data\\labels.txt'

# Build the rename map from labels.txt
rename_map = {}

with open(labels_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 3:
            original_name = parts[0].strip().lower()  # Convert to lowercase for case-insensitive matching
            unicode_name = parts[2].strip().replace(' ', '_')
            rename_map[original_name] = f"{unicode_name}.png"

print(f"✅ Loaded {len(rename_map)} entries from labels.txt")
print("🔍 Sample mappings:")
for i, (k, v) in enumerate(rename_map.items()):
    print(f"  {k} -> {v}")
    if i >= 4:
        break

# Rename files in the folder
print(f"\n📁 Scanning folder: {folder_path}")
renamed = 0
skipped = 0

for filename in os.listdir(folder_path):
    filename_lower = filename.lower()
    print(f"Checking file: {filename}")
    
    # Check if the lowercase version of the filename matches a key in rename_map
    if filename_lower in rename_map:
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, rename_map[filename_lower])
        
        # Check if destination file already exists
        if not os.path.exists(dst):
            os.rename(src, dst)
            print(f"✅ Renamed: {filename} -> {rename_map[filename_lower]}")
            renamed += 1
        else:
            print(f"⚠️ Skipped (destination exists): {rename_map[filename_lower]}")
            skipped += 1
    else:
        print(f"❌ No matching label for: {filename_lower}")

print(f"\n🎉 Done. Renamed {renamed} files, skipped {skipped}.")
