import os

# Paths
base_folder = "/content/drive/MyDrive/emoji_classify/Category Wise Emojis"  # Root folder with subfolders
label_file_path = "/content/drive/MyDrive/emoji_classify/labels.txt"

# Read label.txt and map image filename to Unicode string
name_mapping = {}

with open(label_file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            original_filename = parts[0].strip()
            unicode_sequence = parts[2].strip().replace("U+", "u").replace(" ", "_")  # e.g. u1F1E6_u1F1EB
            name_mapping[original_filename] = f"{unicode_sequence}.png"

# Rename matching files in all subfolders
renamed_count = 0
conflict_count = 0

for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file in name_mapping:
            src_path = os.path.join(root, file)
            new_filename = name_mapping[file]
            dst_path = os.path.join(root, new_filename)

            # Avoid overwriting existing files
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(new_filename)
                i = 1
                while True:
                    alt_name = f"{base}_{i}{ext}"
                    alt_path = os.path.join(root, alt_name)
                    if not os.path.exists(alt_path):
                        dst_path = alt_path
                        conflict_count += 1
                        break
                    i += 1

            os.rename(src_path, dst_path)
            renamed_count += 1
            print(f"✅ Renamed: {file} -> {os.path.basename(dst_path)}")

print(f"\n🎉 Total images renamed: {renamed_count}")
print(f"⚠️ Name conflicts handled with suffix: {conflict_count}")
