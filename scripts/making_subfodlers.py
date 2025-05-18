import os
import shutil

base_path = "/content/drive/MyDrive/emoji_classify/Category_Wise_Aug"

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if not os.path.isdir(category_path):
        continue

    print(f"📁 Processing category: {category}")

    for filename in os.listdir(category_path):
        file_path = os.path.join(category_path, filename)

        if not os.path.isfile(file_path):
            continue

        name_part, ext = os.path.splitext(filename)
        if not name_part.startswith("u") or ext.lower() != ".png":
            continue

        parts = name_part.split("_")

        unicode_parts = []
        for p in parts:
            if p.startswith("aug"):
                break
            unicode_parts.append(p.upper().replace("U", "U+"))

        unicode_folder_name = "_".join(unicode_parts)
        new_folder_path = os.path.join(category_path, unicode_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        new_filename = "_".join(unicode_parts) + "_" + parts[-1] + ".png"
        new_path = os.path.join(new_folder_path, new_filename)

        if os.path.exists(new_path):
            continue  # Skip if already moved

        try:
            shutil.move(file_path, new_path)
            print(f"✅ Moved: {filename} → {unicode_folder_name}/{new_filename}")
        except Exception as e:
            print(f"❌ Error moving {file_path}: {e}")
