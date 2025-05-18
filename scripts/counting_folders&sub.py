import os

base_path = "/content/drive/MyDrive/emoji_classify/Category_Wise_Aug"

grand_total_files = 0  # 🔢 Accumulate total across all categories

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)

    if not os.path.isdir(category_path):
        continue

    print(f"\n📁 Processing category: {category}")

    total_files = 0

    # Walk through all subdirectories and count files
    for root, dirs, files in os.walk(category_path):
        total_files += len([f for f in files if os.path.isfile(os.path.join(root, f))])

    grand_total_files += total_files  # ➕ Add to overall total

    if total_files == 0:
        print(f"⚠️ No files found in subfolders of category: {category}")
    else:
        print(f"✅ Found {total_files} file(s) in subfolders of category: {category}")

print(f"\n📊 Grand total of all emoji images: {grand_total_files}")
