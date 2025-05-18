import os
import zipfile

zip_path = "/content/drive/MyDrive/emoji_classify/Category_Wise_Aug.zip"
extract_path = "/content/drive/MyDrive/emoji_classify/Category_Wise_Aug"

# Open the zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    all_zip_files = zip_ref.namelist()

    extracted_count = 0
    skipped_count = 0

    print("🔍 Checking for missing files...")

    for file in all_zip_files:
        dest_file_path = os.path.join(extract_path, file)

        # Skip folders
        if file.endswith('/'):
            continue

        # If the file is missing, extract it
        if not os.path.exists(dest_file_path):
            # Ensure the subfolder exists
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            zip_ref.extract(file, extract_path)
            extracted_count += 1
        else:
            skipped_count += 1

print(f"✅ Extracted missing files: {extracted_count}")
print(f"⏭️ Skipped existing files: {skipped_count}")
