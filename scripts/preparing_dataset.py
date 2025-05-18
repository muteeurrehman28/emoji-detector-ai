import rarfile
import os

# Path to your .rar file in Google Drive
rar_path = "/content/drive/MyDrive/emoji_classify/Category Wise Emojis.rar"  # 🔁 <- Update if needed
extract_to = "/content/drive/MyDrive/emoji_classify/Category Wise Emojis"  # Output folder

# Unzip
with rarfile.RarFile(rar_path) as rf:
    rf.extractall(extract_to)

print("✅ Unzipped successfully!")

len(os.listdir("/content/drive/MyDrive/emoji_classify/Category_Wise_Aug"))
