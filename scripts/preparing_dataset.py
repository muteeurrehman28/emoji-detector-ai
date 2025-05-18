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



# Set the base directory where your images and label.txt are located
base_folder = "/content/drive/MyDrive/emoji_classify"  # <- change this if needed

# Input folder (where cropped transparent emojis are stored)
input_folder = os.path.join(base_folder, "Category Wise Emojis")  # assuming subfolders exist

# Output folder for augmented images
output_folder = os.path.join(base_folder, "Cropped_Augmented")
os.makedirs(output_folder, exist_ok=True)
