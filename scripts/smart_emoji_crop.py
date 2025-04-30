import os
from PIL import Image
import numpy as np

# Paths
input_folder = r'D:\University Work\My Projects\Emoji Detector\emoji-detector-ai\emoji_data\bg_transparent_images'
output_folder = r'D:\University Work\My Projects\Emoji Detector\emoji-detector-ai\emoji_data\cropped_emojis'
os.makedirs(output_folder, exist_ok=True)

def crop_to_emoji(image_path, save_path):
    img = Image.open(image_path).convert("RGBA")
    np_img = np.array(img)

    # If image has alpha channel (transparency)
    if np_img.shape[2] == 4:
        alpha_channel = np_img[:, :, 3]
        mask = alpha_channel > 0
    else:
        # Fallback for JPEG: assume white background
        gray = np.mean(np_img[:, :, :3], axis=2)
        mask = gray < 250  # Tune threshold as needed

    coords = np.argwhere(mask)
    if coords.size == 0:
        print(f"No emoji found in: {image_path}")
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(save_path)

# Iterate through files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            crop_to_emoji(input_path, output_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
