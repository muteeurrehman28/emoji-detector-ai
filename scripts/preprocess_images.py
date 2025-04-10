import os
import cv2
import numpy as np

# Define paths
dataset_path = r"D:\University Work\Semester IV\Software Engineering\Emoji Detector\emoji-detector-ai\screenshots"
processed_path = os.path.join(dataset_path, "processed")

# Create folder if it doesn't exist
if not os.path.exists(processed_path):
    os.makedirs(processed_path)

# Resize all images and save them
IMG_SIZE = (128, 128)

for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)  # Read image
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img = img / 255.0  # Normalize (optional)

        # Convert image back to uint8 format (0-255)
        img = (img * 255).astype(np.uint8)

        # Save the processed image
        processed_file = os.path.join(processed_path, filename)
        cv2.imwrite(processed_file, img)

print("✅ Image preprocessing complete! Check the 'processed' folder.")

"""
This script preprocesses images by resizing them to (128, 128), normalizing pixel values,  
and saving them in the 'processed' folder. It ensures images are in the correct format  
for model training by converting them back to uint8 before saving.
"""
