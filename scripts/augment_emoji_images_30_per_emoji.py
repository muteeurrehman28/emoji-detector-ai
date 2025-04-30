import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import imgaug.augmenters as iaa
import os
import numpy as np
from PIL import Image

# 🏷 Set paths
input_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\classified_emojis"
output_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\augmented_classified_emojis"

# 📁 Create output directory
os.makedirs(output_dir, exist_ok=True)

# 📊 Define TensorFlow ImageDataGenerator for basic augmentations
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# 📊 Define imgaug augmentation pipeline for advanced augmentations
aug_pipeline = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.LinearContrast((0.8, 1.2))),  # Adjust contrast
    iaa.Sometimes(0.5, iaa.Multiply((0.7, 1.3))),  # Adjust brightness
    iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),  # Adjust sharpness
    iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),  # Random cropping (up to 10% of image)
    iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),  # Light blur for noise
    iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),  # Gaussian noise
    iaa.Sometimes(0.3, iaa.AddToHueAndSaturation((-20, 20)))  # Color jittering
])

# 🔢 Number of augmentations per image
augmentations_per_image = 30

# 📈 Process each category folder
image_count = 0
for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)
    if os.path.isdir(category_path):
        # Create corresponding category folder in output directory
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        # Process each image in the category
        for filename in os.listdir(category_path):
            if filename.endswith(".png"):
                # Load image as uint8
                img_path = os.path.join(category_path, filename)
                img = load_img(img_path, target_size=(128, 128))  # Resize to match training
                img_array = np.array(img, dtype=np.uint8)  # Keep as uint8 for imgaug

                # Apply imgaug augmentations
                for i in range(augmentations_per_image):
                    # Augment with imgaug (uint8 input)
                    aug_img = aug_pipeline.augment_image(img_array)
                    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)  # Ensure valid pixel values

                    # Convert to float32 for ImageDataGenerator
                    aug_img_float = aug_img.astype(np.float32) / 255.0  # Normalize for TensorFlow
                    aug_img_float = aug_img_float.reshape((1,) + aug_img_float.shape)  # Add batch dimension

                    # Apply ImageDataGenerator augmentations
                    aug_iter = datagen.flow(
                        aug_img_float,
                        batch_size=1
                    )
                    aug_img = next(aug_iter)[0]  # Get one augmented image
                    aug_img = (aug_img * 255).astype(np.uint8)  # Convert back to uint8 for saving

                    # Save augmented image in the category folder
                    output_filename = f"{filename.replace('.png', '')}_aug{i}.png"
                    output_path = os.path.join(output_category_path, output_filename)
                    Image.fromarray(aug_img).save(output_path, format="PNG")

                image_count += 1
                if image_count % 100 == 0:
                    print(f"Processed {image_count} images...")

# 📝 Print summary
total_original_images = image_count
total_augmented_images = image_count * augmentations_per_image
print(f"\n🎉 Augmentation complete!")
print(f"Total original images processed: {total_original_images}")
print(f"Total augmented images generated: {total_augmented_images}")
print(f"Augmented images saved in: {output_dir}")