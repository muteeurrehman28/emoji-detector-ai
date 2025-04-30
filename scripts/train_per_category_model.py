import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
from shutil import copy2
import tempfile
import random

# 📁 Paths
categorized_dataset_dir = r"D:\University Work\My Projects\Emoji Detector\emoji-detector-ai\emoji_data\categorized_images"
model_save_path = r"D:\University Work\My Projects\Emoji Detector\emoji-detector-ai\models\per_category_models"
os.makedirs(model_save_path, exist_ok=True)

# ⚙️ Hyperparameters
img_size = (128, 128)
batch_size = 32
epochs = 20

# 🔁 Get all categories
categories = [d for d in os.listdir(categorized_dataset_dir) if os.path.isdir(os.path.join(categorized_dataset_dir, d))]

for category in categories:
    print(f"\n🚀 Training model for category: {category}")

    with tempfile.TemporaryDirectory() as temp_dir:
        pos_dir = os.path.join(temp_dir, "positive")
        neg_dir = os.path.join(temp_dir, "negative")
        os.makedirs(pos_dir)
        os.makedirs(neg_dir)

        # 🟢 Copy positive images
        pos_src = os.path.join(categorized_dataset_dir, category)
        for file in os.listdir(pos_src):
            copy2(os.path.join(pos_src, file), pos_dir)

        # 🔴 Copy negative images from other categories
        other_categories = [c for c in categories if c != category]
        sampled_negatives = random.sample(other_categories, min(3, len(other_categories)))

        for neg_cat in sampled_negatives:
            neg_src = os.path.join(categorized_dataset_dir, neg_cat)
            neg_files = os.listdir(neg_src)
            random.shuffle(neg_files)
            num_pos = len(os.listdir(pos_src)) // len(sampled_negatives)
            for file in neg_files[:num_pos]:
                copy2(os.path.join(neg_src, file), neg_dir)

        # 🔄 ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        train_data = datagen.flow_from_directory(
            temp_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )

        val_data = datagen.flow_from_directory(
            temp_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )

        # 🧠 CNN Model (binary)
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.fit(train_data, validation_data=val_data, epochs=epochs)

        # 💾 Save model
        save_path = os.path.join(model_save_path, f"{category}_binary_model.h5")
        model.save(save_path)
        print(f"✅ Saved binary model for category '{category}' to: {save_path}")
