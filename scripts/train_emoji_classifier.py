import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# 🏷 **Set dataset path**
categorized_dataset_dir = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\categorized_images"

# 📊 **Data Augmentation for Better Performance**
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 80% training, 20% validation
)

# 📂 **Load training & validation data**
train_data = datagen.flow_from_directory(
    categorized_dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

val_data = datagen.flow_from_directory(
    categorized_dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)

# 🏗 **Define CNN Model**
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),  # Reduces overfitting
    Dense(len(train_data.class_indices), activation="softmax"),
])

# 🎯 **Compile Model**
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lowered learning rate
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 📌 **Train Model & Save Every 5 Epochs**
num_epochs = 100
save_interval = 5
model_save_path = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\models"

os.makedirs(model_save_path, exist_ok=True)  # Ensure directory exists

for epoch in range(1, num_epochs + 1):
    print(f"\n🔹 Training Epoch {epoch}/{num_epochs}...\n")
    model.fit(train_data, validation_data=val_data, epochs=1)  # Train 1 epoch at a time

    # 📁 **Save Model Every 5 Epochs**
    if epoch % save_interval == 0:
        model_filename = os.path.join(model_save_path, f"emoji_model_epoch_{epoch}.h5")
        model.save(model_filename)
        print(f"✅ Model saved at: {model_filename}")

# 📝 **Save Final Model & Class Labels**
final_model_path = os.path.join(model_save_path, "emoji_detector_final.h5")
model.save(final_model_path)

class_labels_path = os.path.join(model_save_path, "class_labels.json")
with open(class_labels_path, "w") as f:
    json.dump(train_data.class_indices, f)

print("\n🎉 Training complete! Final model saved.")

