import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import os
import json
import datetime

# 🏷 Set paths
categorized_dataset_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\augmented_classified_emojis"
model_save_path = r"D:\University Work\Backup\emoji-detector-ai\models"
log_dir = os.path.join(model_save_path, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 📊 Data augmentation for training
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% training, 20% validation
)

# 📂 Load training and validation data
train_data = datagen.flow_from_directory(
    categorized_dataset_dir,
    target_size=(128, 128),
    batch_size=16,  # Smaller batch size for large dataset
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    categorized_dataset_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

# 🧠 Define lightweight custom CNN architecture
def create_model(num_classes):
    model = Sequential([
        SeparableConv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        SeparableConv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        SeparableConv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax")
    ])
    return model

# 📌 Load existing model or create new one
checkpoint_path = os.path.join(model_save_path, "best_emoji_model.h5")
if os.path.exists(checkpoint_path):
    print(f"Loading existing model from: {checkpoint_path}")
    model = load_model(checkpoint_path)
else:
    print("No checkpoint found. Creating new model.")
    model = create_model(num_classes=len(train_data.class_indices))

# ⚙️ Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 📌 Define callbacks
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# 📈 Training
num_epochs = 20  # Train for 20 epochs
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=num_epochs,
    callbacks=[checkpoint, early_stopping, lr_scheduler, tensorboard],
    verbose=1
)

# 📝 Save the final model in HDF5 and SavedModel formats
final_model_h5_path = os.path.join(model_save_path, "emoji_detector_final.h5")
model.save(final_model_h5_path)

final_model_savedmodel_path = os.path.join(model_save_path, "emoji_detector_final")
model.save(final_model_savedmodel_path, save_format="tf")

# 💾 Save class labels
class_labels_path = os.path.join(model_save_path, "class_labels.json")
with open(class_labels_path, "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=2, ensure_ascii=False)

# 📊 Save training history
history_path = os.path.join(model_save_path, "training_history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history.history, f, indent=2)

print(f"\n🎉 Training complete!")
print(f"Best model saved at: {checkpoint_path}")
print(f"Final model saved at: {final_model_h5_path} (HDF5) and {final_model_savedmodel_path} (SavedModel)")
print(f"Class labels saved at: {class_labels_path}")
print(f"Training history saved at: {history_path}")
print(f"TensorBoard logs at: {log_dir}")