import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
import json
import random

# 🏷 Set paths
model_path = r"D:\University Work\Backup\emoji-detector-ai\models\category_classification_model\mobilenetv2_finetuned.h5"  # Updated model path
class_labels_path = r'D:\University Work\Backup\emoji-detector-ai\models\category_classification_model\class_labels.json'  # Update this path if needed
dataset_dir = r"D:\University Work\Backup\emoji-detector-ai\emoji_data\classified_emojis"  # Path to the classified_emojis directory

# 📊 Load class labels
with open(class_labels_path, 'r', encoding='utf-8') as f:
    class_labels = json.load(f)
class_names = list(class_labels.keys())
index_to_class = {v: k for k, v in class_labels.items()}
print(f"Loaded {len(class_names)} classes: {class_names}")

# 🧠 Load the trained model
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# 📂 Collect all image paths from the classified_emojis directory
image_paths = []
true_labels = []
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue
    for img_name in os.listdir(category_path):
        if img_name.endswith('.png') or img_name.endswith('.jpg'):  # Include both .png and .jpg formats
            img_path = os.path.join(category_path, img_name)
            image_paths.append(img_path)
            true_labels.append(category)

# 🧩 Debugging step: Check the number of images collected
print(f"Total images found: {len(image_paths)}")
if len(image_paths) == 0:
    print("No images found. Please check the dataset path or format of images.")
else:
    # 📌 Randomly select 100 images for prediction
    num_images_to_predict = 100  # Adjust as needed
    if len(image_paths) < num_images_to_predict:
        num_images_to_predict = len(image_paths)
        print(f"Dataset has only {num_images_to_predict} images. Predicting on all available images.")
    else:
        print(f"Selecting {num_images_to_predict} random images for prediction...")

    random_indices = random.sample(range(len(image_paths)), num_images_to_predict)
    selected_image_paths = [image_paths[i] for i in random_indices]
    selected_true_labels = [true_labels[i] for i in random_indices]

    # 📈 Predict categories and track results
    print("\nPredictions:\n")
    table_data = []
    correct_predictions = 0
    predicted_labels = []
    for img_path, true_label in zip(selected_image_paths, selected_true_labels):
        # Load and preprocess the image
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = index_to_class[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx] * 100

        # Track results
        img_name = os.path.basename(img_path)
        predicted_labels.append(predicted_class)
        if predicted_class == true_label:
            correct_predictions += 1
        table_data.append([img_name, true_label, predicted_class, f"{confidence:.2f}%"])

    # Print predictions in a table
    headers = ["Image", "True Category", "Predicted Category", "Confidence"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 📊 Calculate and display metrics
    accuracy = (correct_predictions / num_images_to_predict) * 100

    print("\n📊 Evaluation Metrics:")
    print(f"Total Images Tested: {num_images_to_predict}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Classification Report with all class names
    print("\nClassification Report:")
    # Ensure all class names are included, even if not present in the sample
    report = classification_report(
        selected_true_labels,
        predicted_labels,
        labels=class_names,  # Specify all class names
        target_names=class_names,
        zero_division=0  # Avoid division by zero for classes with no predictions
    )
    print(report)

    # Confusion Matrix with all class names
    print("\nConfusion Matrix:")
    cm = confusion_matrix(
        selected_true_labels,
        predicted_labels,
        labels=class_names  # Specify all class names
    )
    cm_table = [[class_names[i]] + list(cm[i]) for i in range(len(class_names))]
    headers_cm = [""] + class_names
    print(tabulate(cm_table, headers=headers_cm, tablefmt="grid"))
