import tensorflow as tf
import numpy as np
import json
import os
import cv2

# Paths
categorized_dataset_dir = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\categorized_images"
model_path = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\models\emoji_model_epoch_100.h5"
class_labels_path = os.path.join(categorized_dataset_dir, "class_labels.json")
test_images_dir = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\test_images"

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load class labels
with open(class_labels_path, "r") as f:
    class_labels = json.load(f)
class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping

# Get all image files in the test images directory
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if necessary)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Iterate over all test images and make predictions
for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)

    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the class index
    predicted_label = class_labels[predicted_class]  # Get class label

    print(f"✅ {image_file} -> Predicted Emoji: {predicted_label}")
