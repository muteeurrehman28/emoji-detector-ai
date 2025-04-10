from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model & Class Labels
MODEL_PATH = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\models\emoji_model_epoch_100.h5"
CLASS_LABELS_PATH = r"D:\University Work\Semester IV\Software Engineering\emoji-detector-ai\screenshots\categorized_images\class_labels.json"

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load class labels
try:
    with open(CLASS_LABELS_PATH, "r") as f:
        class_labels = json.load(f)
    class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping
    print("✅ Class labels loaded successfully!")
except Exception as e:
    print(f"❌ Error loading class labels: {e}")
    class_labels = {}

# Function to preprocess image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Emoji Detector API!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Process Image
    img_array = preprocess_image(filepath)
    if img_array is None:
        os.remove(filepath)  # Clean up
        return jsonify({"error": "Invalid image format"}), 400

    if model is None:
        os.remove(filepath)  # Clean up
        return jsonify({"error": "Model not loaded"}), 500

    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels.get(predicted_class, "Unknown")

        os.remove(filepath)  # Delete the file after prediction

        return jsonify({"emoji": predicted_label})
    except Exception as e:
        os.remove(filepath)  # Clean up
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
