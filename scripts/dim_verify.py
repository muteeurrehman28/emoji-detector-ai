import cv2
import json
import os

def verify_image_dimensions(image_folder, annotation_folder):
    """
    Verify that the dimensions in COCO-format JSON annotation files match the actual image dimensions.

    Args:
        image_folder (str): Path to the folder containing images.
        annotation_folder (str): Path to the folder containing COCO-format JSON annotations.
    """
    # Ensure folders exist
    if not os.path.exists(image_folder):
        print(f"❌ Image folder does not exist: {image_folder}")
        return
    if not os.path.exists(annotation_folder):
        print(f"❌ Annotation folder does not exist: {annotation_folder}")
        return

    # List all JSON annotation files
    json_files = [f for f in os.listdir(annotation_folder) if f.endswith(".json")]
    if not json_files:
        print(f"❌ No JSON files found in: {annotation_folder}")
        return

    for annotation_file in json_files:
        json_path = os.path.join(annotation_folder, annotation_file)

        # Load the annotation data
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load JSON file {annotation_file}: {e}")
            continue

        # Validate COCO structure
        if "images" not in data or not isinstance(data["images"], list):
            print(f"❌ Invalid COCO format in {annotation_file}: Missing or invalid 'images' section")
            continue

        # Process each image in the 'images' section
        for image_info in data["images"]:
            # Check required fields
            if not all(key in image_info for key in ["file_name", "width", "height", "id"]):
                print(f"❌ Missing required fields in 'images' section of {annotation_file}")
                continue

            image_name = image_info["file_name"]
            json_w = image_info["width"]
            json_h = image_info["height"]

            # Construct the image path
            image_path = os.path.join(image_folder, image_name)

            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                continue

            # Get actual image dimensions
            actual_h, actual_w = img.shape[:2]

            # Check if dimensions match
            if json_w != actual_w or json_h != actual_h:
                print(f"⚠️ Mismatch in dimensions for {image_name} (JSON: {annotation_file}):")
                print(f"    Image: {actual_w}x{actual_h}")
                print(f"    JSON : {json_w}x{json_h}")
            else:
                print(f"✅ Dimensions match for {image_name} (JSON: {annotation_file})")

# Folder paths
image_folder = "/content/drive/MyDrive/emoji_dataset/1_screenshotimages"
annotation_folder = "/content/drive/MyDrive/emoji_dataset/coco_format_annotation"

# Run the verification
verify_image_dimensions(image_folder, annotation_folder)
