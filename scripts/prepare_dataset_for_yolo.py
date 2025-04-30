import os
import shutil
import random
from PIL import Image

# Paths
dataset_dir = "assets/whatsapp_synthetic"
images_dir = dataset_dir  # Images are directly in dataset_dir, not in a subdirectory
annotations_dir = os.path.join(dataset_dir, "annotations")
yolo_dir = os.path.join(dataset_dir, "yolo_dataset")
train_images_dir = os.path.join(yolo_dir, "images", "train")
val_images_dir = os.path.join(yolo_dir, "images", "val")
train_labels_dir = os.path.join(yolo_dir, "labels", "train")
val_labels_dir = os.path.join(yolo_dir, "labels", "val")

# Create directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Load all unique emoji classes
emoji_classes = set()
annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
for annotation_file in annotation_files:
    with open(os.path.join(annotations_dir, annotation_file), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {os.path.join(annotations_dir, annotation_file)}: {line.strip()}")
                continue
            unicode = parts[0]  # e.g., U+1F468_U+1F3FD_U+200D_U+1F9B3
            emoji_classes.add(unicode)
emoji_classes = sorted(list(emoji_classes))
class_to_id = {cls: idx for idx, cls in enumerate(emoji_classes)}

# Save class names to a file
with open(os.path.join(yolo_dir, "classes.txt"), 'w', encoding='utf-8') as f:
    for cls in emoji_classes:
        f.write(f"{cls}\n")

# Split dataset (80% train, 20% val)
image_files = [f for f in os.listdir(images_dir) if f.startswith("chat_") and f.endswith(".png")]
random.shuffle(image_files)
train_split = int(0.8 * len(image_files))
train_images = image_files[:train_split]
val_images = image_files[train_split:]

# Convert annotations to YOLO format
def convert_to_yolo_format(image_path, annotation_path, output_path):
    # Load image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Read annotations
    yolo_annotations = []
    with open(annotation_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Skipping malformed line in {annotation_path}: {line.strip()}")
                continue
            unicode, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Convert to YOLO format: class_id center_x center_y width height (normalized)
            class_id = class_to_id[unicode]
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            yolo_annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")

    # Save YOLO annotations
    with open(output_path, 'w', encoding='utf-8') as f:
        for annotation in yolo_annotations:
            f.write(f"{annotation}\n")

# Process training set
for image_file in train_images:
    image_path = os.path.join(images_dir, image_file)
    annotation_file = f"chat_{image_file.split('_')[1].split('.')[0]}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_file)
    if not os.path.exists(annotation_path):
        print(f"Annotation file {annotation_path} not found. Skipping.")
        continue

    # Copy image
    shutil.copy(image_path, os.path.join(train_images_dir, image_file))

    # Convert and save YOLO annotation
    yolo_annotation_path = os.path.join(train_labels_dir, f"chat_{image_file.split('_')[1].split('.')[0]}.txt")
    convert_to_yolo_format(image_path, annotation_path, yolo_annotation_path)

# Process validation set
for image_file in val_images:
    image_path = os.path.join(images_dir, image_file)
    annotation_file = f"chat_{image_file.split('_')[1].split('.')[0]}.txt"
    annotation_path = os.path.join(annotations_dir, annotation_file)
    if not os.path.exists(annotation_path):
        print(f"Annotation file {annotation_path} not found. Skipping.")
        continue

    # Copy image
    shutil.copy(image_path, os.path.join(val_images_dir, image_file))

    # Convert and save YOLO annotation
    yolo_annotation_path = os.path.join(val_labels_dir, f"chat_{image_file.split('_')[1].split('.')[0]}.txt")
    convert_to_yolo_format(image_path, annotation_path, yolo_annotation_path)

# Create data.yaml with relative paths (relative to yolov5/ directory)
relative_train_path = os.path.relpath(train_images_dir, "yolov5").replace("\\", "/")
relative_val_path = os.path.relpath(val_images_dir, "yolov5").replace("\\", "/")

data_yaml = f"""
train: {relative_train_path}
val: {relative_val_path}
nc: {len(emoji_classes)}
names: {emoji_classes}
"""
with open(os.path.join(yolo_dir, "data.yaml"), 'w', encoding='utf-8') as f:
    f.write(data_yaml)

print(f"Dataset prepared at {yolo_dir}")
print(f"Number of classes: {len(emoji_classes)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")