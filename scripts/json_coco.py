import json
import os
from pprint import pprint

# === Paths ===
input_dir = "/content/drive/MyDrive/emoji_dataset/1_annotations_rebased"
output_dir = "/content/drive/MyDrive/emoji_dataset/coco_format_annotation"
os.makedirs(output_dir, exist_ok=True)

# === Convert all JSON files in the folder ===
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    with open(input_path, 'r') as f:
        original_data = json.load(f)

    # Build COCO format
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. Prepare categories from original_data["classes"]
    class_name_to_id = {name: idx + 1 for idx, name in enumerate(original_data["classes"])}
    for name, idx in class_name_to_id.items():
        coco["categories"].append({
            "id": idx,
            "name": name,
            "supercategory": "none"
        })

    # 2. Add single image info
    image_id = 1
    coco["images"].append({
        "id": image_id,
        "file_name": original_data["image"],
        "width": original_data["width"],
        "height": original_data["height"]
    })

    # 3. Add annotations
    annotation_id = 1
    for ann in original_data["annotations"]:
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_name_to_id[ann["class"]],
            "bbox": ann["bbox"],
            "area": ann["bbox"][2] * ann["bbox"][3],  # width * height
            "iscrowd": 0
        })
        annotation_id += 1

    # 4. Save COCO-format file with the same filename
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)

    print(f"✅ Converted: {filename} → {output_path}")


# === Optional: Validate a sample output ===
def validate_coco(json_data):
    required_keys = {"images", "annotations", "categories"}
    if not required_keys.issubset(json_data.keys()):
        print(f"❌ Missing keys: {required_keys - json_data.keys()}")
        return False

    if not isinstance(json_data["images"], list) or not all("id" in i and "file_name" in i for i in json_data["images"]):
        print("❌ Invalid 'images' section")
        return False

    if not isinstance(json_data["annotations"], list) or not all("id" in a and "bbox" in a and "image_id" in a and "category_id" in a for a in json_data["annotations"]):
        print("❌ Invalid 'annotations' section")
        return False

    if not isinstance(json_data["categories"], list) or not all("id" in c and "name" in c for c in json_data["categories"]):
        print("❌ Invalid 'categories' section")
        return False

    print("✅ COCO format is valid!")
    return True

# Validate one file
sample_file = os.path.join(output_dir, os.listdir(output_dir)[0])
print(f"\n🧪 Validating: {sample_file}")
with open(sample_file, 'r') as f:
    sample_data = json.load(f)
validate_coco(sample_data)

# Preview sample entries
print("\n📷 Image Entry:")
pprint(sample_data["images"][0])

print("\n🏷️ Annotation Entry:")
pprint(sample_data["annotations"][0])

print("\n📚 Category Entry:")
pprint(sample_data["categories"][0])
