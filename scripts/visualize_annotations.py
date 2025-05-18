import os
import json
import cv2
import matplotlib.pyplot as plt

def visualize_annotations_in_range(image_dir, json_dir, start=0, end=10):
    # Color cycle: Red, Orange, Blue (in BGR)
    box_colors = [(255, 0, 0), (0, 165, 255), (0, 0, 255)]  # Red, Orange, Blue

    files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    for i, fname in enumerate(files[start:end]):
        json_path = os.path.join(json_dir, fname)
        image_name = fname.replace('.json', '.png')  # Or .jpg if that's what you're using
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"❌ Image not found for: {image_name}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for idx, ann in enumerate(data.get('annotations', [])):
            x, y, w, h = ann['bbox']
            label = ann.get('label', 'object')
            color = box_colors[idx % len(box_colors)]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        plt.figure(figsize=(6, 10))
        plt.imshow(img)
        plt.title(f"Annotated: {image_name}")
        plt.axis('off')
        plt.show()



# ── 3) configure and run ───────────────────────────────────────────────────────
# json_dir  = "/content/drive/MyDrive/emoji_dataset/2_annotations_rebased"
# image_dir = "/content/drive/MyDrive/emoji_dataset/2_screebshotimages/screenshot_images_data2"
json_dir  = "/content/drive/MyDrive/emoji_dataset/coco_format_annotation"
image_dir = "/content/drive/MyDrive/emoji_dataset/1_screenshotimages"


# Show images from index 150 to 160
visualize_annotations_in_range(image_dir, json_dir, start=150, end=161)
