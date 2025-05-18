import os
import torch
from PIL import Image
import json
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ─── Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["__background__", "emoji", "timestamp", "message"]

# ─── Transforms ────────────────────────────────────────────────────────
def get_transforms(train: bool):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(normalize)
    return T.Compose(transforms)

# ─── Model Loader ──────────────────────────────────────────────────────
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(checkpoint_path):
    model = get_model(len(CLASS_NAMES)).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

# ─── Visualization ─────────────────────────────────────────────────────
def visualize_predictions(model, image_paths, threshold=0.5, box_padding=10, save_crops=True):
    model.eval()
    output_dir = Path("cropped_predictions")
    if save_crops:
        output_dir.mkdir(exist_ok=True)
        print(f"✅ Save directory created at: {output_dir.resolve()}")

    for img_path in image_paths:
        print(f"\n🔍 Processing: {img_path}")
        image = Image.open(img_path).convert("RGB")
        img_tensor = get_transforms(False)(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)

        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_scores = outputs[0]['scores'].cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()

        keep = pred_scores >= threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        if len(pred_boxes) == 0:
            print("⚠️ No predictions above threshold.")
            continue

        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        for idx, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            x1, y1, x2, y2 = box

            # Expand the box
            x1_new = max(int(x1 - box_padding), 0)
            y1_new = max(int(y1 - box_padding), 0)
            x2_new = min(int(x2 + box_padding), image.width)
            y2_new = min(int(y2 + box_padding), image.height)

            # Crop region
            cropped = image.crop((x1_new, y1_new, x2_new, y2_new))

            if save_crops:
                label_name = CLASS_NAMES[label]
                crop_name = f"{Path(img_path).stem}_obj{idx+1}_{label_name}.png"
                crop_path = output_dir / crop_name
                cropped.save(crop_path)
                print(f"🖼️ Saved crop: {crop_path}")

            # Draw box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='lime', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, CLASS_NAMES[label], color='lime', fontsize=12, weight='bold')

        plt.axis('off')
        plt.show()

# ─── Run Inference ─────────────────────────────────────────────────────

# Path to trained model
model_path = "/content/drive/MyDrive/emoji_dataset/models/model_epoch_20.pth"
model = load_model(model_path)

# Images to test
test_images = [
    "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_301.png",
    "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_302.png"
]

# Run and display predictions
visualize_predictions(model, test_images, threshold=0.3, box_padding=10, save_crops=True)


