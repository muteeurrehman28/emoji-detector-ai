import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ─── Setup ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["__background__", "emoji", "timestamp", "message"]
CROP_SAVE_DIR = "/content/drive/MyDrive/emoji_dataset/cropped_predictions"

os.makedirs(CROP_SAVE_DIR, exist_ok=True)

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

# ─── Prediction and Crop Saving ───────────────────────────────────────
def save_predictions(model, image_paths, threshold=0.5, padding=10):
    model.eval()
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        img_tensor = get_transforms(False)(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)

        pred_boxes = outputs[0]['boxes'].cpu().numpy()
        pred_scores = outputs[0]['scores'].cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()

        keep = pred_scores >= threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]

        for idx, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            x1, y1, x2, y2 = map(int, box)
            # Apply padding while ensuring boundaries
            x1 = max(x1 - padding, 0)
            y1 = max(y1 - padding, 0)
            x2 = min(x2 + padding, orig_w)
            y2 = min(y2 + padding, orig_h)

            crop = image.crop((x1, y1, x2, y2))
            label_name = CLASS_NAMES[label]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            crop_filename = f"{img_name}_{label_name}_{idx}.png"
            crop.save(os.path.join(CROP_SAVE_DIR, crop_filename))

        print(f"✅ Cropped predictions saved for: {img_path}")

# ─── Run Inference ─────────────────────────────────────────────────────

model_path = "/content/drive/MyDrive/emoji_dataset/models/model_epoch_20.pth"
model = load_model(model_path)

test_images = [
    "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_301.png",
    "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_302.png"
]

save_predictions(model, test_images)
