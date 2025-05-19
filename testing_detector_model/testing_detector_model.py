import os
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["__background__", "emoji", "timestamp", "message"]

def get_transforms(train: bool):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(normalize)
    return T.Compose(transforms)

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

def test_model_on_images(model, image_paths, threshold=0.5):
    model.eval()
    for img_path in image_paths:
        print(f"\n🔍 Testing: {img_path}")
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

        print(f"Found {len(pred_boxes)} predictions above threshold {threshold}")
        for idx, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            x1, y1, x2, y2 = box
            print(f"  Prediction {idx+1}: Label={CLASS_NAMES[label]}, Box={box}")

# Example usage
if __name__ == "__main__":
    model_path = "/content/drive/MyDrive/emoji_dataset/models/model_epoch_20.pth"
    model = load_model(model_path)

    test_images = [
        "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_301.png",
        "/content/drive/MyDrive/emoji_dataset/prediction/Copy of custom_stitched_302.png"
    ]

    test_model_on_images(model, test_images, threshold=0.3)