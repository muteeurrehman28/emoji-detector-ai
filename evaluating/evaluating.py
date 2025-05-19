import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ─── Setup ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["__background__", "emoji", "timestamp", "message"]
CLASS2IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
SCORE_THRESH = 0.5
IOU_THRESH = 0.5

# ─── Dataset ───────────────────────────────────────────────────────────────────
class EmojiChatDataset(Dataset):
    def __init__(self, images_dir, ann_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.images = sorted([p.name for p in self.images_dir.glob("*") if p.suffix.lower() in [".png", ".jpg"]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = self.images_dir / img_name
        ann_path = self.ann_dir / f"{Path(img_name).stem}.json"

        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        data = json.loads(ann_path.read_text())
        boxes, labels = [], []
        for obj in data["annotations"]:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(CLASS2IDX[obj["class"]])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (torch.as_tensor(boxes)[:, 2] - torch.as_tensor(boxes)[:, 0]) *
                    (torch.as_tensor(boxes)[:, 3] - torch.as_tensor(boxes)[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return img, target

def get_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

# ─── Load Model ────────────────────────────────────────────────────────────────
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def evaluate_model_accuracy(model, data_loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, targ in zip(outputs, targets):
                gt_boxes = targ["boxes"].cpu().numpy()
                gt_labels = targ["labels"].cpu().numpy()
                pred_boxes = out["boxes"].cpu().numpy()
                pred_labels = out["labels"].cpu().numpy()
                pred_scores = out["scores"].cpu().numpy()

                keep = pred_scores >= SCORE_THRESH
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                matched = set()
                for g_box, g_lbl in zip(gt_boxes, gt_labels):
                    cands = [(i, compute_iou(g_box, p_box))
                             for i, p_box in enumerate(pred_boxes)
                             if pred_labels[i] == g_lbl]
                    best = [(i, iou) for i, iou in cands if iou >= IOU_THRESH]
                    if best:
                        best_i = max(best, key=lambda x: x[1])[0]
                        if best_i not in matched:
                            tp += 1
                            matched.add(best_i)
                        else:
                            fn += 1
                    else:
                        fn += 1
                fp += len(pred_boxes) - len(matched)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"\n✅ TEST RESULT")
    print(f"   TP={tp}, FP={fp}, FN={fn}")
    print(f"   Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ─── Run Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_IMG = "/content/drive/MyDrive/emoji_dataset/prediction"
    DATA_ANN = "/content/drive/MyDrive/emoji_dataset/prediction_anno"
    CKPT_PATH = "/content/drive/MyDrive/emoji_dataset/models/model_epoch_20.pth"

    test_dataset = EmojiChatDataset(DATA_IMG, DATA_ANN, transforms=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = get_model(len(CLASS_NAMES)).to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    evaluate_model_accuracy(model, test_loader, DEVICE)
