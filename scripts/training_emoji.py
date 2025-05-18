# ✅ Install Dependencies (run once)
# !pip install torch torchvision scikit-learn --quiet

import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR

# ─── Configuration ─────────────────────────────────────────────────────────────

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS    = 20
BATCH_SIZE    = 2
LEARNING_RATE = 5e-3
WARMUP_ITERS  = 500                # number of iterations for warmup
STEP_SIZE     = 5                  # step LR every this many epochs
GAMMA         = 0.1                # LR decay factor
IOU_THRESH    = 0.5                # for simple evaluation
SCORE_THRESH  = 0.5                # discard low‑confidence preds
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Class Mapping ──────────────────────────────────────────────────────────────

CLASS_NAMES = ["__background__", "emoji", "timestamp", "message"]
CLASS2IDX   = {n:i for i,n in enumerate(CLASS_NAMES)}

# ─── Dataset ───────────────────────────────────────────────────────────────────

class EmojiChatDataset(Dataset):
    def __init__(self, images_dir, ann_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.ann_dir    = Path(ann_dir)
        self.transforms = transforms
        self.images = sorted([p.name for p in self.images_dir.glob("*") if p.suffix.lower() in [".png",".jpg"]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_path  = self.images_dir / img_name
        ann_path  = self.ann_dir / f"{Path(img_name).stem}.json"

        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        data = json.loads(ann_path.read_text())
        boxes, labels = [], []
        for obj in data["annotations"]:
            x, y, w, h = obj["bbox"]
            boxes.append([x, y, x+w, y+h])
            labels.append(CLASS2IDX[obj["class"]])

        target = {
            "boxes":  torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area":    (torch.as_tensor(boxes)[:,2] - torch.as_tensor(boxes)[:,0]) *
                       (torch.as_tensor(boxes)[:,3] - torch.as_tensor(boxes)[:,1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return img, target

# ─── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(train: bool):
    # ImageNet normalization
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(normalize)
    return T.Compose(transforms)

# ─── Collate Function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))

# ─── Model Initialization ──────────────────────────────────────────────────────

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ─── IoU & Simple Evaluator ────────────────────────────────────────────────────

def compute_iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter  = interW * interH
    areaA  = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB  = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def evaluate_simple(model, data_loader, device):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, targ in zip(outputs, targets):
                gt_boxes   = targ["boxes"].cpu().numpy()
                gt_labels  = targ["labels"].cpu().numpy()
                pred_boxes = out["boxes"].cpu().numpy()
                pred_labels= out["labels"].cpu().numpy()
                pred_scores= out["scores"].cpu().numpy()

                # filter by score
                keep = pred_scores >= SCORE_THRESH
                pred_boxes  = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                matched = set()
                # match GT → predictions
                for g_box, g_lbl in zip(gt_boxes, gt_labels):
                    # candidates same label
                    cands = [(i, compute_iou(g_box, p_box))
                             for i,p_box in enumerate(pred_boxes)
                             if pred_labels[i] == g_lbl]
                    # find best IoU above threshold
                    best = [(i,iou) for i,iou in cands if iou >= IOU_THRESH]
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

    precision = tp / (tp+fp) if tp+fp>0 else 0.0
    recall    = tp / (tp+fn) if tp+fn>0 else 0.0
    f1        = 2*precision*recall/(precision+recall+1e-8)
    print(f"\n🔍 Eval | TP={tp}, FP={fp}, FN={fn}")
    print(f"       Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ─── Training & Evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(images_dir, ann_dir):
    # prepare dataset & loaders
    full = EmojiChatDataset(images_dir, ann_dir, transforms=get_transforms(True))
    n = len(full)
    n_train = int(0.8 * n)
    n_test  = n - n_train
    train_ds, test_ds = random_split(full, [n_train, n_test])

    train_ds.dataset.transforms = get_transforms(True)
    test_ds.dataset.transforms  = get_transforms(False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              num_workers=2, collate_fn=collate_fn)

    # Set model and optimizer
    model = get_model(len(CLASS_NAMES)).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=5e-4)

    # warmup + step LR
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-3,
                                total_iters=WARMUP_ITERS)
    step_scheduler   = StepLR(optimizer, step_size=STEP_SIZE,
                              gamma=GAMMA)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, step_scheduler],
                             milestones=[WARMUP_ITERS])

    # 🔁 Load checkpoint if available
    latest_ckpt_path = "/content/drive/MyDrive/emoji_dataset/models/model_epoch_20.pth"
    if os.path.exists(latest_ckpt_path):
        print("✅ Loading existing model from checkpoint...")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        print("🆕 No checkpoint found — training from scratch.")
        start_epoch = 1

    # 🚀 Training loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"📦 Epoch {epoch}/{NUM_EPOCHS} → Loss: {avg_loss:.4f}")

        # Save checkpoint to Drive
        save_path = f"/content/drive/MyDrive/emoji_dataset/models/model_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }, save_path)

        # Simple Evaluation
        evaluate_simple(model, test_loader, DEVICE)

    print("\n🏁 Training complete.")


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # replace these paths with your actual directories:
    DATA1_IMGS = "/content/drive/MyDrive/emoji_dataset/1_screenshotimages"
    DATA1_ANN  = "/content/drive/MyDrive/emoji_dataset/1_annotations_rebased"

    train_and_evaluate(DATA1_IMGS, DATA1_ANN)
