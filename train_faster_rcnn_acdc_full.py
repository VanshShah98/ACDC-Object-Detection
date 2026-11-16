import os
import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"D:\ACDC\rgb_anon"
TRAIN_JSON = r"D:\ACDC\acdc_filtered\train_car.json"

OUTPUT_DIR = "R_CNN"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 2        # background + car
BATCH_SIZE = 2         # increase to 4 if VRAM allows
EPOCHS = 10            # Faster convergence
LR = 0.005
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# =========================================================
# AUGMENTATION (FAST & SAFE)
# =========================================================
transform = A.Compose(
    [
        A.Resize(512, 512),              # 🔥 MAJOR SPEEDUP
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3
    ),
)

# =========================================================
# DATASET
# =========================================================
class CocoCarDataset(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]

        img_np = np.array(img.convert("RGB"))

        aug = transform(image=img_np, bboxes=boxes, labels=labels)

        image = aug["image"]
        target = {
            "boxes": torch.tensor(aug["bboxes"], dtype=torch.float32),
            "labels": torch.tensor(aug["labels"], dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================================================
# MAIN (WINDOWS SAFE)
# =========================================================
def main():
    print(f"🚀 Training on: {DEVICE}")

    dataset = CocoCarDataset(DATA_ROOT, TRAIN_JSON)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    print(f"📦 Training images: {len(dataset)}")

    # =====================================================
    # MODEL
    # =====================================================
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 🔥 Replace predictor correctly
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # 🔥 SPEED OPTIMIZATIONS
    model.rpn.pre_nms_top_n_train = 1000
    model.rpn.post_nms_top_n_train = 500
    model.rpn.pre_nms_top_n_test = 500
    model.rpn.post_nms_top_n_test = 300
    model.roi_heads.detections_per_img = 50

    # 🔥 FREEZE BACKBONE
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.to(DEVICE)
    print("🧠 Model device:", next(model.parameters()).device)

    # =====================================================
    # OPTIMIZER + AMP
    # =====================================================
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    scaler = torch.cuda.amp.GradScaler()

    # =====================================================
    # METRICS LOG
    # =====================================================
    metrics_csv = os.path.join(OUTPUT_DIR, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "avg_loss"])

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_loss])

        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

    # =====================================================
    # SAVE MODEL
    # =====================================================
    model_path = os.path.join(OUTPUT_DIR, "faster_rcnn_acdc_fast.pth")
    torch.save(model.state_dict(), model_path)

    print("\n🎉 Training completed successfully")
    print(f"💾 Model saved to: {model_path}")
    print(f"📊 Metrics saved to: {metrics_csv}")


# =========================================================
# ENTRY POINT (REQUIRED ON WINDOWS)
# =========================================================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
