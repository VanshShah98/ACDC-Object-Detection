import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms.functional import to_tensor

from sklearn.metrics import precision_recall_curve
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = r"D:\ACDC\rgb_anon"
VAL_JSON = r"D:\ACDC\acdc_filtered\val_car.json"
MODEL_PATH = r"R_CNN\faster_rcnn_acdc_fast.pth"
OUT_DIR = "R_CNN"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"🚀 Evaluating on: {DEVICE}")

# =========================================================
# DATASET
# =========================================================
class CocoValDataset(CocoDetection):
    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        img = to_tensor(img)
        return img, anns


val_dataset = CocoValDataset(DATA_ROOT, VAL_JSON)

# =========================================================
# MODEL
# =========================================================
model = fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

print("🧠 Model loaded successfully")

# =========================================================
# INFERENCE
# =========================================================
y_true, y_scores = [], []
coco_results = []

coco_gt = COCO(VAL_JSON)

for idx in tqdm(range(len(val_dataset)), desc="Running inference"):
    img, gt_anns = val_dataset[idx]
    img = img.to(DEVICE)

    image_id = val_dataset.ids[idx]  # 🔥 COCO-CORRECT ID

    with torch.no_grad():
        outputs = model([img])[0]

    scores = outputs["scores"].cpu().numpy()
    boxes = outputs["boxes"].cpu().numpy()

    gt_present = len(gt_anns) > 0

    for score in scores:
        y_scores.append(score)
        y_true.append(1 if gt_present else 0)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        coco_results.append({
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "score": float(score),
        })


# =========================================================
# PRECISION / RECALL / F1
# =========================================================
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

# =========================================================
# SAVE GRAPHS
# =========================================================
plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Faster R-CNN)")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "PR_curve_R_CNN.png"))
plt.close()

plt.figure()
plt.plot(thresholds, f1[:-1])
plt.xlabel("Confidence Threshold")
plt.ylabel("F1-score")
plt.title("F1-score vs Confidence (Faster R-CNN)")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "F1_curve_R_CNN.png"))
plt.close()

# =========================================================
# COCO mAP@0.5
# =========================================================
with open(os.path.join(OUT_DIR, "coco_results.json"), "w") as f:
    json.dump(coco_results, f)

coco_dt = coco_gt.loadRes(os.path.join(OUT_DIR, "coco_results.json"))
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

coco_eval.params.iouThrs = [0.5]  # mAP@0.5 only
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

map_50 = coco_eval.stats[0]

# =========================================================
# SAVE METRICS
# =========================================================
metrics = {
    "precision_mean": float(np.mean(precision)),
    "recall_mean": float(np.mean(recall)),
    "f1_max": float(np.max(f1)),
    "mAP@0.5": float(map_50),
}

with open(os.path.join(OUT_DIR, "eval_metrics_R_CNN.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Evaluation completed successfully")
print("📊 Metrics & graphs saved in R_CNN/")
