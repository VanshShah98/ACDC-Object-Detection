import json
import os
import shutil

ACDC_ROOT = r"D:\ACDC"
YOLO_ROOT = "data"

# create YOLO folders
os.makedirs(os.path.join(YOLO_ROOT, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_ROOT, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(YOLO_ROOT, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_ROOT, "labels", "val"), exist_ok=True)

def convert(coco_json, split):
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    ann_map = {}

    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for img_id, img in images.items():
        if img_id not in ann_map:
            continue

        rel_path = img["file_name"]
        src_img = os.path.join(ACDC_ROOT, "rgb_anon", rel_path)

        if not os.path.exists(src_img):
            print("❌ Image not found:", src_img)
            continue

        h, w = img["height"], img["width"]

        img_name = os.path.basename(rel_path)
        dst_img = os.path.join(YOLO_ROOT, "images", split, img_name)
        shutil.copy2(src_img, dst_img)

        label_path = os.path.join(
            YOLO_ROOT, "labels", split, img_name.replace(".png", ".txt")
        )

        with open(label_path, "w") as f:
            for ann in ann_map[img_id]:
                x, y, bw, bh = ann["bbox"]

                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"✅ {split} conversion done")

if __name__ == "__main__":
    convert("acdc_filtered/train_car.json", "train")
    convert("acdc_filtered/val_car.json", "val")
