import json
import os

ACDC_ROOT = r"D:\ACDC"   # 👈 this matches your setup

def filter_coco_to_car(in_json, out_json):
    with open(in_json, "r") as f:
        coco = json.load(f)

    # find car category id
    car_ids = [c["id"] for c in coco["categories"] if c["name"] == "car"]
    if not car_ids:
        raise RuntimeError("Car category not found")

    car_id = car_ids[0]

    new_annotations = []
    image_ids = set()

    for ann in coco["annotations"]:
        if ann["category_id"] == car_id:
            new_annotations.append(ann)
            image_ids.add(ann["image_id"])

    new_images = [img for img in coco["images"] if img["id"] in image_ids]
    new_categories = [c for c in coco["categories"] if c["id"] == car_id]

    filtered = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(filtered, f)

    print(f"Saved {len(new_images)} images, {len(new_annotations)} annotations")

if __name__ == "__main__":
    filter_coco_to_car(
        os.path.join(ACDC_ROOT,"gt_detection" ,"gt_detection", "instancesonly_train_gt_detection.json"),
        "acdc_filtered/train_car.json"
    )

    filter_coco_to_car(
        os.path.join(ACDC_ROOT,"gt_detection" ,"gt_detection", "instancesonly_val_gt_detection.json"),
        "acdc_filtered/val_car.json"
    )
