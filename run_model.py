import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------
# Load Model
# -----------------------
model = YOLO(
    r"./train11/weights/best.pt"
)#F:/Object-Detection-in-Adverse-Weather-Conditions/runs/detect/train11/weights/best.pt"

# Use GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# Load Video (FIXED PATH)
# -----------------------
cap = cv2.VideoCapture(r"D:\ACDC\snow_driving.mp4")#r"D:/Test clips/City2.mp4" "D:/ACDC/rain_driving.mp4"

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow("Img", cv2.WINDOW_NORMAL)

# -----------------------
# Video Loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, device=device, conf=0.4)

    result = results[0]

    # Draw boxes + labels manually
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Put label text
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Img", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()