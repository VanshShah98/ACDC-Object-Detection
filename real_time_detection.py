import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='D:/ACDC/yolov5/runs/train/acdc_yolov5/weights/best.pt',
    #source='local'
)

# Open video
cap = cv2.VideoCapture(r"D:\Test clips\City.mp4")

# Check if video opened
if not cap.isOpened():
    print("Error opening video file")

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Convert results
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        label = f"{model.names[int(cls)]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label
        cv2.putText(frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    # Show frame
    cv2.imshow("YOLOv5 Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()