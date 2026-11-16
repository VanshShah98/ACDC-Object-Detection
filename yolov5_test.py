import torch
import cv2

# Load YOLOv5 custom model
model = torch.hub.load(
    'D:/ACDC/yolov5',   # path to YOLOv5 repo
    'custom',
    path='D:/ACDC/yolov5/runs/train/acdc_yolov5/weights/best.pt',
    source='local'
)

# Open video
cap = cv2.VideoCapture(r"D:\ACDC\rain_driving.mp4")

if not cap.isOpened():
    print("Error opening video")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame)

    # Get detections
    detections = results.xyxy[0]

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)

        label = f"{model.names[int(cls)]} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        # Draw label
        cv2.putText(frame,
                    label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

    # Show video
    cv2.imshow("YOLOv5 Detection", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()