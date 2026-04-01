from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO at 320x320 for speed
    results = model(frame, imgsz=320, verbose=False)

    # Draw boxes on frame
    annotated = results[0].plot()

    cv2.imshow("OmniSight", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()