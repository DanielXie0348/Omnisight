from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    results = model(frame, imgsz=320, verbose=False)

    top_down = np.zeros((400, 600, 3), dtype=np.uint8)

    # Danger zone — tweak these numbers after you see where your dot lands
    cv2.rectangle(top_down, (150, 200), (450, 400), (0, 0, 255), -1)
    cv2.putText(top_down, "DANGER ZONE", (210, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = y2  # foot point

            # Simple pixel scaling — no homography
            mx = int((cx / 640) * 600)
            my = int((cy / 480) * 400)
            mx = max(0, min(599, mx))
            my = max(0, min(399, my))

            in_danger = (150 <= mx <= 450) and (200 <= my <= 400)
            color = (0, 0, 255) if in_danger else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 6, color, -1)

            cv2.circle(top_down, (mx, my), 12, color, -1)
            cv2.putText(top_down, "DANGER" if in_danger else "SAFE",
                        (mx + 14, my + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS counter
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("OmniSight - Camera", frame)
    cv2.imshow("OmniSight - Map", top_down)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()