from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

# Homography points - we'll calibrate these properly later
camera_points = np.float32([
    [0,   100],
    [640, 100],
    [640, 480],
    [0,   480],
])

map_points = np.float32([
    [0,   0],
    [600, 0],
    [600, 400],
    [0,   400],
])

H = cv2.getPerspectiveTransform(camera_points, map_points)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 15)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:  # only process every 3rd frame
        continue

    small = cv2.resize(frame, (320, 240))
    results = model(small, imgsz=160, verbose=False)

    # Create blank top down map
    top_down = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw danger zone
    cv2.rectangle(top_down, (200, 150), (400, 300), (0, 0, 255), -1)
    cv2.putText(top_down, "DANGER ZONE", (210, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # 0 = person
                continue

            # Get bounding box center
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Transform to map coordinates
            pt = np.float32([[[cx, cy]]])
            mapped = cv2.perspectiveTransform(pt, H)
            mx, my = int(mapped[0][0][0]), int(mapped[0][0][1])

            # Check danger zone
            in_danger = (200 <= mx <= 400) and (150 <= my <= 300)
            color = (0, 0, 255) if in_danger else (0, 255, 0)

            # Draw on camera feed
            cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)
            cv2.circle(small, (cx, cy), 5, color, -1)

            # Draw on map
            mx = max(0, min(599, mx))
            my = max(0, min(399, my))
            cv2.circle(top_down, (mx, my), 10, color, -1)

            status = "DANGER" if in_danger else "SAFE"
            cv2.putText(top_down, status, (mx + 12, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("OmniSight - Camera", small)
    cv2.imshow("OmniSight - Map", top_down)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()