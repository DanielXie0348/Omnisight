import cv2
import numpy as np

camera_points = np.float32([ # defines fake camera points for Testing purpose
    [100, 200],  # top-left
    [540, 200],  # top-right
    [640, 400],  # bottom-right
    [0,   400],  # bottom-left
])

map_points = np.float32([ # mapping to these Map points in 2-D space
    [0,   0],    # top-left
    [600, 0],    # top-right
    [600, 400],  # bottom-right
    [0,   400],  # bottom-left
])

H = cv2.getPerspectiveTransform(camera_points, map_points)

person_positions_camera = [
    [100, 300],
    [200, 300],
    [320, 300],
    [440, 300],
    [540, 300],
]

top_down_map = np.zeros((400, 600, 3), dtype=np.uint8)

cv2.rectangle(top_down_map, (400, 150), (600, 400), (0, 0, 255), -1)
cv2.putText(top_down_map, "DANGER ZONE", (410, 280), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

for pos in person_positions_camera:
    pt = np.float32([[pos]]) 
    mapped = cv2.perspectiveTransform(pt, H)
    x, y = int(mapped[0][0][0]), int(mapped[0][0][1])
    
    # Check if in danger zone
    in_danger = (400 <= x <= 600) and (150 <= y <= 400)
    color = (0, 0, 255) if in_danger else (0, 255, 0)
    
    cv2.circle(top_down_map, (x, y), 10, color, -1)
    print(f"Camera {pos} → Map ({x}, {y}) {'⚠️ DANGER' if in_danger else '✅ Safe'}")

cv2.imshow("Top-Down Map", top_down_map)
cv2.waitKey(0)
cv2.destroyAllWindows()