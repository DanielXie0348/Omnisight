import cv2
import numpy as np

FLOOR_WIDTH_CM  = 36
FLOOR_HEIGHT_CM = 22
CAMERA_INDEX    = 1

real_world_pts = np.float32([
    [0,              0             ],
    [FLOOR_WIDTH_CM, 0             ],
    [FLOOR_WIDTH_CM, FLOOR_HEIGHT_CM],
    [0,              FLOOR_HEIGHT_CM],
])

clicked = []
img_display = None

LABELS = ["1 - Top Left", "2 - Top Right", "3 - Bottom Right", "4 - Bottom Left"]
COLORS = [(0,255,255), (0,180,255), (0,80,255), (255,80,0)]

def click(event, x, y, flags, param):
    global clicked, img_display
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))
        i = len(clicked) - 1
        cv2.circle(img_display, (x, y), 10, COLORS[i], -1)
        cv2.circle(img_display, (x, y), 14, (255,255,255), 2)
        cv2.putText(img_display, LABELS[i], (x+16, y+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
        print(f"  ✓ {LABELS[i]}: ({x}, {y})")
        if len(clicked) == 4:
            save()

def save():
    src = np.float32(clicked)
    H, _ = cv2.findHomography(src, real_world_pts)
    np.save("homography_matrix.npy", H)
    print("\n✓ Saved to homography_matrix.npy")
    print(f"Matrix:\n{H}")
    print("\nPress any key to exit.")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("OmniSight Calibration")
print(f"Desk: {FLOOR_WIDTH_CM}cm x {FLOOR_HEIGHT_CM}cm")
print("Click: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
print("R to reset, Q to quit\n")

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if len(clicked) < 4:
        img_display = frame.copy()
        for i, pt in enumerate(clicked):
            cv2.circle(img_display, pt, 10, COLORS[i], -1)
            cv2.circle(img_display, pt, 14, (255,255,255), 2)
            cv2.putText(img_display, LABELS[i], (pt[0]+16, pt[1]+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)

    if len(clicked) < 4:
        cv2.putText(img_display, f"Click {LABELS[len(clicked)]}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[len(clicked)], 2)
    else:
        cv2.putText(img_display, "DONE — press any key to exit",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.putText(img_display, "R=reset  Q=quit", (15, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    cv2.imshow("Calibration", img_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        clicked = []
        print("Reset — click 4 points again")
    elif key != 255 and len(clicked) == 4:
        break

cap.release()
cv2.destroyAllWindows()