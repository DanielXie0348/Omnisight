from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
import asyncio
import websockets
import json
import base64
from collections import deque
import serial

model = YOLO("yolov8n.pt") # Start of setup section
model.fuse()

H = np.load("homography_matrix.npy")

ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)
print("Arduino connected")
# -----------------------------------------------------------------

latest_frame = None # shared state section (important vals)
frame_lock = threading.Lock()
state_lock = threading.Lock()
clients = set()

state = {
    "status": "SAFE",
    "fps": 0,
    "inference_ms": 0,
    "persons": [],
    "frame_b64": None,
}

# --------------------------------------

def camera_thread(): # camera thread, for connection between devices
    global latest_frame
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Camera started")
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame

# ----------------------------------------------

def detection_thread(): # full detection thread for detecting a person in the danger zone
    fps_history = deque(maxlen=30)
    last_cmd = None

    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        t0 = time.perf_counter()
        results = model(frame, imgsz=256, verbose=False)
        t1 = time.perf_counter()
        inference_ms = round((t1 - t0) * 1000, 1) 

        persons = []
        any_danger = False

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = y2

                # Homography transform
                pt = np.float32([[[cx, cy]]])
                mapped = cv2.perspectiveTransform(pt, H)
                mx = float(mapped[0][0][0])
                my = float(mapped[0][0][1])

                # filters out readings outside the calibration area so it does not interfere
                if mx < -5 or mx > 38 or my < -5 or my > 23:
                    continue

                # setting the official danger zone
                in_danger = (0 <= mx <= 4) and (18 <= my <= 22)

                if in_danger:
                    any_danger = True

                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "map": [round(mx, 1), round(my, 1)],
                    "danger": in_danger,
                    "conf": round(float(box.conf[0]), 2)
                })

        # arduino block for moving the servo to s (stop) or g (go)
        cmd = 'S' if any_danger else 'G'
        if cmd != last_cmd:
            ser.write(cmd.encode())
            last_cmd = cmd

        fps_history.append(t1 - t0)
        fps = round(1 / (sum(fps_history) / len(fps_history)), 1)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        # debug
        for p in persons:
            print(f"  mx={p['map'][0]}, my={p['map'][1]} → {'DANGER' if p['danger'] else 'SAFE'}")

        with state_lock: # posting information on the UI (html)
            state["status"] = "DANGER" if any_danger else "SAFE"
            state["fps"] = fps
            state["inference_ms"] = inference_ms
            state["persons"] = persons
            state["frame_b64"] = frame_b64

# for handling the websocket setup
async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    except:
        pass
    finally:
        clients.discard(websocket)

async def broadcast():
    while True:
        with state_lock:
            msg = json.dumps(state)
        dead = set()
        for ws in list(clients):
            try:
                await ws.send(msg)
            except:
                dead.add(ws)
        for ws in dead:
            clients.discard(ws)
        await asyncio.sleep(1/30)

async def main():
    async with websockets.serve(ws_handler, "localhost", 8765):
        await broadcast()

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=detection_thread, daemon=True).start()
time.sleep(1)

print("\nOmniSight running — open dashboard.html in Chrome\n")
asyncio.run(main())