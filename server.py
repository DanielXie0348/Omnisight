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

model = YOLO("yolov8n.pt")

latest_frame = None
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

def camera_thread(): # pulls frame from camera
    global latest_frame
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame

def detection_thread(): # runs yolo on every frame
    fps_history = deque(maxlen=30) # dequeues so doesn't get built up

    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        t0 = time.perf_counter() # start time
        results = model(frame, imgsz=320, verbose=False) # measuring times
        t1 = time.perf_counter() # end time
        inference_ms = round((t1 - t0) * 1000, 1) # s to ms

        persons = []
        any_danger = False

        for result in results: # start of person detection loop
            for box in result.boxes:
                if int(box.cls[0]) != 0: # if not a person, skips the rest of the loop
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = y2 # uses foot point for measurements

                mx = int((cx / 640) * 600)
                my = int((cy / 480) * 400)
                mx = max(0, min(599, mx))
                my = max(0, min(399, my))

                in_danger = (150 <= mx <= 450) and (200 <= my <= 400)
                if in_danger:
                    any_danger = True

                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "map": [mx, my],
                    "danger": in_danger,
                    "conf": round(float(box.conf[0]), 2)
                })

        fps_history.append(t1 - t0)
        fps = round(1 / (sum(fps_history) / len(fps_history)), 1)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8') # compresses to 60% quality for websocket

        with state_lock:
            state["status"] = "DANGER" if any_danger else "SAFE"
            state["fps"] = fps
            state["inference_ms"] = inference_ms
            state["persons"] = persons
            state["frame_b64"] = frame_b64

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

print("OmniSight server running — open dashboard.html in Chrome")
asyncio.run(main())