# Computer vision module for camera and YOLO processing

import threading
import time

import cv2
import requests
import state

YOLO_URL = "http://localhost:8000/predict"
CAP_WIDTH = 640
CAP_HEIGHT = 480
latest_jpeg = None
_started = False
_busy = False
_busy_lock = threading.Lock()


def _send_worker(jpg):
    global _busy
    try:
        resp = requests.post(
            YOLO_URL,
            files={"image": ("frame.jpg", jpg, "image/jpeg")},
            timeout=1.5,
        )
        if resp.ok:
            data = resp.json()
            detections = (
                data.get("detections")
                or data.get("boxes")
                or data.get("data")
                or []
            )
            if not isinstance(detections, list):
                detections = []
            state.update_state(detections)
    except Exception:
        pass
    finally:
        with _busy_lock:
            _busy = False


def _try_send_async(jpg):
    global _busy
    with _busy_lock:
        if _busy:
            return
        _busy = True
    threading.Thread(target=_send_worker, args=(jpg,), daemon=True).start()


def _loop():
    global latest_jpeg
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg = buf.tobytes()
            latest_jpeg = jpg
            _try_send_async(jpg)
            time.sleep(0.01)
    finally:
        cap.release()


def start():
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_loop, daemon=True).start()


def get_frame():
    return latest_jpeg

