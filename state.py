# Application state management

import threading
import time

_lock = threading.Lock()
_state = {"detections": [], "timestamp": 0.0}


def update_state(detections):
    with _lock:
        _state["detections"] = detections
        _state["timestamp"] = time.time()


def get_state():
    with _lock:
        return dict(_state)
