#!/usr/bin/env python3
import sys
import os

# Add common ultralytics paths
sys.path.insert(0, '/ultralytics')
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    # Try alternative import
    try:
        import ultralytics
        print(f"Ultralytics location: {ultralytics.__file__}")
        from ultralytics import YOLO
    except Exception as e2:
        print(f"Alternative import failed: {e2}")
        raise

app = FastAPI()
model = YOLO("yolov8s.pt")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    results = model(img, verbose=False)
    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            detections.append({
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "xyxy": b.xyxy[0].tolist(),
            })
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
