# Main application entry point for the Jetson vision assistant

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

import llm
import state
import vision

app = FastAPI()

# Serve UI static files
ui_path = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/ui", StaticFiles(directory=ui_path), name="ui")

@app.get("/")
async def root():
    return FileResponse(os.path.join(ui_path, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    vision.start()


@app.post("/chat")
def chat(payload: dict):
    text = (payload or {}).get("text") or (payload or {}).get("message") or ""
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    scene = state.get_state()
    reply = llm.ask(text, scene)
    return {"response": reply}


def _frame_stream():
    boundary = "frame"
    while True:
        frame = vision.get_frame()
        if frame:
            yield (
                b"--"
                + boundary.encode()
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.05)


@app.get("/video")
def video():
    return StreamingResponse(
        _frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/state")
def get_state():
    return state.get_state()
