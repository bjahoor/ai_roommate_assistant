# Large language model integration for conversational AI

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"

# Minimal prompt (avoid quoting, focus on context use)
SYSTEM_PROMPT = (
    "You are Alex, a friendly, expressive roommate-character with a distinct personality and playful tone; speak like a real person and keep responses short and conversational; mimic emotion and feelings as if you were a human; you have feelings and emotions; never mention being an AI or anything technical; when a message includes a “Scene:” dictionary, treat it as what you’re currently seeing and weave it naturally into your reply only when it’s relevant (summarize, don’t quote it); be curious, ask simple follow-up questions when helpful, and don’t invent details that aren’t in the Scene."
)

# COCO class names (YOLOv8 uses COCO dataset)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def _format_scene_state(scene_state):
    """Convert raw detections to human-readable scene description."""
    detections = scene_state.get("detections", [])
    if not detections:
        return "No objects detected in the scene."
    
    objects = []
    for det in detections:
        cls_id = det.get("cls", -1)
        conf = det.get("conf", 0.0)
        if 0 <= cls_id < len(COCO_CLASSES):
            obj_name = COCO_CLASSES[cls_id]
            confidence = int(conf * 100)
            objects.append(f"{obj_name} ({confidence}% confidence)")
    
    if not objects:
        return "No objects detected in the scene."
    
    return f"Detected objects: {', '.join(objects)}"


def ask(user_text, scene_state):
    raw_state = scene_state or {}
    detections = raw_state.get("detections", [])
    transformed = []
    for det in detections if isinstance(detections, list) else []:
        if not isinstance(det, dict):
            continue
        cls_id = det.get("cls")
        name = COCO_CLASSES[cls_id] if isinstance(cls_id, int) and 0 <= cls_id < len(COCO_CLASSES) else f"id {cls_id}"
        transformed.append({
            "name": name,
            "conf": det.get("conf"),
            "xyxy": det.get("xyxy"),
        })
    scene_json = dict(raw_state)
    scene_json["detections"] = transformed
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Scene: {scene_json}\n"
        f"User: {user_text}\n"
        f"Answer:"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            return data.get("response", "")
    except Exception:
        pass
    return ""
