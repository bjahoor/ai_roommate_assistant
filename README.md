**Made and tested for Jetson Orin Nano Super Developer Kit (with Jetpack 6.1)**

---

# Compressed Demo Video

[Watch Demo](https://raw.githubusercontent.com/bjahoor/ai_roommate_assistant/main/Ai%20Room%20Assistant%20Demo.mp4)

---

# System Architecture

<img width="404" height="543" alt="image" src="https://github.com/user-attachments/assets/e807d2cb-9c06-4c71-b28f-db987eb4a182" />

---

# Run Ollama

Quick Link: `https://www.jetson-ai-lab.com/tutorials/ollama/#2-docker-container-for-ollama`

Run Ollama container in terminal 1: `jetson-containers run --name ollama $(autotag ollama)` (leave hanging)

Pull Model in new terminal: `ollama run qwen2.5:0.5b` (doesn't have to be inside docker container). Optionally use `/bye` to exit model and close terminal.

Optionally test with "Open Web UI Sever": `docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main`

# Run YOLO

Terminal 2: `docker rm -f yolo 2>/dev/null; t=ultralytics/ultralytics:latest-jetson-jetpack6 && sudo docker pull $t && sudo docker run -d --name yolo --ipc=host --runtime=nvidia -p 8000:8000 -v $(pwd)/yolo_server.py:/yolo_server.py -w /ultralytics $t bash -c "pip install fastapi uvicorn python-multipart && python3 /yolo_server.py"`

# Run FastAPI

Can be run in terminal 2: `python3 -m uvicorn app:app --host 0.0.0.0 --port 8001` (leave hannging)

# Run UI Sever

Terminal 3: `cd ui/` then run `python3 -m http.server 8002` and type `http://localhost:8002/` in browser
