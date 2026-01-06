**Made and tested for Jetson Orin Nano Super Developer Kit (with Jetpack 6.1)**


# Run Ollama

Quick Link: `https://www.jetson-ai-lab.com/tutorials/ollama/#2-docker-container-for-ollama`

Run Ollama container in terminal 1: `jetson-containers run --name ollama $(autotag ollama)` (leave hanging)

Pull Model in new terminal: `ollama run qwen2.5:0.5b` (doesn't have to be inside docker container). Optionally use `/bye` to exit model and close terminal.

Optionally test with "Open Web UI Sever": `docker run -d --network=host -v open-webui:/app/backend/data -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main`

