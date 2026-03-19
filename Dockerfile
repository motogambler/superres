# Simple CPU-based image demo container. For GPU, base on nvidia/cuda and install proper drivers and runtimes.
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 7860
CMD ["python", "app.py"]

# NVIDIA GPU note:
# To build a GPU-enabled image, start FROM nvidia/cuda:12.1-runtime-ubuntu22.04, install python, pip, and the CUDA-aware packages
# and use `nvidia-container-toolkit` when running the container.
