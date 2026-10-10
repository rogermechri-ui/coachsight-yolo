FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tesseract-ocr ffmpeg && rm -rf /var/lib/apt/lists/*
# Donne acces a la puce video du GPU (encodage NVENC) pour la copie allegee.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
COPY handler.py pitch_fit.py pitch_track.py botsort_coachsight.yaml ./
CMD ["python", "-u", "handler.py"]
