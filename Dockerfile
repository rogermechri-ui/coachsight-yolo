FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tesseract-ocr ffmpeg fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
# Donne acces a la puce video du GPU (encodage NVENC) pour la copie allegee.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt
# Liste des versions reellement installees, visible dans le journal de construction
# (sert a figer les versions a l'identique ; ne change rien au fonctionnement).
RUN echo "PIP_FREEZE_BEGIN" && python --version && pip freeze && echo "PIP_FREEZE_END" && (ffmpeg -version | head -1)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
COPY safe.py handler.py pitch_fit.py pitch_track.py actors.py hot.py export.py ball.py botsort_coachsight.yaml ./
CMD ["python", "-u", "handler.py"]
