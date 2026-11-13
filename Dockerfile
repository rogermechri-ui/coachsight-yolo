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
# Le moteur tourne sans droits administrateur (lot 3b, partie 2) : un fichier ou un
# logiciel piege ne pourrait ni modifier le systeme ni le code installe. Le compte
# peut lire /app (modele compris) ; ses ecritures vont dans /tmp et son dossier personnel.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin coach
ENV HOME=/home/coach YOLO_CONFIG_DIR=/tmp/Ultralytics
USER coach
CMD ["python", "-u", "handler.py"]
