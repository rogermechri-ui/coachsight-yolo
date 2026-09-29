"""CoachSight AI - moteur de detection YOLO pour RunPod Serverless.
Recoit la video d'un match, detecte joueurs et ballon, lit les numeros de
maillot quand ils sont visibles et renvoie le tout a l'application
(callback_url)."""
import os
import tempfile
from collections import Counter

import cv2
import numpy as np
import requests
import runpod
from ultralytics import YOLO

try:
    import pytesseract
except Exception:  # OCR indisponible : le suivi continue sans numeros
    pytesseract = None

MODEL = YOLO(os.environ.get("YOLO_MODEL", "yolov8m.pt"))
PITCH_L, PITCH_W = 105.0, 68.0
SAMPLE_FPS = float(os.environ.get("SAMPLE_FPS", "2"))
READ_NUMBERS = os.environ.get("READ_NUMBERS", "1") != "0"
# Lire les numeros une image sur N : suffisant pour un vote majoritaire
# fiable par piste, sans exploser la duree du traitement.
OCR_EVERY_N = int(os.environ.get("OCR_EVERY_N", "5"))


def shirt_color(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    crop = frame[max(y1 + int(h * 0.15), 0):y1 + int(h * 0.5), max(x1, 0):x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)


def read_number(frame, box):
    """Tente de lire un numero de maillot (1 a 99) sur le torse ou le dos."""
    if pytesseract is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    # Trop petit pour etre lisible : on ne tente pas.
    if h < 30 or w < 24:
        return None
    crop = frame[y1 + int(h * 0.10):y1 + int(h * 0.55), max(x1, 0):x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    for img in (binary, cv2.bitwise_not(binary)):
        try:
            txt = pytesseract.image_to_string(img, config=config).strip()
        except Exception:
            return None
        if txt.isdigit() and 1 <= int(txt) <= 99:
            return int(txt)
    return None


def process(inp):
    url, start, end = inp["video_url"], float(inp.get("start_s", 0)), float(inp["end_s"])
    fps_out = min(float(inp.get("fps", SAMPLE_FPS)), SAMPLE_FPS)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    cap = cv2.VideoCapture(tmp)
    raw = []      # (t, [(trackId, x, y, ball, color)])
    votes = {}    # trackId -> numeros lus (vote majoritaire en fin de traitement)
    frame_i = 0
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = MODEL.track(frame, persist=True, classes=[0, 32], conf=0.25, verbose=False)[0]
        pts = []
        if res.boxes is not None:
            ids = res.boxes.id.tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for box, cls, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), ids):
                ball = int(cls) == 32
                cx, cy = (box[0] + box[2]) / 2, box[3] if not ball else (box[1] + box[3]) / 2
                col = None if ball else shirt_color(frame, box)
                tid_i = int(tid) if tid is not None else -1
                if (not ball and READ_NUMBERS and pytesseract is not None
                        and tid_i >= 0 and frame_i % OCR_EVERY_N == 0):
                    n = read_number(frame, box)
                    if n is not None:
                        votes.setdefault(tid_i, Counter()).update([n])
                pts.append((tid_i, cx / w * PITCH_L, cy / h * PITCH_W, ball, col))
        raw.append((round(t, 2), pts))
        frame_i += 1
        t += 1.0 / fps_out
    cap.release(); os.remove(tmp)

    cols = np.array([p[4] for _, pts in raw for p in pts if p[4] is not None], dtype=np.float32)
    centers = None
    if len(cols) >= 4:
        _, _, centers = cv2.kmeans(cols, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    # Numero retenu par piste : celui le plus souvent lu (vote majoritaire).
    numbers = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    tracks = []
    seen = {}
    frames = []
    for tt, pts in raw:
        out = []
        for tid, x, y, ball, col in pts:
            team = None
            if col is not None and centers is not None:
                team = int(np.argmin(np.linalg.norm(centers - col, axis=1)))
            num = None if ball else numbers.get(tid)
            out.append({
                "trackId": tid, "team": team,
                "x": round(x, 2), "y": round(y, 2),
                "ball": ball,
                "number": num,
            })
            if not ball and tid >= 0 and tid not in seen:
                seen[tid] = True
                tracks.append({"track_id": tid, "team": team, "number": num})
        frames.append({"t": tt, "points": out})
    return {"simulated": False, "pitch": {"length": PITCH_L, "width": PITCH_W}, "tracks": tracks, "frames": frames}


def handler(job):
    inp = job["input"]
    cb = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
    try:
        result = process(inp)
        cb.update(status="completed", result=result, stats={"frames": len(result["frames"]), "tracks": len(result.get("tracks", []))})
    except Exception as e:
        cb.update(status="failed", error=str(e)[:900])
    requests.post(inp["callback_url"], json=cb, timeout=120)
    return {"status": cb["status"], "frames": len(cb.get("result", {}).get("frames", []))}


runpod.serverless.start({"handler": handler})
