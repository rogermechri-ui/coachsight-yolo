"""CoachSight AI - moteur de detection YOLO pour RunPod Serverless.
Recoit la video d'un match, detecte joueurs et ballon, lit les numeros de
maillot quand ils sont visibles et renvoie le tout a l'application
(callback_url)."""
import io
import json
import os
import tempfile
import zipfile
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
# Images envoyees a l'application pour construire stats et rapport cote
# serveur (le navigateur du coach n'a plus a relire la video).
FRAMES_EVERY_S = float(os.environ.get("FRAMES_EVERY_S", "12"))
FRAMES_WIDTH = int(os.environ.get("FRAMES_WIDTH", "960"))
FRAMES_QUALITY = int(os.environ.get("FRAMES_QUALITY", "70"))
# Lecture continue de la video pendant le tracking (plus rapide que de
# "sauter" a chaque image). Mettre SEQUENTIAL_READ=0 pour revenir a l'ancien mode.
SEQUENTIAL_READ = os.environ.get("SEQUENTIAL_READ", "1") != "0"


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


class FrameCollector:
    """Garde une petite image JPEG toutes les `every_s` secondes, puis les
    envoie en un seul zip (avec un index manifest.json) vers une URL signee."""

    def __init__(self, inp):
        self.url = inp.get("frames_upload_url")
        self.every = float(inp.get("frames_every_s", FRAMES_EVERY_S))
        self.width = int(inp.get("frames_width", FRAMES_WIDTH))
        self.quality = int(inp.get("frames_quality", FRAMES_QUALITY))
        self.buf = io.BytesIO()
        self.zip = zipfile.ZipFile(self.buf, "w", zipfile.ZIP_STORED)
        self.index = []
        self.last_t = None

    @property
    def enabled(self):
        return bool(self.url) and self.every > 0

    def maybe_add(self, t, frame):
        if not self.enabled or (self.last_t is not None and t - self.last_t < self.every - 1e-6):
            return
        self.last_t = t
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        name = "frame_%09d.jpg" % int(round(t * 1000))
        self.zip.writestr(name, jpg.tobytes())
        self.index.append({"t": round(t, 2), "file": name})

    def upload(self):
        """Envoie le zip. Une erreur ici ne fait jamais echouer le tracking."""
        if not self.enabled:
            return {"uploaded": False, "count": 0, "reason": "no frames_upload_url"}
        info = {"uploaded": False, "count": len(self.index), "every_s": self.every, "width": self.width}
        try:
            self.zip.writestr("manifest.json", json.dumps({"frames": self.index}))
            self.zip.close()
            data = self.buf.getvalue()
            info["bytes"] = len(data)
            r = requests.put(self.url, data=data, timeout=600,
                             headers={"Content-Type": "application/zip", "x-upsert": "true"})
            r.raise_for_status()
            info["uploaded"] = True
        except Exception as e:
            info["error"] = str(e)[:300]
        return info


def iter_frames(cap, start, end, fps_out):
    """Donne (t, image) toutes les 1/fps_out secondes entre start et end.
    Mode continu : on se place une seule fois au debut puis on lit la video
    dans l'ordre (grab sans decodage complet des images ignorees)."""
    step = 1.0 / fps_out
    vfps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not SEQUENTIAL_READ or not (1 <= vfps <= 240):
        t = start
        while t <= end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                return
            yield t, frame
            t += step
        return
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    next_t = start
    t0 = None
    n = 0
    while True:
        if not cap.grab():
            return
        pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t0 is None:
            t0 = pos if pos > 0 else start
        t = pos if pos > 0 else t0 + n / vfps
        n += 1
        if t > end + 1e-6:
            return
        if t + 0.5 / vfps < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            return
        yield next_t, frame
        next_t += step


def collect_frames(cap, inp, start, end):
    """Passe rapide : une image toutes les `frames_every_s` secondes, puis envoi."""
    collector = FrameCollector(inp)
    if not collector.enabled:
        return collector.upload()
    t = start
    while t <= end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        collector.maybe_add(t, frame)
        t += collector.every
    return collector.upload()


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
    # Images pour stats/rapport : extraites et envoyees AVANT le tracking,
    # pour que le lien d'envoi (valable ~2 h) ne soit jamais expire.
    frames_zip = collect_frames(cap, inp, start, end)
    if inp.get("early_frames_callback") and frames_zip.get("uploaded"):
        notify(inp, status="frames_ready", frames_zip=frames_zip)
    for t, frame in iter_frames(cap, start, end, fps_out):
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
    # Couleur moyenne du maillot de chaque groupe (0 et 1), pour que
    # l'application relie chaque groupe a l'equipe du coach ou a l'adversaire.
    team_colors = []
    if centers is not None:
        for i, lab in enumerate(centers):
            px = np.uint8([[np.clip(lab, 0, 255)]])
            b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0][0]
            team_colors.append({"team": i, "hex": "#%02x%02x%02x" % (r, g, b)})
    ball_frames = sum(1 for f in frames if any(p["ball"] for p in f["points"]))
    return {"simulated": False, "pitch": {"length": PITCH_L, "width": PITCH_W}, "tracks": tracks, "frames": frames,
            "frames_zip": frames_zip, "team_colors": team_colors,
            "ball_frames": ball_frames, "total_frames": len(frames)}


def notify(inp, **fields):
    """Message intermediaire vers l'application (n'interrompt jamais le travail)."""
    try:
        body = {"job_id": inp["job_id"], "callback_token": inp["callback_token"]}
        body.update(fields)
        requests.post(inp["callback_url"], json=body, timeout=60)
    except Exception:
        pass


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
